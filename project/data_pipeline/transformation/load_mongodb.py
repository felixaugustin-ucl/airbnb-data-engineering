"""
load_mongodb.py
---------------
ELT pipeline: loads neighbourhood boundary polygons and Google Places POI data
into MongoDB as the operational geospatial store.

Architecture justification (ELT pattern):
  Data is loaded into MongoDB first in its raw document form, then lightly
  enriched via MongoDB aggregation pipelines after loading. This schema-on-read
  approach allows the document model to evolve without migration scripts —
  appropriate for the variable-field structure of POI data.

Collections:
  neighbourhoods  — GeoJSON MultiPolygon boundaries per neighbourhood
                    2dsphere index enables $geoWithin / $geoIntersects queries
  places          — POI documents with GeoJSON Point geometry
                    2dsphere index enables $nearSphere proximity queries

GeoJSON justification:
  All geometries are stored as RFC 7946 GeoJSON (Butler et al., 2016).
  MongoDB's geospatial operators require GeoJSON-conformant geometry objects.
  GeoJSON is also the standard interchange format for mapping tools
  (Leaflet, Mapbox, QGIS), ensuring interoperability beyond this project.

Usage:
    python load_mongodb.py

Environment variables (from .env):
    MONGO_URI  — MongoDB connection string (default: mongodb://localhost:27017)
    MONGO_DB   — database name (default: airbnb_geo)
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
import pymongo
from pymongo import MongoClient, GEOSPHERE
from shapely.geometry import shape, mapping
from shapely.validation import make_valid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT         = DATA_PIPELINE_DIR.parents[1]

RAW_AIRBNB_DIR    = DATA_PIPELINE_DIR / "raw" / "airbnb"
RAW_PLACES_DIR    = DATA_PIPELINE_DIR / "raw" / "places"

LINEAGE_PATH      = DATA_PIPELINE_DIR / "processed" / "mongodb" / "lineage.json"


# ── MongoDB connection ─────────────────────────────────────────────────────────

def get_db():
    uri    = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB",  "airbnb_geo")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    # Ping to verify connection before proceeding
    client.admin.command("ping")
    log.info(f"Connected to MongoDB at {uri}, database: {db_name}")
    return client[db_name]


# ── Snapshot detection ─────────────────────────────────────────────────────────

def latest_snapshot() -> Path:
    snapshots = sorted(RAW_AIRBNB_DIR.glob("up_to_*"))
    if not snapshots:
        raise FileNotFoundError(f"No Airbnb snapshots found in {RAW_AIRBNB_DIR}")
    return snapshots[-1]


# ── Neighbourhood boundaries ───────────────────────────────────────────────────

def load_neighbourhoods(db, snapshot: Path) -> dict:
    """
    Load neighbourhood boundary polygons from neighbourhoods.geojson into
    the 'neighbourhoods' collection.

    Each document stores the neighbourhood name, borough, and the full
    GeoJSON geometry (MultiPolygon) as required by MongoDB's 2dsphere index.

    A computed centroid (GeoJSON Point) is also stored for quick lookups
    without needing to compute it from the polygon at query time.

    Returns a lineage record.
    """
    geojson_path = snapshot / "neighbourhoods.geojson"
    log.info(f"Loading neighbourhoods from {geojson_path}")

    with open(geojson_path) as f:
        fc = json.load(f)

    collection = db["neighbourhoods"]

    # Drop and recreate for idempotency — ensures clean state on re-run
    collection.drop()

    documents = []
    for feature in fc["features"]:
        props    = feature["properties"]
        # Repair geometry and compute centroid in one Shapely pass.
        # Source data contains duplicate vertices and self-intersecting rings
        # that fail MongoDB 2dsphere; make_valid() resolves all GEOS errors.
        geometry, (centroid_lon, centroid_lat) = _repair_geometry(feature["geometry"])

        doc = {
            # Identifiers
            "neighbourhood": props["neighbourhood"],
            "borough":       props["neighbourhood_group"],

            # Full boundary polygon — stored as GeoJSON for 2dsphere queries
            # MongoDB requires the field be named "geometry" or a custom field
            # with a 2dsphere index for geospatial operators
            "geometry": geometry,   # RFC 7946 MultiPolygon

            # Pre-computed centroid point — useful for $nearSphere proximity
            "centroid": {
                "type":        "Point",
                "coordinates": [centroid_lon, centroid_lat]
            },

            # Metadata
            "loaded_at": datetime.now(timezone.utc).isoformat(),
        }
        documents.append(doc)

    collection.insert_many(documents)
    log.info(f"  → Inserted {len(documents)} neighbourhood documents")

    # Create 2dsphere index on geometry field
    # This enables $geoWithin, $geoIntersects, $nearSphere queries
    collection.create_index([("geometry", GEOSPHERE)])
    collection.create_index([("centroid", GEOSPHERE)])
    collection.create_index("neighbourhood")
    collection.create_index("borough")
    log.info("  → 2dsphere indexes created on geometry and centroid fields")

    return {
        "collection":       "neighbourhoods",
        "documents_loaded": len(documents),
        "indexes":          ["geometry (2dsphere)", "centroid (2dsphere)", "neighbourhood", "borough"],
        "source":           str(geojson_path),
    }


# ── Places / POI data ──────────────────────────────────────────────────────────

def load_places(db) -> dict:
    """
    Load Google Places POI documents into the 'places' collection.

    Each Place is stored as an individual document with:
    - A GeoJSON Point geometry (required for 2dsphere spatial queries)
    - Neighbourhood name and borough (for non-spatial lookups)
    - Original Google Places fields (rating, types, vicinity, etc.)

    Deduplication: place_id is used as the unique key. If a place appears
    in multiple neighbourhood searches (overlap zones), it is stored once —
    upserted on place_id to avoid duplicates.
    """
    log.info(f"Loading Places POI data from {RAW_PLACES_DIR}")

    collection = db["places"]
    collection.drop()

    # Create unique index on place_id before inserting — ensures deduplication
    collection.create_index("place_id", unique=True)

    total_inserted = 0
    total_skipped  = 0

    for json_file in sorted(RAW_PLACES_DIR.glob("*.json")):
        if json_file.name == "lineage.json":
            continue

        with open(json_file) as f:
            data = json.load(f)

        neighbourhood = data.get("neighbourhood", "")
        borough_map   = _load_borough_map(RAW_AIRBNB_DIR)
        borough       = borough_map.get(neighbourhood, "Unknown")

        # Deduplicated POI list for this neighbourhood
        for place in data.get("results_deduped", []):
            place_id = place.get("place_id")
            if not place_id:
                continue

            # Extract lat/lon from Google Places geometry
            loc = place.get("geometry", {}).get("location", {})
            lat = loc.get("lat")
            lng = loc.get("lng")

            # Build the document — geometry stored as GeoJSON Point (RFC 7946)
            # MongoDB 2dsphere requires: {"type": "Point", "coordinates": [lon, lat]}
            # Note: GeoJSON uses [longitude, latitude] order (not lat/lon)
            doc = {
                "place_id":     place_id,
                "name":         place.get("name"),
                "neighbourhood": neighbourhood,
                "borough":      borough,
                "types":        place.get("types", []),
                "rating":       place.get("rating"),
                "user_ratings_total": place.get("user_ratings_total"),
                "vicinity":     place.get("vicinity"),
                "price_level":  place.get("price_level"),
                "business_status": place.get("business_status"),

                # GeoJSON Point geometry — required for 2dsphere index
                "geometry": {
                    "type":        "Point",
                    "coordinates": [lng, lat]   # [longitude, latitude] per RFC 7946
                } if lat and lng else None,

                "loaded_at": datetime.now(timezone.utc).isoformat(),
            }

            try:
                # Upsert on place_id — idempotent; handles neighbourhood overlap
                collection.update_one(
                    {"place_id": place_id},
                    {"$set": doc},
                    upsert=True,
                )
                total_inserted += 1
            except pymongo.errors.DuplicateKeyError:
                total_skipped += 1

    log.info(f"  → {total_inserted} places upserted, {total_skipped} duplicates skipped")

    # 2dsphere index on geometry — enables $nearSphere, $geoWithin queries
    collection.create_index([("geometry", GEOSPHERE)])
    collection.create_index("neighbourhood")
    collection.create_index("borough")
    collection.create_index("types")
    log.info("  → 2dsphere index created on geometry field")

    return {
        "collection":      "places",
        "documents_upserted": total_inserted,
        "duplicates_skipped": total_skipped,
        "indexes":         ["geometry (2dsphere)", "neighbourhood", "borough", "types"],
        "source":          str(RAW_PLACES_DIR),
    }


# ── Utilities ──────────────────────────────────────────────────────────────────

def _repair_geometry(geometry: dict) -> tuple[dict, tuple[float, float]]:
    """
    Repair a GeoJSON geometry and return (repaired_geojson, (lon, lat) centroid).

    The Inside Airbnb source GeoJSON contains several classes of invalid
    geometry that cause MongoDB 2dsphere index failures (error 16755):
      - Duplicate consecutive/non-consecutive vertices
      - Self-intersecting rings / crossing edges

    Fix: apply Shapely make_valid() (GEOS ValidOp algorithm). If make_valid
    returns a GeometryCollection (severely degenerate input), extract the
    polygonal components and take their union so we always store a
    Polygon or MultiPolygon — the only types with 'coordinates' that the
    2dsphere index accepts.

    Returns the repaired GeoJSON dict and the centroid (lon, lat) computed
    from the Shapely object (avoids a second pass through the coordinates).
    """
    try:
        shp = shape(geometry)
        if not shp.is_valid:
            shp = make_valid(shp)

        # make_valid may produce a GeometryCollection for severely broken input.
        # Reduce to polygonal parts only.
        if shp.geom_type == "GeometryCollection":
            from shapely.ops import unary_union
            from shapely.geometry import MultiPolygon as MPolygon
            polys = [g for g in shp.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
            shp = unary_union(polys) if polys else shp

        c = shp.centroid
        return mapping(shp), (c.x, c.y)

    except Exception as e:
        log.warning(f"  Geometry repair failed ({e}) — using original + manual centroid")
        lon, lat = _compute_centroid(geometry)
        return geometry, (lon, lat)


def _compute_centroid(geometry: dict) -> tuple[float, float]:
    """
    Compute the arithmetic centroid of a GeoJSON Polygon or MultiPolygon.
    Returns (longitude, latitude) — GeoJSON coordinate order.
    """
    geo_type = geometry["type"]
    coords   = geometry["coordinates"]

    def ring_centroid(ring):
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        return sum(lons) / len(lons), sum(lats) / len(lats)

    if geo_type == "Polygon":
        return ring_centroid(coords[0])

    if geo_type == "MultiPolygon":
        centroids = [ring_centroid(poly[0]) for poly in coords]
        avg_lon = sum(c[0] for c in centroids) / len(centroids)
        avg_lat = sum(c[1] for c in centroids) / len(centroids)
        return avg_lon, avg_lat

    raise ValueError(f"Unsupported geometry type: {geo_type}")


def _load_borough_map(raw_airbnb_dir: Path) -> dict[str, str]:
    """
    Build a neighbourhood → borough mapping from the latest GeoJSON snapshot.
    Used to enrich places documents with borough field.
    """
    snapshots = sorted(raw_airbnb_dir.glob("up_to_*"))
    if not snapshots:
        return {}
    geojson_path = snapshots[-1] / "neighbourhoods.geojson"
    if not geojson_path.exists():
        return {}
    with open(geojson_path) as f:
        fc = json.load(f)
    return {
        feat["properties"]["neighbourhood"]: feat["properties"]["neighbourhood_group"]
        for feat in fc["features"]
    }


def write_lineage(lineage: dict):
    LINEAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LINEAGE_PATH, "w") as f:
        json.dump(lineage, f, indent=2)
    log.info(f"Lineage written to {LINEAGE_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    load_dotenv(REPO_ROOT / ".env")

    try:
        db = get_db()
    except Exception as e:
        log.error(f"MongoDB connection failed: {e}")
        log.error("Ensure MongoDB is running (docker-compose up mongodb)")
        return

    snapshot  = latest_snapshot()
    run_start = datetime.now(timezone.utc).isoformat()

    lineage = {
        "source":         "Google Places API + Inside Airbnb GeoJSON",
        "run_started_at": run_start,
        "collections":    {},
    }

    lineage["collections"]["neighbourhoods"] = load_neighbourhoods(db, snapshot)
    lineage["collections"]["places"]         = load_places(db)

    lineage["run_finished_at"] = datetime.now(timezone.utc).isoformat()
    write_lineage(lineage)
    log.info("MongoDB load complete.")


if __name__ == "__main__":
    run()
