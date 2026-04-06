"""
fetch_places.py
---------------
Fetches Google Places Nearby Search POI data for each NYC neighbourhood
centroid and saves raw JSON responses to:
    data_pipeline/raw/places/<neighbourhood_name>.json

Also writes a lineage.json to raw/places/ on each run.

Usage:
    python fetch_places.py
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

# parents[0] = ingestion/, parents[1] = data_pipeline/
DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
RAW_AIRBNB_DIR    = DATA_PIPELINE_DIR / "raw" / "airbnb"
RAW_PLACES_DIR    = DATA_PIPELINE_DIR / "raw" / "places"

# ── Config ─────────────────────────────────────────────────────────────────────

PLACES_API_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
SEARCH_RADIUS  = 1000  # metres
PLACE_TYPES    = [
    "restaurant",
    "bar",
    "lodging",
    "tourist_attraction",
    "transit_station",
]
SLEEP_BETWEEN_REQUESTS = 0.2  # seconds — stays well within 50 QPS default quota
MAX_NEIGHBOURHOODS     = int(os.getenv("MAX_NEIGHBOURHOODS", "0"))  # 0 = no limit


# ── Snapshot detection ─────────────────────────────────────────────────────────

def latest_airbnb_snapshot() -> Path:
    """Return the Path of the most recent up_to_<date> snapshot directory."""
    snapshots = sorted(RAW_AIRBNB_DIR.glob("up_to_*"))
    if not snapshots:
        raise FileNotFoundError(f"No Airbnb snapshot directories found in {RAW_AIRBNB_DIR}")
    latest = snapshots[-1]
    log.info(f"Using snapshot: {latest.name}")
    return latest


# ── GeoJSON helpers ────────────────────────────────────────────────────────────

def _centroid_of_ring(ring: list) -> tuple[float, float]:
    """Arithmetic centroid of a polygon exterior ring (list of [lon, lat])."""
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return sum(lons) / len(lons), sum(lats) / len(lats)


def centroid(geometry: dict) -> tuple[float, float]:
    """
    Compute centroid for Polygon or MultiPolygon geometry.
    For MultiPolygon, average the centroids of all outer rings weighted equally.
    """
    geo_type = geometry["type"]
    coords   = geometry["coordinates"]

    if geo_type == "Polygon":
        return _centroid_of_ring(coords[0])

    if geo_type == "MultiPolygon":
        ring_centroids = [_centroid_of_ring(poly[0]) for poly in coords]
        avg_lon = sum(c[0] for c in ring_centroids) / len(ring_centroids)
        avg_lat = sum(c[1] for c in ring_centroids) / len(ring_centroids)
        return avg_lon, avg_lat

    raise ValueError(f"Unsupported geometry type: {geo_type}")


def load_neighbourhoods(geojson_path: Path) -> list[dict]:
    """
    Parse neighbourhoods.geojson and return a list of dicts:
        {"name": str, "lat": float, "lon": float}
    """
    with open(geojson_path) as f:
        fc = json.load(f)

    neighbourhoods = []
    for feature in fc["features"]:
        name = feature["properties"]["neighbourhood"]
        lon, lat = centroid(feature["geometry"])
        neighbourhoods.append({"name": name, "lat": lat, "lon": lon})

    log.info(f"Loaded {len(neighbourhoods)} neighbourhoods from {geojson_path.name}")
    return neighbourhoods


# ── Google Places API ──────────────────────────────────────────────────────────

def fetch_nearby(lat: float, lon: float, place_type: str, api_key: str) -> list[dict]:
    """
    Call Nearby Search for one (location, type) pair, following next_page_token
    pagination up to 3 pages (≤60 results).  Returns list of place dicts.
    """
    results = []
    params = {
        "location": f"{lat},{lon}",
        "radius":   SEARCH_RADIUS,
        "type":     place_type,
        "key":      api_key,
    }

    for page in range(3):
        try:
            resp = requests.get(PLACES_API_URL, params=params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.error(f"Request failed (page {page + 1}): {e}")
            break

        data = resp.json()
        status = data.get("status")

        if status == "ZERO_RESULTS":
            break
        if status not in ("OK", "UNKNOWN_ERROR"):
            log.warning(f"API status {status} for type={place_type} lat={lat} lon={lon}")
            break

        results.extend(data.get("results", []))

        next_token = data.get("next_page_token")
        if not next_token:
            break

        # Google requires a short delay before the next_page_token becomes valid
        time.sleep(2)
        params = {"pagetoken": next_token, "key": api_key}

    return results


def fetch_all_types(lat: float, lon: float, api_key: str) -> dict:
    """
    Query all PLACE_TYPES for a single centroid.
    Returns {"by_type": {type: [results]}, "all_results": [deduplicated results]}.
    """
    by_type: dict[str, list] = {}
    seen_ids: set[str] = set()
    all_results: list[dict] = []

    for place_type in PLACE_TYPES:
        raw = fetch_nearby(lat, lon, place_type, api_key)
        by_type[place_type] = raw
        time.sleep(SLEEP_BETWEEN_REQUESTS)

        for place in raw:
            pid = place.get("place_id")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_results.append(place)

    return {"by_type": by_type, "all_results": all_results}


# ── Safe filename ──────────────────────────────────────────────────────────────

def safe_filename(name: str) -> str:
    """Convert neighbourhood name to a safe filename (spaces → underscores)."""
    return name.replace(" ", "_").replace("/", "-")


# ── Lineage ────────────────────────────────────────────────────────────────────

def write_lineage(lineage: dict):
    RAW_PLACES_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_PLACES_DIR / "lineage.json"
    with open(path, "w") as f:
        json.dump(lineage, f, indent=2)
    log.info(f"Lineage written to {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    # Load .env from repo root (two levels above data_pipeline/: data_pipeline/ → project/ → repo root)
    env_path = DATA_PIPELINE_DIR.parents[1] / ".env"
    load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_PLACES_API_KEY not set — check your .env file")

    RAW_PLACES_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_dir  = latest_airbnb_snapshot()
    geojson_path  = snapshot_dir / "neighbourhoods.geojson"
    neighbourhoods = load_neighbourhoods(geojson_path)

    run_start = datetime.now(timezone.utc).isoformat()
    lineage = {
        "source":           "Google Places Nearby Search API",
        "geojson_snapshot": snapshot_dir.name,
        "search_radius_m":  SEARCH_RADIUS,
        "place_types":      PLACE_TYPES,
        "run_started_at":   run_start,
        "neighbourhoods":   {},
    }

    global_seen_ids: set[str] = set()
    total_new = total_skipped = total_failed = 0

    for nb in neighbourhoods:
        name     = nb["name"]
        lat, lon = nb["lat"], nb["lon"]
        out_path = RAW_PLACES_DIR / f"{safe_filename(name)}.json"

        if out_path.exists():
            log.info(f"Skipping (already downloaded): {name}")
            lineage["neighbourhoods"][name] = {"status": "skipped", "file": str(out_path)}
            total_skipped += 1
            continue

        log.info(f"Fetching POIs for: {name}  ({lat:.5f}, {lon:.5f})")
        try:
            payload = fetch_all_types(lat, lon, api_key)
        except Exception as e:
            log.error(f"Failed for {name}: {e}")
            lineage["neighbourhoods"][name] = {"status": "error", "error": str(e)}
            total_failed += 1
            continue

        # Cross-neighbourhood deduplication: tag each result and track global IDs
        new_global = 0
        for place in payload["all_results"]:
            pid = place.get("place_id")
            if pid and pid not in global_seen_ids:
                global_seen_ids.add(pid)
                new_global += 1

        # Persist raw response
        envelope = {
            "neighbourhood":    name,
            "centroid":         {"lat": lat, "lon": lon},
            "fetched_at":       datetime.now(timezone.utc).isoformat(),
            "place_types":      PLACE_TYPES,
            "search_radius_m":  SEARCH_RADIUS,
            "results_by_type":  payload["by_type"],
            "results_deduped":  payload["all_results"],
        }
        with open(out_path, "w") as f:
            json.dump(envelope, f, indent=2)

        total_count = len(payload["all_results"])
        log.info(f"  → {total_count} unique POIs ({new_global} new globally) saved to {out_path.name}")

        lineage["neighbourhoods"][name] = {
            "status":            "ok",
            "file":              str(out_path),
            "fetched_at":        envelope["fetched_at"],
            "total_results":     total_count,
            "new_global_unique": new_global,
        }
        total_new += 1

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    lineage["run_finished_at"] = datetime.now(timezone.utc).isoformat()
    lineage["summary"] = {
        "neighbourhoods_total":     len(neighbourhoods),
        "neighbourhoods_fetched":   total_new,
        "neighbourhoods_skipped":   total_skipped,
        "neighbourhoods_failed":    total_failed,
        "global_unique_place_ids":  len(global_seen_ids),
    }

    write_lineage(lineage)
    log.info(
        f"Done. {total_new} fetched, {total_skipped} skipped, "
        f"{total_failed} failed. {len(global_seen_ids)} unique place IDs total."
    )


if __name__ == "__main__":
    run()
