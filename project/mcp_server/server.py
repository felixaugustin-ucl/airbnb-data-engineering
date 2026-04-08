"""
server.py
---------
MCP server exposing four tools over the NYC Airbnb Hospitality Lakehouse.

Tools:
  query_warehouse          — read-only SQL against PostgreSQL (star schema)
  query_geodata            — MongoDB geospatial / document queries
  search_reviews           — semantic search over 56k guest reviews (ChromaDB)
  get_neighbourhood_context — semantic search over 598 Wikipedia paragraphs (ChromaDB)

Architecture:
  FastMCP registers Python functions as MCP tools, exposing them over stdio
  so any MCP-compatible client (Claude desktop, agent.py) can call them.

  All four stores are read-only from the server's perspective — no writes are
  exposed. PostgreSQL queries are restricted to SELECT statements.

Environment variables (from .env):
  POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
  MONGO_URI, MONGO_DB
  CHROMA_PATH   — path to ChromaDB persistence dir (default: processed/chromadb)

PostgreSQL schema reference:
  dim_date             (date_id, full_date, year, month, quarter, day_of_week, season, …)
  dim_neighbourhood    (neighbourhood_id, neighbourhood, borough, poi_restaurant,
                        poi_bar, poi_lodging, poi_tourist_attraction, poi_transit_station,
                        poi_total)
  dim_host             (host_id, host_name, host_since, host_is_superhost,
                        host_response_rate, host_acceptance_rate, host_total_listings_count)
  dim_listing          (listing_id, host_id, neighbourhood_id, room_type, property_type,
                        accommodates, bedrooms, beds, bathrooms, minimum_nights,
                        estimated_occupancy_l365d, instant_bookable,
                        review_scores_rating, review_scores_cleanliness, … , latitude, longitude)
  fact_review          (review_id, listing_id, host_id, neighbourhood_id, date_id,
                        review_date, complaint_count)
  fact_review_weather  (review_id, weather_date, days_before_review, temperature_max,
                        temperature_min, precipitation_sum, windspeed_max, weathercode)
  noise_borough_month  (borough, year, month, complaint_count, complaint_types)

MongoDB collections:
  neighbourhoods  — GeoJSON MultiPolygon boundaries, centroid, borough
  places          — GeoJSON Point POIs (restaurants, bars, hotels, …)
                    fields: place_id, name, neighbourhood, borough, types,
                            rating, user_ratings_total, vicinity, price_level

ChromaDB collections:
  review_db               — metadata: listing_id, neighbourhood, review_date
  neighbourhood_context_db — metadata: neighbourhood, borough, wikipedia_url
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# CRITICAL only — all lower-level logs suppressed to keep agent output clean
logging.basicConfig(level=logging.CRITICAL, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Path resolution ────────────────────────────────────────────────────────────

MCP_DIR          = Path(__file__).resolve().parent
PROJECT_DIR      = MCP_DIR.parent
REPO_ROOT        = PROJECT_DIR.parent
DATA_PIPELINE_DIR = PROJECT_DIR / "data_pipeline"

load_dotenv(REPO_ROOT / ".env")

# ── FastMCP instance ───────────────────────────────────────────────────────────

mcp = FastMCP("nyc-airbnb-lakehouse")


# ══════════════════════════════════════════════════════════════════════════════
# CONNECTION HELPERS
# Connections are established on first call and reused — lazy initialisation
# avoids startup failures when a store is temporarily unavailable.
# ══════════════════════════════════════════════════════════════════════════════

_pg_engine    = None
_mongo_client = None
_chroma_client = None


def _get_pg_engine():
    global _pg_engine
    if _pg_engine is None:
        from sqlalchemy import create_engine
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        db   = os.getenv("POSTGRES_DB",   "airbnb")
        user = os.getenv("POSTGRES_USER", "postgres")
        pw   = os.getenv("POSTGRES_PASSWORD", "postgres")
        url  = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"
        _pg_engine = create_engine(url, pool_pre_ping=True)
        log.info(f"PostgreSQL engine created: {host}:{port}/{db}")
    return _pg_engine


def _get_mongo_db():
    global _mongo_client
    if _mongo_client is None:
        from pymongo import MongoClient
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        _mongo_client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        log.info(f"MongoDB client created: {uri}")
    db_name = os.getenv("MONGO_DB", "airbnb_geo")
    return _mongo_client[db_name]


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        path = os.getenv(
            "CHROMA_PATH",
            str(DATA_PIPELINE_DIR / "processed" / "chromadb"),
        )
        _chroma_client = chromadb.PersistentClient(path=path)
        log.info(f"ChromaDB client created: {path}")
    return _chroma_client


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — query_warehouse
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def query_warehouse(sql: str, limit: Any = 100) -> str:
    """
    Execute a read-only SQL SELECT query against the PostgreSQL star schema
    and return results as a JSON array of row objects.

    The query must be a SELECT statement — DML and DDL are rejected for safety.
    If the query contains no LIMIT clause, one is automatically appended using
    the `limit` parameter (default 100, max 1000).

    Available tables:
      dim_date, dim_neighbourhood, dim_listing, dim_host,
      fact_review, fact_review_weather, noise_borough_month

    Example queries:
      "SELECT neighbourhood, borough, poi_total FROM dim_neighbourhood ORDER BY poi_total DESC LIMIT 10"
      "SELECT season, COUNT(*) as reviews FROM fact_review JOIN dim_date USING(date_id) GROUP BY season"
      "SELECT AVG(temperature_max), COUNT(*) FROM fact_review_weather WHERE weathercode IN (61,63,65)"

    Args:
        sql:   A SELECT statement to execute.
        limit: Row cap applied when no LIMIT clause is present (default 100, max 1000).

    Returns:
        JSON string — array of row objects, or {"error": "..."} on failure.
    """
    sql_clean = sql.strip().rstrip(";")

    # Safety: only SELECT statements
    if not sql_clean.upper().startswith("SELECT"):
        return json.dumps({"error": "Only SELECT statements are permitted."})

    # Apply limit — coerce defensively: llama3.2 occasionally passes
    # {"type": "integer", "value": "1"} instead of a plain integer.
    if isinstance(limit, dict):
        limit = limit.get("value", 100)
    effective_limit = min(int(limit), 1000)
    if "LIMIT" not in sql_clean.upper():
        sql_clean = f"{sql_clean} LIMIT {effective_limit}"

    try:
        import pandas as pd
        engine = _get_pg_engine()
        with engine.connect() as conn:
            df = pd.read_sql(sql_clean, conn)
        # Convert non-JSON-serialisable types (Timestamps, Decimals, etc.)
        records = json.loads(df.to_json(orient="records", date_format="iso"))
        return json.dumps({"rows": records, "row_count": len(records)}, indent=2)
    except Exception as e:
        log.error(f"query_warehouse error: {e}")
        return json.dumps({"error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — query_geodata
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def query_geodata(
    collection: str,
    filter: dict | None = None,
    projection: dict | None = None,
    limit: int = 20,
    sort: list[list] | None = None,
    pipeline: list | None = None,
) -> str:
    """
    Query the MongoDB geospatial store and return matching documents as JSON.

    Two modes:
      1. Find mode (default)  — filter + projection + sort + limit (simple queries)
      2. Aggregate mode       — pass a `pipeline` list of MongoDB aggregation stages

    Use aggregate mode when you need GROUP BY, AVG, COUNT, or other aggregations
    across documents (e.g. average restaurant rating by borough).

    Available collections:
      neighbourhoods — GeoJSON MultiPolygon boundaries per neighbourhood
        Fields: neighbourhood, borough, geometry (MultiPolygon), centroid (Point)

      places — Google Places POI documents
        Fields: place_id, name, neighbourhood, borough, types, rating,
                user_ratings_total, vicinity, price_level, geometry (Point)
        types values: "restaurant", "bar", "lodging", "tourist_attraction",
                      "transit_station"

    Find mode example (restaurants in Williamsburg):
      collection: "places"
      filter: {"neighbourhood": "Williamsburg", "types": "restaurant"}
      sort: [["rating", -1]]
      limit: 10

    Aggregate mode example (average restaurant rating per borough):
      collection: "places"
      pipeline: [
        {"$match": {"types": "restaurant", "rating": {"$exists": true}}},
        {"$group": {"_id": "$borough", "avg_rating": {"$avg": "$rating"},
                    "place_count": {"$sum": 1}}},
        {"$sort": {"avg_rating": -1}}
      ]

    Args:
        collection:  "neighbourhoods" or "places"
        filter:      MongoDB filter dict — find mode only (default: {})
        projection:  Fields to include/exclude — find mode only
        limit:       Max documents — find mode only (default 20, max 200)
        sort:        [field, direction] pairs — find mode only
        pipeline:    Aggregation pipeline stages — aggregate mode
                     (overrides filter/projection/sort/limit when provided)

    Returns:
        JSON string — {"documents": [...], "count": N} or {"error": "..."}.
    """
    if collection not in ("neighbourhoods", "places"):
        return json.dumps({"error": f"Unknown collection '{collection}'. Use 'neighbourhoods' or 'places'."})

    try:
        db  = _get_mongo_db()
        col = db[collection]

        # ── Aggregate mode ─────────────────────────────────────────────────────
        if pipeline:
            # Append $project to strip _id from aggregation results
            has_project = any("$project" in stage for stage in pipeline)
            if not has_project:
                pipeline = pipeline + [{"$project": {"_id": 0}}]

            docs = list(col.aggregate(pipeline))
            return json.dumps(
                {"documents": docs, "count": len(docs)},
                indent=2,
                default=str,
            )

        # ── Find mode ──────────────────────────────────────────────────────────
        if isinstance(limit, dict):
            limit = limit.get("value", 20)
        effective_limit = min(int(limit), 200)
        query_filter    = filter or {}
        proj            = projection or {}
        proj["_id"]     = 0

        cursor = col.find(query_filter, proj).limit(effective_limit)
        if sort:
            cursor = cursor.sort(sort)

        docs = list(cursor)
        return json.dumps(
            {"documents": docs, "count": len(docs)},
            indent=2,
            default=str,
        )

    except Exception as e:
        log.error(f"query_geodata error: {e}")
        return json.dumps({"error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — search_reviews
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_reviews(
    query: str,
    n_results: int = 10,
    where: dict | None = None,
) -> str:
    """
    Semantic search over 56,132 stratified guest review texts using dot-product
    similarity (multi-qa-mpnet-base-dot-v1 encoder).

    Use this tool to answer experiential questions about listings or neighbourhoods,
    e.g. "quiet street", "close to subway", "great host", "noisy at night".

    Each result includes:
      - document: the review text
      - distance: similarity score (higher = more similar, dot-product space)
      - metadata: listing_id, neighbourhood, review_date

    The metadata `where` filter uses ChromaDB's filter syntax and supports
    exact match and comparison operators:
      {"neighbourhood": "Williamsburg"}
      {"$and": [{"neighbourhood": "Williamsburg"}, {"review_date": {"$gte": "2023-01-01"}}]}

    Args:
        query:     Natural language query string.
        n_results: Number of results to return (default 10, max 50).
        where:     Optional ChromaDB metadata filter dict.

    Returns:
        JSON string — {"results": [...], "query": query} or {"error": "..."}.
    """
    try:
        from sentence_transformers import SentenceTransformer
        client = _get_chroma_client()
        collection = client.get_collection("review_db")

        # Encode query with the same model used at index time
        encoder = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
        q_vec   = encoder.encode(query).tolist()

        effective_n = min(int(n_results), 50)
        kwargs: dict[str, Any] = {"query_embeddings": [q_vec], "n_results": effective_n}
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)

        formatted = []
        for i in range(len(results["documents"][0])):
            formatted.append({
                "document": results["documents"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
            })

        return json.dumps({"results": formatted, "query": query}, indent=2)
    except Exception as e:
        log.error(f"search_reviews error: {e}")
        return json.dumps({"error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — get_neighbourhood_context
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_neighbourhood_context(
    query: str,
    n_results: int = 5,
    neighbourhood: str | None = None,
) -> str:
    """
    Semantic search over 598 Wikipedia neighbourhood intro paragraphs.

    Use this tool to retrieve explanatory or contextual information about a
    neighbourhood — its character, history, demographics, or development trends.
    This is intended as a complementary layer to review signals: reviews may
    surface emerging patterns ("artsy vibe", "great restaurants") while Wikipedia
    provides the structural explanation ("Williamsburg has undergone significant
    gentrification since the early 2000s").

    Each result includes:
      - document: the Wikipedia paragraph text
      - distance: similarity score
      - metadata: neighbourhood, borough, wikipedia_url

    Args:
        query:         Natural language query string, e.g. "gentrification", "transit".
        n_results:     Number of paragraphs to return (default 5, max 20).
        neighbourhood: Optional exact neighbourhood name to restrict results.

    Returns:
        JSON string — {"results": [...], "query": query} or {"error": "..."}.
    """
    try:
        from sentence_transformers import SentenceTransformer
        client = _get_chroma_client()
        collection = client.get_collection("neighbourhood_context_db")

        encoder = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
        q_vec   = encoder.encode(query).tolist()

        effective_n = min(int(n_results), 20)
        kwargs: dict[str, Any] = {"query_embeddings": [q_vec], "n_results": effective_n}
        if neighbourhood:
            kwargs["where"] = {"neighbourhood": neighbourhood}

        results = collection.query(**kwargs)

        formatted = []
        for i in range(len(results["documents"][0])):
            formatted.append({
                "document": results["documents"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
            })

        return json.dumps({"results": formatted, "query": query}, indent=2)
    except Exception as e:
        log.error(f"get_neighbourhood_context error: {e}")
        return json.dumps({"error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mcp.run()
