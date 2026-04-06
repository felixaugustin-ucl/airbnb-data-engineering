"""
fetch_subway_stations.py
------------------------
Fetches MTA subway station data from the NYC Open Data Socrata API and saves
the raw response to:
    data_pipeline/raw/nyc_open_data/subway_stations.json

This is a small, static reference dataset (~500 stations) fetched in a
single request — no pagination or date range required.

Also writes/updates a lineage.json in raw/nyc_open_data/ on each run.

Usage:
    python fetch_subway_stations.py
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_PIPELINE_DIR  = Path(__file__).resolve().parents[1]
RAW_OPEN_DATA_DIR  = DATA_PIPELINE_DIR / "raw" / "nyc_open_data"

# ── Config ─────────────────────────────────────────────────────────────────────

SOCRATA_URL = "https://data.cityofnewyork.us/resource/kk4q-3rt2.json"

# Fields relevant to spatial and analytical use
SELECT_FIELDS = (
    "name,line,notes,the_geom"
)


# ── Fetch ──────────────────────────────────────────────────────────────────────

def fetch_stations(app_token: str | None) -> list[dict]:
    """
    Fetch all subway station records in a single request.
    The dataset is small enough (~500 rows) that pagination is not needed.
    """
    headers = {}
    if app_token:
        headers["X-App-Token"] = app_token

    params = {
        "$limit":  1000,   # well above the actual station count
        "$select": SELECT_FIELDS,
        "$order":  "name ASC",
    }

    log.info(f"Fetching subway stations from {SOCRATA_URL}")
    try:
        resp = requests.get(SOCRATA_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Socrata request failed: {e}") from e

    records = resp.json()
    log.info(f"Received {len(records)} subway station records")
    return records


# ── Lineage ────────────────────────────────────────────────────────────────────

def write_lineage(entry: dict):
    RAW_OPEN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_OPEN_DATA_DIR / "lineage.json"

    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)

    existing["subway_stations"] = entry

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    log.info(f"Lineage written to {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    env_path = DATA_PIPELINE_DIR.parents[0] / ".env"
    load_dotenv(dotenv_path=env_path)

    app_token = os.getenv("SOCRATA_APP_TOKEN")  # optional

    RAW_OPEN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_OPEN_DATA_DIR / "subway_stations.json"

    # Subway stations are static — skip if file already exists
    if out_path.exists():
        log.info("subway_stations.json already exists — skipping fetch")
        log.info("Delete the file manually to force a refresh")
        return

    records = fetch_stations(app_token)

    envelope = {
        "meta": {
            "source":       "NYC Open Data — MTA Subway Stations",
            "dataset_id":   "kk4q-3rt2",
            "url":          SOCRATA_URL,
            "record_count": len(records),
            "fetched_at":   datetime.now(timezone.utc).isoformat(),
            "note":         (
                "Static reference dataset. the_geom field contains GeoJSON Point "
                "geometry suitable for MongoDB 2dsphere indexing."
            ),
        },
        "records": records,
    }

    with open(out_path, "w") as f:
        json.dump(envelope, f, indent=2)

    log.info(f"Saved {len(records)} subway stations to {out_path}")

    write_lineage({
        "source":       "NYC Open Data — MTA Subway Stations",
        "dataset_id":   "kk4q-3rt2",
        "record_count": len(records),
        "output":       str(out_path),
        "fetched_at":   envelope["meta"]["fetched_at"],
    })


if __name__ == "__main__":
    run()
