"""
fetch_noise_complaints.py
--------------------------
Fetches NYC 311 noise complaints from the NYC Open Data Socrata API and saves
the raw paginated results to:
    data_pipeline/raw/nyc_open_data/noise_complaints.json

The date range is derived automatically from the Airbnb snapshot directories
present in raw/airbnb/, so this script should be run after fetch_airbnb.py.

A Socrata App Token can optionally be set in .env as SOCRATA_APP_TOKEN to
increase rate limits (from 500 to unlimited requests/hour). The script runs
without one but will be throttled on large date ranges.

Also writes a lineage.json to raw/nyc_open_data/ on each run.

Usage:
    python fetch_noise_complaints.py
    MAX_RECORDS=10000 python fetch_noise_complaints.py  # cap for testing
"""

import csv
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_PIPELINE_DIR   = Path(__file__).resolve().parents[1]
RAW_AIRBNB_DIR      = DATA_PIPELINE_DIR / "raw" / "airbnb"
RAW_OPEN_DATA_DIR   = DATA_PIPELINE_DIR / "raw" / "nyc_open_data"

# ── Config ─────────────────────────────────────────────────────────────────────

SOCRATA_BASE_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
PAGE_SIZE        = 50_000   # max per request with App Token; 1000 without
SLEEP_BETWEEN_PAGES = 0.5   # seconds — polite rate
MAX_RECORDS      = int(os.getenv("MAX_RECORDS", "0"))  # 0 = no limit


# ── Date range detection ───────────────────────────────────────────────────────

STAY_LOOKBACK_DAYS = 10  # window before review date assumed to cover the stay

def review_date_range() -> tuple[str, str]:
    """
    Derive the true date range from review dates across all snapshots,
    then subtract STAY_LOOKBACK_DAYS from the start to capture noise
    complaints that occurred during the stay preceding the earliest review.
    """
    snapshots = sorted(RAW_AIRBNB_DIR.glob("up_to_*"))
    if not snapshots:
        raise FileNotFoundError(f"No Airbnb snapshots found in {RAW_AIRBNB_DIR}")

    all_min, all_max = [], []
    for snapshot in snapshots:
        reviews_path = snapshot / "reviews.csv"
        if not reviews_path.exists():
            continue
        dates = []
        with open(reviews_path, newline="") as f:
            for row in csv.DictReader(f):
                d = row.get("date", "").strip()
                if d:
                    dates.append(d)
        if dates:
            all_min.append(min(dates))
            all_max.append(max(dates))

    if not all_min:
        raise ValueError("No review dates found across any snapshot")

    # Subtract lookback window from earliest review date
    earliest = datetime.strptime(min(all_min), "%Y-%m-%d") - timedelta(days=STAY_LOOKBACK_DAYS)
    start_date = earliest.strftime("%Y-%m-%d")
    end_date   = max(all_max)

    log.info(f"Review date range: {min(all_min)} → {end_date} "
             f"(fetch from {start_date} with {STAY_LOOKBACK_DAYS}-day lookback)")
    return start_date, end_date


# ── Socrata fetch ──────────────────────────────────────────────────────────────

def fetch_all_complaints(start_date: str, end_date: str, app_token: str | None) -> list[dict]:
    """
    Paginate through the Socrata SODA API, filtering to noise complaints
    within the date range. Returns a list of raw complaint dicts.
    """
    headers = {}
    if app_token:
        headers["X-App-Token"] = app_token
        log.info("Using Socrata App Token for higher rate limits")
    else:
        log.warning("No SOCRATA_APP_TOKEN set — limited to 500 requests/hour and 1000 records/page")

    # SoQL filter: only noise complaint types, within date range
    where_clause = (
        f"complaint_type like 'Noise%' "
        f"AND created_date >= '{start_date}T00:00:00' "
        f"AND created_date <= '{end_date}T23:59:59'"
    )

    effective_page_size = PAGE_SIZE if app_token else 1000

    all_records = []
    offset = 0

    while True:
        params = {
            "$where":  where_clause,
            "$order":  "created_date ASC",
            "$limit":  effective_page_size,
            "$offset": offset,
            # Only fetch the fields we need — reduces payload size
            "$select": (
                "unique_key,created_date,complaint_type,descriptor,"
                "incident_zip,borough,community_board,"
                "latitude,longitude,status,resolution_description"
            ),
        }

        try:
            resp = requests.get(SOCRATA_BASE_URL, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.error(f"Request failed at offset {offset}: {e}")
            break

        page = resp.json()
        if not page:
            break

        all_records.extend(page)
        log.info(f"  Fetched {len(all_records)} records so far (page offset {offset})")

        # Respect MAX_RECORDS cap if set
        if MAX_RECORDS and len(all_records) >= MAX_RECORDS:
            all_records = all_records[:MAX_RECORDS]
            log.info(f"MAX_RECORDS={MAX_RECORDS} reached — stopping pagination")
            break

        if len(page) < effective_page_size:
            break  # last page

        offset += effective_page_size
        time.sleep(SLEEP_BETWEEN_PAGES)

    return all_records


# ── Lineage ────────────────────────────────────────────────────────────────────

def write_lineage(entry: dict):
    RAW_OPEN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_OPEN_DATA_DIR / "lineage.json"

    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)

    existing["noise_complaints"] = entry

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    log.info(f"Lineage written to {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    env_path = DATA_PIPELINE_DIR.parents[1] / ".env"
    load_dotenv(dotenv_path=env_path)

    app_token = os.getenv("SOCRATA_APP_TOKEN")  # optional

    RAW_OPEN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_OPEN_DATA_DIR / "noise_complaints.json"

    start_date, end_date = review_date_range()

    # Skip if file already covers the same date range
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        if (
            existing.get("meta", {}).get("start_date") == start_date
            and existing.get("meta", {}).get("end_date") == end_date
        ):
            log.info("Noise complaints already up to date — skipping fetch")
            return
        log.info("Date range changed — re-fetching noise complaints")

    records = fetch_all_complaints(start_date, end_date, app_token)

    envelope = {
        "meta": {
            "source":       "NYC Open Data — 311 Service Requests",
            "dataset_id":   "erm2-nwe9",
            "url":          SOCRATA_BASE_URL,
            "filter":       "complaint_type LIKE 'Noise%'",
            "start_date":   start_date,
            "end_date":     end_date,
            "record_count": len(records),
            "capped_at":    MAX_RECORDS if MAX_RECORDS else None,
            "fetched_at":   datetime.now(timezone.utc).isoformat(),
        },
        "records": records,
    }

    with open(out_path, "w") as f:
        json.dump(envelope, f, indent=2)

    log.info(f"Saved {len(records)} noise complaints to {out_path}")

    write_lineage({
        "source":       "NYC Open Data — 311 Noise Complaints",
        "dataset_id":   "erm2-nwe9",
        "start_date":   start_date,
        "end_date":     end_date,
        "record_count": len(records),
        "capped_at":    MAX_RECORDS if MAX_RECORDS else None,
        "output":       str(out_path),
        "fetched_at":   envelope["meta"]["fetched_at"],
    })


if __name__ == "__main__":
    run()
