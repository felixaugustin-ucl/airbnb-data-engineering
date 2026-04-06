"""
fetch_airbnb.py
---------------
Dynamically discovers and downloads all available NYC Inside Airbnb snapshots
(listings, calendar, reviews, neighbourhoods) into:
    data_pipeline/raw/airbnb/<snapshot_date>/<file>

Snapshot dates are discovered by scraping the Inside Airbnb get-the-data page,
so new snapshots are automatically picked up on re-runs without any code changes.

A lineage log (lineage.json) is written to raw/airbnb/ on each run, recording
which snapshots were discovered, which files were downloaded, and when.

Usage:
    python fetch_airbnb.py
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_URL       = "https://data.insideairbnb.com/united-states/ny/new-york-city"
INDEX_URL      = "https://insideairbnb.com/get-the-data/"
NYC_URL_FRAG   = "united-states/ny/new-york-city"

DATA_FILES = [
    "data/listings.csv.gz",
    "data/calendar.csv.gz",
    "data/reviews.csv.gz",
]

VIS_FILES = [
    "visualisations/listings.csv",
    "visualisations/reviews.csv",
    "visualisations/neighbourhoods.csv",
    "visualisations/neighbourhoods.geojson",
]

ALL_FILES = DATA_FILES + VIS_FILES

# Output root: <project>/data_pipeline/raw/airbnb
# parents[0] = ingestion/, parents[1] = data_pipeline/
RAW_DIR = Path(__file__).resolve().parents[1] / "raw" / "airbnb"

# ── Snapshot discovery ─────────────────────────────────────────────────────────

def discover_nyc_snapshots() -> list[str]:
    """
    Scrape the Inside Airbnb get-the-data page and return all unique NYC
    snapshot dates found in download URLs, sorted ascending.
    """
    log.info(f"Discovering NYC snapshots from {INDEX_URL}")
    try:
        response = requests.get(INDEX_URL, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        log.error(f"Failed to fetch index page: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    dates = set()

    for link in soup.find_all("a", href=True):
        href = link["href"]
        match = re.search(rf"{NYC_URL_FRAG}/(\d{{4}}-\d{{2}}-\d{{2}})/", href)
        if match:
            dates.add(match.group(1))

    sorted_dates = sorted(dates)
    log.info(f"Found {len(sorted_dates)} NYC snapshots: {sorted_dates}")
    return sorted_dates

# ── Download ───────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path) -> dict:
    """
    Download a single file to dest.
    Returns a lineage record dict describing the outcome.
    """
    record = {
        "url": url,
        "dest": str(dest),
        "status": None,
        "downloaded_at": None,
    }

    if dest.exists():
        log.info(f"Already exists, skipping: {dest.name}")
        record["status"] = "skipped"
        return record

    try:
        log.info(f"Downloading {url}")
        response = requests.get(url, timeout=120, stream=True)

        if response.status_code == 404:
            log.warning(f"Not found (404): {url}")
            record["status"] = "not_found"
            return record

        response.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        record["status"] = "downloaded"
        record["downloaded_at"] = datetime.now(timezone.utc).isoformat()
        log.info(f"Saved: {dest}")

    except requests.RequestException as e:
        log.error(f"Failed: {url} — {e}")
        record["status"] = "error"
        record["error"] = str(e)

    return record

# ── Lineage ────────────────────────────────────────────────────────────────────

def write_lineage(lineage: dict):
    """Write lineage log to raw/airbnb/lineage.json."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    lineage_path = RAW_DIR / "lineage.json"

    # Merge with existing lineage if present
    existing = {}
    if lineage_path.exists():
        with open(lineage_path) as f:
            existing = json.load(f)

    existing.update(lineage)

    with open(lineage_path, "w") as f:
        json.dump(existing, f, indent=2)

    log.info(f"Lineage written to {lineage_path}")

# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    """
    Main entry point:
    1. Discover all available NYC snapshots dynamically
    2. Download all files for each snapshot
    3. Write a lineage log
    """
    log.info(f"Starting NYC Airbnb ingestion → {RAW_DIR}")
    run_start = datetime.now(timezone.utc).isoformat()

    snapshots = discover_nyc_snapshots()
    if not snapshots:
        log.error("No snapshots discovered — aborting.")
        return

    lineage = {
        "source": "Inside Airbnb",
        "city": "New York City",
        "index_url": INDEX_URL,
        "run_started_at": run_start,
        "snapshots": {}
    }

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for date in snapshots:
        log.info(f"─── Snapshot: {date} ───")
        snapshot_dir = RAW_DIR / f"up_to_{date}"
        snapshot_records = []

        for file_path in ALL_FILES:
            url  = f"{BASE_URL}/{date}/{file_path}"
            dest = snapshot_dir / Path(file_path).name
            record = download_file(url, dest)
            snapshot_records.append(record)

            if record["status"] == "downloaded":
                total_downloaded += 1
            elif record["status"] == "skipped":
                total_skipped += 1
            else:
                total_failed += 1

        lineage["snapshots"][date] = snapshot_records

    lineage["run_finished_at"] = datetime.now(timezone.utc).isoformat()
    lineage["summary"] = {
        "snapshots_found": len(snapshots),
        "files_downloaded": total_downloaded,
        "files_skipped": total_skipped,
        "files_failed": total_failed,
    }

    write_lineage(lineage)

    log.info(
        f"Done. {total_downloaded} downloaded, "
        f"{total_skipped} skipped, {total_failed} failed."
    )


if __name__ == "__main__":
    run()
