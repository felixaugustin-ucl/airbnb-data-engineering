"""
fetch_weather.py
----------------
Fetches historical daily weather data for NYC from Open-Meteo's free archive API
(no API key required) and saves the raw response to:
    data_pipeline/raw/weather/nyc_weather.json

The date range is derived from actual review dates across all Airbnb snapshots,
minus a 10-day lookback to cover the stay period preceding each review.
At transformation time, weather is joined per review as the average of the
10 days before the review date.

Also writes a lineage.json to raw/weather/ on each run.

Usage:
    python fetch_weather.py
"""

import csv
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
RAW_AIRBNB_DIR    = DATA_PIPELINE_DIR / "raw" / "airbnb"
RAW_WEATHER_DIR   = DATA_PIPELINE_DIR / "raw" / "weather"

# ── Config ─────────────────────────────────────────────────────────────────────

NYC_LAT = 40.7128
NYC_LON = -74.0060

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
    "weathercode",
]

STAY_LOOKBACK_DAYS = 10  # window before review date assumed to cover the stay


# ── Date range detection ───────────────────────────────────────────────────────

def review_date_range() -> tuple[str, str]:
    """
    Derive date range from actual review dates across all snapshots,
    then subtract STAY_LOOKBACK_DAYS from the earliest review date so
    weather is available for the full estimated stay window.
    """
    snapshots = sorted(RAW_AIRBNB_DIR.glob("up_to_*"))
    if not snapshots:
        raise FileNotFoundError(f"No Airbnb snapshot directories found in {RAW_AIRBNB_DIR}")

    all_min, all_max = [], []

    for snapshot in snapshots:
        reviews_path = snapshot / "reviews.csv"
        if not reviews_path.exists():
            log.warning(f"No reviews.csv in {snapshot.name}, skipping")
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
            log.info(f"{snapshot.name}: reviews span {min(dates)} → {max(dates)}")

    if not all_min:
        raise ValueError("No review dates found across any snapshot")

    earliest = datetime.strptime(min(all_min), "%Y-%m-%d") - timedelta(days=STAY_LOOKBACK_DAYS)
    start_date = earliest.strftime("%Y-%m-%d")
    end_date   = max(all_max)

    log.info(f"Fetching weather: {start_date} → {end_date} "
             f"(review range {min(all_min)} → {end_date}, {STAY_LOOKBACK_DAYS}-day lookback)")
    return start_date, end_date


# ── API fetch ──────────────────────────────────────────────────────────────────

def fetch_weather(start_date: str, end_date: str) -> dict:
    params = {
        "latitude":   NYC_LAT,
        "longitude":  NYC_LON,
        "start_date": start_date,
        "end_date":   end_date,
        "daily":      ",".join(DAILY_VARIABLES),
        "timezone":   "America/New_York",
    }
    log.info(f"Fetching weather from Open-Meteo: {start_date} → {end_date}")
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Open-Meteo request failed: {e}") from e

    data = resp.json()
    n_days = len(data.get("daily", {}).get("time", []))
    log.info(f"Received {n_days} daily records")
    return data


# ── Lineage ────────────────────────────────────────────────────────────────────

def write_lineage(lineage: dict):
    RAW_WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_WEATHER_DIR / "lineage.json"
    with open(path, "w") as f:
        json.dump(lineage, f, indent=2)
    log.info(f"Lineage written to {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    RAW_WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_WEATHER_DIR / "nyc_weather.json"

    start_date, end_date = review_date_range()

    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        if (
            existing.get("meta", {}).get("start_date") == start_date
            and existing.get("meta", {}).get("end_date") == end_date
        ):
            log.info("Weather data already up to date — skipping fetch")
            return
        log.info("Date range has changed — re-fetching weather data")

    raw = fetch_weather(start_date, end_date)

    envelope = {
        "meta": {
            "source":            "Open-Meteo Historical Archive API",
            "url":               OPEN_METEO_URL,
            "latitude":          NYC_LAT,
            "longitude":         NYC_LON,
            "timezone":          "America/New_York",
            "start_date":        start_date,
            "end_date":          end_date,
            "stay_lookback_days": STAY_LOOKBACK_DAYS,
            "variables":         DAILY_VARIABLES,
            "fetched_at":        datetime.now(timezone.utc).isoformat(),
        },
        "daily": raw.get("daily", {}),
    }

    with open(out_path, "w") as f:
        json.dump(envelope, f, indent=2)

    n_days = len(envelope["daily"].get("time", []))
    log.info(f"Saved {n_days} daily weather records to {out_path}")

    write_lineage({
        "source":            "Open-Meteo Historical Archive API",
        "start_date":        start_date,
        "end_date":          end_date,
        "stay_lookback_days": STAY_LOOKBACK_DAYS,
        "n_days":            n_days,
        "variables":         DAILY_VARIABLES,
        "output":            str(out_path),
        "fetched_at":        envelope["meta"]["fetched_at"],
    })


if __name__ == "__main__":
    run()
