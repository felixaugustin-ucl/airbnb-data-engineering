"""
fetch_wikipedia.py
------------------
Scrapes the introductory section (above the first heading) of the Wikipedia
article for each NYC neighbourhood and saves the raw text to:
    data_pipeline/raw/wikipedia/<neighbourhood_name>.json

Neighbourhood names are read from the latest Airbnb snapshot's
neighbourhoods.geojson, matching the same source used by fetch_places.py.

Only the intro section is scraped — this is where descriptive language
("affluent", "gentrified", "historic", etc.) is concentrated.

Also writes a lineage.json to raw/wikipedia/ on each run.

Usage:
    python fetch_wikipedia.py
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
RAW_AIRBNB_DIR    = DATA_PIPELINE_DIR / "raw" / "airbnb"
RAW_WIKI_DIR      = DATA_PIPELINE_DIR / "raw" / "wikipedia"

# ── Config ─────────────────────────────────────────────────────────────────────

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
SLEEP_BETWEEN_REQUESTS = 0.5  # seconds — polite crawl rate

# Wikipedia requires a descriptive User-Agent for bots
HEADERS = {
    "User-Agent": "NYC-Airbnb-DataPipeline/1.0 (academic project; contact: student@ucl.ac.uk)"
}


# ── Snapshot detection ─────────────────────────────────────────────────────────

def latest_airbnb_snapshot() -> Path:
    snapshots = sorted(RAW_AIRBNB_DIR.glob("up_to_*"))
    if not snapshots:
        raise FileNotFoundError(f"No Airbnb snapshots found in {RAW_AIRBNB_DIR}")
    latest = snapshots[-1]
    log.info(f"Using snapshot: {latest.name}")
    return latest


# ── Neighbourhood loading ──────────────────────────────────────────────────────

def load_neighbourhoods(geojson_path: Path) -> list[dict]:
    """Return list of {name, borough} dicts from neighbourhoods.geojson."""
    with open(geojson_path) as f:
        fc = json.load(f)
    neighbourhoods = [
        {
            "name":    feat["properties"]["neighbourhood"],
            "borough": feat["properties"]["neighbourhood_group"],
        }
        for feat in fc["features"]
    ]
    log.info(f"Loaded {len(neighbourhoods)} neighbourhoods")
    return neighbourhoods


# ── Wikipedia scraping ─────────────────────────────────────────────────────────

NYC_KEYWORDS = {
    "new york", "nyc", "manhattan", "brooklyn", "queens",
    "bronx", "staten island", "new york city",
}


def is_nyc_article(intro_text: str) -> bool:
    """Return True if the intro text contains at least one NYC-related keyword."""
    text_lower = intro_text.lower()
    return any(kw in text_lower for kw in NYC_KEYWORDS)


def search_wikipedia(name: str, borough: str) -> str | None:
    """
    Search Wikipedia using both neighbourhood name and borough to avoid
    matching same-named places outside NYC (e.g. Bayswater, Perth).
    Returns the best matching page title or None.
    """
    query = f"{name}, {borough}, New York City"
    params = {
        "action":   "query",
        "list":     "search",
        "srsearch": query,
        "srlimit":  1,
        "format":   "json",
    }
    try:
        resp = requests.get(WIKIPEDIA_API, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        if results:
            return results[0]["title"]
    except requests.RequestException as e:
        log.warning(f"Search failed for '{query}': {e}")
    return None


def fetch_intro(page_title: str) -> dict | None:
    """
    Fetch the HTML of a Wikipedia page and extract the introductory section
    (all paragraphs before the first heading tag).

    Returns a dict with:
        title, url, intro_paragraphs (list of str), intro_text (joined str)
    or None on failure.
    """
    params = {
        "action":    "parse",
        "page":      page_title,
        "prop":      "text",
        "format":    "json",
        "redirects": True,
    }
    try:
        resp = requests.get(WIKIPEDIA_API, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        log.warning(f"Fetch failed for '{page_title}': {e}")
        return None

    if "error" in data:
        log.warning(f"Wikipedia API error for '{page_title}': {data['error']}")
        return None

    html = data.get("parse", {}).get("text", {}).get("*", "")
    soup = BeautifulSoup(html, "html.parser")

    # Collect paragraphs until we hit the first heading
    intro_paragraphs = []
    for tag in soup.find_all(["p", "h2", "h3"]):
        if tag.name in ("h2", "h3"):
            break
        text = tag.get_text(separator=" ", strip=True)
        if text:
            intro_paragraphs.append(text)

    if not intro_paragraphs:
        log.warning(f"No intro paragraphs found for '{page_title}'")
        return None

    page_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"

    return {
        "title":             page_title,
        "url":               page_url,
        "intro_paragraphs":  intro_paragraphs,
        "intro_text":        "\n\n".join(intro_paragraphs),
    }


# ── Safe filename ──────────────────────────────────────────────────────────────

def safe_filename(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-")


# ── Lineage ────────────────────────────────────────────────────────────────────

def write_lineage(lineage: dict):
    RAW_WIKI_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_WIKI_DIR / "lineage.json"
    with open(path, "w") as f:
        json.dump(lineage, f, indent=2)
    log.info(f"Lineage written to {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    RAW_WIKI_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_dir = latest_airbnb_snapshot()
    geojson_path = snapshot_dir / "neighbourhoods.geojson"
    neighbourhoods = load_neighbourhoods(geojson_path)

    run_start = datetime.now(timezone.utc).isoformat()
    lineage = {
        "source":           "Wikipedia (via MediaWiki API)",
        "geojson_snapshot": snapshot_dir.name,
        "run_started_at":   run_start,
        "neighbourhoods":   {},
    }

    total_ok = total_skipped = total_not_found = total_failed = 0

    for nb in neighbourhoods:
        name    = nb["name"]
        borough = nb["borough"]
        out_path = RAW_WIKI_DIR / f"{safe_filename(name)}.json"

        if out_path.exists():
            log.info(f"Skipping (already downloaded): {name}")
            lineage["neighbourhoods"][name] = {"status": "skipped", "file": str(out_path)}
            total_skipped += 1
            continue

        log.info(f"Searching Wikipedia for: {name} ({borough})")

        page_title = search_wikipedia(name, borough)
        if not page_title:
            log.warning(f"No Wikipedia article found for: {name}")
            lineage["neighbourhoods"][name] = {"status": "not_found"}
            total_not_found += 1
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            continue

        intro = fetch_intro(page_title)
        if not intro:
            log.warning(f"Could not extract intro for: {name} (page: {page_title})")
            lineage["neighbourhoods"][name] = {"status": "error", "page_title": page_title}
            total_failed += 1
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            continue

        # Validate the article is actually about an NYC neighbourhood
        if not is_nyc_article(intro["intro_text"]):
            log.warning(f"Article '{page_title}' does not appear to be NYC — skipping: {name}")
            lineage["neighbourhoods"][name] = {
                "status":     "wrong_location",
                "page_title": page_title,
                "url":        intro["url"],
            }
            total_not_found += 1
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            continue

        envelope = {
            "neighbourhood":    name,
            "borough":          borough,
            "wikipedia_title":  intro["title"],
            "wikipedia_url":    intro["url"],
            "fetched_at":       datetime.now(timezone.utc).isoformat(),
            "intro_paragraphs": intro["intro_paragraphs"],
            "intro_text":       intro["intro_text"],
            "paragraph_count":  len(intro["intro_paragraphs"]),
            "char_count":       len(intro["intro_text"]),
        }

        with open(out_path, "w") as f:
            json.dump(envelope, f, indent=2, ensure_ascii=False)

        log.info(f"  → {len(intro['intro_paragraphs'])} paragraphs, {len(intro['intro_text'])} chars → {out_path.name}")
        lineage["neighbourhoods"][name] = {
            "status":          "ok",
            "file":            str(out_path),
            "wikipedia_title": intro["title"],
            "wikipedia_url":   intro["url"],
            "fetched_at":      envelope["fetched_at"],
            "paragraph_count": envelope["paragraph_count"],
            "char_count":      envelope["char_count"],
        }
        total_ok += 1
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    lineage["run_finished_at"] = datetime.now(timezone.utc).isoformat()
    lineage["summary"] = {
        "total":      len(neighbourhoods),
        "ok":         total_ok,
        "skipped":    total_skipped,
        "not_found":  total_not_found,
        "failed":     total_failed,
    }

    write_lineage(lineage)
    log.info(
        f"Done. {total_ok} fetched, {total_skipped} skipped, "
        f"{total_not_found} not found, {total_failed} failed."
    )


if __name__ == "__main__":
    run()
