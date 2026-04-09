"""
build_vector_index.py
---------------------
ELT pipeline: encodes text data into two separate ChromaDB vector indexes.

Architecture justification (ELT + two separate indexes):
  Two indexes are maintained rather than one combined index because retrieval
  intent differs: reviewDB supports experiential queries ("listings praised
  for quietness") while neighbourhood_context_db supports contextual queries
  ("gentrifying neighbourhoods"). Mixing document types in a single index
  degrades retrieval precision (Karpukhin et al., 2020, DPR paper).

Encoder: multi-qa-mpnet-base-dot-v1
  Fine-tuned on 215M QA pairs — optimised for retrieval tasks where a short
  agent query must match longer passages. Uses dot product similarity.
  Runs locally on CPU (~420MB) — no data sent to remote providers, consistent
  with the brief's data privacy guidance.

Indexes:
  review_db              — guest review text from listings.csv.gz
                           metadata: listing_id, review_date, neighbourhood
  neighbourhood_context_db — Wikipedia intro sections per neighbourhood
                              metadata: neighbourhood, borough, wikipedia_url

Chunking strategy:
  Reviews: one document per review comment. Reviews are naturally short
    (median ~150 words) so chunking would fragment context.
  Wikipedia: one document per intro paragraph. Paragraphs are coherent
    semantic units. Citation markers ([1], [2]) are stripped before encoding
    as they carry no semantic content.

Stratified sampling for reviews:
  Full reviews.csv.gz (~1M rows) is too large to encode on CPU in reasonable
  time. We cap at MAX_REVIEWS using two-level proportional stratification
  (Cochran, 1977):

    Level 1 — neighbourhood quota:
      quota_nb = MAX_REVIEWS × (nb_reviews / total_valid_reviews)
      A MIN_REVIEWS_PER_NB floor ensures rare neighbourhoods retain enough
      documents for meaningful retrieval.

    Level 2 — per-listing quota within neighbourhood:
      quota_listing = quota_nb × (listing_reviews / nb_reviews)
      This prevents any single popular listing from dominating its
      neighbourhood's allocation, ensuring individual Airbnb properties
      are each represented rather than a handful of high-volume listings.

  Selection is systematic (every kth review per listing) so samples are
  spread across the full review date range rather than clustered at the start.
  Two streaming passes over reviews.csv.gz — no full dataset loaded into RAM.

Usage:
    python build_vector_index.py

Environment variables (from .env):
    CHROMA_PATH   — path for ChromaDB persistence (default: project/data/chromadb)
    MAX_REVIEWS   — total review cap after stratified sampling (default: 50000)
"""

import argparse
import json
import logging
import math
import os
import re
import gzip
import csv
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR       = DATA_PIPELINE_DIR.parent
REPO_ROOT         = PROJECT_DIR.parent

RAW_AIRBNB_DIR    = DATA_PIPELINE_DIR / "raw" / "airbnb"
RAW_WIKI_DIR      = DATA_PIPELINE_DIR / "raw" / "wikipedia"

LINEAGE_PATH      = DATA_PIPELINE_DIR / "processed" / "vectors" / "lineage.json"

# ── Config ─────────────────────────────────────────────────────────────────────

# Model fine-tuned on 215M QA pairs — optimised for dot-product retrieval
ENCODER_MODEL  = "multi-qa-mpnet-base-dot-v1"

# Batch size for encoding — adjust down if RAM is constrained on CPU
ENCODE_BATCH   = 64

# ChromaDB collection names
REVIEW_COLLECTION   = "review_db"
WIKI_COLLECTION     = "neighbourhood_context_db"

# Citation pattern: [1], [2], [ 5 ] etc. — stripped before encoding
CITATION_RE = re.compile(r"\[\s*\d+\s*\]")

# ── Stratified sampling config ─────────────────────────────────────────────────

# Total reviews to encode (override via MAX_REVIEWS env var)
MAX_REVIEWS_DEFAULT = 50_000

# Minimum reviews per neighbourhood — prevents zero allocation for small areas
MIN_REVIEWS_PER_NB  = 10


# ── Encoder ────────────────────────────────────────────────────────────────────

def load_encoder() -> SentenceTransformer:
    """
    Load the sentence transformer model.
    On first run, the model is downloaded (~420MB) and cached locally.
    Subsequent runs load from cache — no internet required.

    IMPORTANT: The same model MUST be used for both indexing and querying.
    Using different encoders for documents and queries produces incomparable
    vector spaces — similarity scores would be meaningless.
    """
    log.info(f"Loading encoder: {ENCODER_MODEL}")
    model = SentenceTransformer(ENCODER_MODEL)
    log.info("  Encoder loaded.")
    return model


# ── ChromaDB client ────────────────────────────────────────────────────────────

def get_chroma_client(persist_path: str) -> chromadb.ClientAPI:
    """
    Create a persistent ChromaDB client.
    Data is written to disk at persist_path — survives process restarts.
    """
    Path(persist_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_path)
    log.info(f"ChromaDB client initialised at {persist_path}")
    return client


# ── Snapshot detection ─────────────────────────────────────────────────────────

def latest_snapshot() -> Path:
    snapshots = sorted(RAW_AIRBNB_DIR.glob("up_to_*"))
    if not snapshots:
        raise FileNotFoundError(f"No Airbnb snapshots in {RAW_AIRBNB_DIR}")
    return snapshots[-1]


def load_neighbourhood_borough_map(snapshot: Path) -> dict[str, str]:
    """Build neighbourhood → borough lookup from GeoJSON for metadata."""
    geojson = snapshot / "neighbourhoods.geojson"
    with open(geojson) as f:
        fc = json.load(f)
    return {
        feat["properties"]["neighbourhood"]: feat["properties"]["neighbourhood_group"]
        for feat in fc["features"]
    }


def load_neighbourhood_from_listings(snapshot: Path) -> dict[int, str]:
    """
    Build listing_id → neighbourhood lookup from listings.csv for review metadata.
    Uses the lightweight visualisations listings.csv (not .gz) as we only
    need id and neighbourhood columns.
    """
    nb_map = {}
    listings_path = snapshot / "listings.csv"
    with open(listings_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                nb_map[int(row["id"])] = row["neighbourhood"]
            except (ValueError, KeyError):
                pass
    log.info(f"  Loaded neighbourhood map for {len(nb_map)} listings")
    return nb_map


# ══════════════════════════════════════════════════════════════════════════════
# REVIEW INDEX (review_db)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_listing_quotas(
    reviews_gz: Path,
    nb_map: dict[int, str],
    max_reviews: int,
) -> tuple[dict[tuple, int], dict[tuple, int]]:
    """
    Pass 1 — count reviews, then compute two-level stratified quotas.

    Returns:
      listing_quotas  — {(neighbourhood, listing_id): quota_to_keep}
      listing_counts  — {(neighbourhood, listing_id): total_valid_reviews}

    Level 1: neighbourhood quota proportional to its share of total reviews,
             with MIN_REVIEWS_PER_NB floor so small neighbourhoods are not
             excluded entirely.
    Level 2: within each neighbourhood, distribute the quota across listings
             proportionally to each listing's review count, ensuring individual
             Airbnb properties each contribute rather than only the busiest ones.
    """
    log.info("  Pass 1: counting reviews per (neighbourhood, listing) ...")

    listing_counts: dict[tuple, int] = {}
    nb_counts: dict[str, int] = {}

    with gzip.open(reviews_gz, "rt", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            comment = row.get("comments", "").strip()
            if not comment or len(comment) < 20:
                continue
            try:
                listing_id = int(row["listing_id"])
            except (ValueError, KeyError):
                continue
            nb = nb_map.get(listing_id, "Unknown")
            key = (nb, listing_id)
            listing_counts[key] = listing_counts.get(key, 0) + 1
            nb_counts[nb] = nb_counts.get(nb, 0) + 1

    total_valid = sum(nb_counts.values())
    log.info(
        f"  Pass 1 complete: {total_valid:,} valid reviews across "
        f"{len(nb_counts)} neighbourhoods, {len(listing_counts):,} listings"
    )

    # ── Level 1: neighbourhood quotas ─────────────────────────────────────────
    # Proportional allocation with minimum floor
    nb_quotas: dict[str, int] = {}
    for nb, count in nb_counts.items():
        nb_quotas[nb] = max(
            MIN_REVIEWS_PER_NB,
            math.floor(max_reviews * count / total_valid),
        )

    # If the MIN floor caused the total to exceed max_reviews, scale down
    # (keeping the floor so all neighbourhoods retain at least MIN reviews)
    total_nb_quota = sum(nb_quotas.values())
    if total_nb_quota > max_reviews:
        scale = max_reviews / total_nb_quota
        for nb in nb_quotas:
            nb_quotas[nb] = max(
                MIN_REVIEWS_PER_NB,
                math.floor(nb_quotas[nb] * scale),
            )

    # ── Level 2: per-listing quotas within each neighbourhood ─────────────────
    # Build neighbourhood → listing sub-dict first for efficiency
    nb_listings: dict[str, dict[tuple, int]] = {}
    for key, cnt in listing_counts.items():
        nb = key[0]
        nb_listings.setdefault(nb, {})[key] = cnt

    listing_quotas: dict[tuple, int] = {}
    for nb, nb_quota in nb_quotas.items():
        listings = nb_listings.get(nb, {})
        nb_total  = sum(listings.values())
        for key, lcount in listings.items():
            # Proportional share of neighbourhood quota, at least 1, at most
            # the listing's actual review count (can't keep more than exist)
            prop = max(1, math.floor(nb_quota * lcount / nb_total))
            listing_quotas[key] = min(prop, lcount)

    total_quota = sum(listing_quotas.values())
    log.info(f"  Quota computed: {total_quota:,} reviews to encode")

    return listing_quotas, listing_counts


def build_review_index(
    client: chromadb.ClientAPI,
    encoder: SentenceTransformer,
    snapshot: Path,
    max_reviews: int,
) -> dict:
    """
    Encode a stratified sample of guest reviews into the review_db collection.

    Sampling strategy — two-level proportional stratification (Cochran, 1977):
      Level 1 (neighbourhood): quota proportional to neighbourhood review share.
        Small neighbourhoods get a MIN_REVIEWS_PER_NB floor so they are not
        lost in the index.
      Level 2 (listing within neighbourhood): quota proportional to the listing's
        share of its neighbourhood's reviews. Prevents high-volume listings from
        consuming an entire neighbourhood's allocation.

    Selection within each listing uses systematic (every-kth) sampling so the
    chosen reviews are spread across the full date range, not clustered at the
    start of the file.

    Two streaming passes over reviews.csv.gz — no full dataset held in RAM.

    Document structure:
      - id:       "{listing_id}_{review_date}"  — stable, unique per review
      - document: review comment text
      - metadata: listing_id, review_date, neighbourhood

    Idempotency: collection deleted and rebuilt on each run.
    """
    log.info(f"Building review_db vector index (max_reviews={max_reviews:,})")

    # Delete existing collection for clean rebuild
    try:
        client.delete_collection(REVIEW_COLLECTION)
        log.info("  Deleted existing review_db collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=REVIEW_COLLECTION,
        metadata={"hnsw:space": "ip"},  # dot product — matches encoder objective
    )

    nb_map = load_neighbourhood_from_listings(snapshot)

    reviews_gz = snapshot / "reviews.csv.gz"
    if not reviews_gz.exists():
        log.warning(f"reviews.csv.gz not found at {reviews_gz} — skipping review index")
        return {"status": "skipped", "reason": "reviews.csv.gz not found"}

    # ── Pass 1: compute per-listing quotas ────────────────────────────────────
    listing_quotas, listing_counts = _compute_listing_quotas(
        reviews_gz, nb_map, max_reviews
    )

    # ── Pass 2: systematic selection + encode ─────────────────────────────────
    # For each listing with quota k out of n total reviews, keep every
    # floor(n/k)-th review (systematic sampling). Tracker stores how many
    # reviews we have seen and kept per listing.
    seen:  dict[tuple, int] = {}   # (nb, listing_id) -> reviews seen so far
    kept:  dict[tuple, int] = {}   # (nb, listing_id) -> reviews kept so far

    ids, documents, metadatas = [], [], []
    batch_count = 0

    log.info("  Pass 2: streaming reviews with systematic stratified selection ...")

    with gzip.open(reviews_gz, "rt", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            comment = row.get("comments", "").strip()
            if not comment or len(comment) < 20:
                continue
            try:
                listing_id  = int(row["listing_id"])
                review_date = row["date"].strip()
            except (ValueError, KeyError):
                continue

            nb  = nb_map.get(listing_id, "Unknown")
            key = (nb, listing_id)

            quota = listing_quotas.get(key, 0)
            if quota == 0:
                continue

            total = listing_counts[key]
            seen[key] = seen.get(key, 0) + 1
            prev_seen = seen[key] - 1

            # Systematic (Bresenham-style) keep decision:
            #   keep this review if floor(seen * quota/total) > floor(prev * quota/total)
            # This distributes exactly `quota` selections evenly across `total` positions.
            if math.floor(seen[key] * quota / total) <= math.floor(prev_seen * quota / total):
                continue

            kept[key] = kept.get(key, 0) + 1
            if kept[key] > quota:
                # Safety guard — should not trigger with correct arithmetic
                continue

            doc_id = f"{listing_id}_{review_date}"
            ids.append(doc_id)
            documents.append(comment)
            metadatas.append({
                "listing_id":    listing_id,
                "review_date":   review_date,
                "neighbourhood": nb,
            })

            if len(ids) >= ENCODE_BATCH * 10:
                _upsert_batch(collection, encoder, ids, documents, metadatas)
                batch_count += len(ids)
                log.info(f"  Encoded {batch_count:,} reviews so far ...")
                ids, documents, metadatas = [], [], []

    # Final batch
    if ids:
        _upsert_batch(collection, encoder, ids, documents, metadatas)
        batch_count += len(ids)

    nb_covered = len({k[0] for k in kept})
    log.info(
        f"  → review_db: {batch_count:,} documents indexed "
        f"across {nb_covered} neighbourhoods"
    )
    return {
        "collection":          REVIEW_COLLECTION,
        "documents_indexed":   batch_count,
        "max_reviews_cap":     max_reviews,
        "neighbourhoods_covered": nb_covered,
        "sampling_strategy":   (
            "two-level proportional stratification: "
            "Level 1 = neighbourhood quota (proportional + MIN floor), "
            "Level 2 = per-listing quota within neighbourhood (proportional); "
            "selection = systematic every-kth review per listing"
        ),
        "encoder":             ENCODER_MODEL,
        "distance_metric":     "dot_product (inner product)",
        "chunking":            "one document per review comment",
    }


# ══════════════════════════════════════════════════════════════════════════════
# NEIGHBOURHOOD CONTEXT INDEX (neighbourhood_context_db)
# ══════════════════════════════════════════════════════════════════════════════

def build_wiki_index(client: chromadb.ClientAPI, encoder: SentenceTransformer,
                     snapshot: Path) -> dict:
    """
    Encode Wikipedia neighbourhood intro sections into neighbourhood_context_db.

    Document structure:
      - id:       "{neighbourhood_name}_para_{i}"  — one per paragraph
      - document: cleaned paragraph text (citations stripped)
      - metadata: neighbourhood, borough, wikipedia_url, paragraph_index

    Chunking: one document per intro paragraph.
    Paragraphs are coherent semantic units — each typically describes one
    aspect of the neighbourhood (geography, demographics, history, character).
    Encoding at paragraph level preserves that coherence while keeping
    documents within the encoder's 512-token context window.

    Citation stripping: markers like [1], [ 5 ] are removed before encoding.
    They carry no semantic content and would add noise to the embedding.
    """
    log.info("Building neighbourhood_context_db vector index")

    try:
        client.delete_collection(WIKI_COLLECTION)
        log.info("  Deleted existing neighbourhood_context_db collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=WIKI_COLLECTION,
        metadata={"hnsw:space": "ip"}
    )

    borough_map = load_neighbourhood_borough_map(snapshot)

    ids, documents, metadatas = [], [], []
    nb_count  = 0
    doc_count = 0

    wiki_files = sorted(RAW_WIKI_DIR.glob("*.json"))
    wiki_files = [f for f in wiki_files if f.name != "lineage.json"]

    for wiki_file in wiki_files:
        with open(wiki_file, encoding="utf-8") as f:
            data = json.load(f)

        neighbourhood = data.get("neighbourhood", "")
        borough       = data.get("borough") or borough_map.get(neighbourhood, "Unknown")
        wiki_url      = data.get("wikipedia_url", "")
        paragraphs    = data.get("intro_paragraphs", [])

        if not paragraphs:
            continue

        for i, para in enumerate(paragraphs):
            # Strip citation markers — they add noise, no semantic value
            clean_para = CITATION_RE.sub("", para).strip()
            if len(clean_para) < 30:
                # Skip trivially short paragraphs (e.g. coordinate-only stubs)
                continue

            doc_id = f"{neighbourhood}_para_{i}"
            ids.append(doc_id)
            documents.append(clean_para)
            metadatas.append({
                "neighbourhood":   neighbourhood,
                "borough":         borough,
                "wikipedia_url":   wiki_url,
                "paragraph_index": i,
            })
            doc_count += 1

        nb_count += 1

    # Encode all Wikipedia paragraphs in one pass — small dataset (~700 docs)
    if ids:
        _upsert_batch(collection, encoder, ids, documents, metadatas)

    log.info(f"  → neighbourhood_context_db: {doc_count} paragraphs from {nb_count} neighbourhoods")
    return {
        "collection":        WIKI_COLLECTION,
        "neighbourhoods":    nb_count,
        "paragraphs_indexed": doc_count,
        "encoder":           ENCODER_MODEL,
        "distance_metric":   "dot_product (inner product)",
        "chunking":          "one document per intro paragraph, citations stripped",
    }


# ── Batch encode + upsert helper ───────────────────────────────────────────────

def _upsert_batch(collection, encoder: SentenceTransformer,
                  ids: list, documents: list, metadatas: list):
    """
    Encode a batch of documents and upsert into ChromaDB.
    Using upsert (not add) ensures idempotency within a run.
    """
    embeddings = encoder.encode(
        documents,
        batch_size=ENCODE_BATCH,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).tolist()

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


# ── Lineage ────────────────────────────────────────────────────────────────────

def write_lineage(lineage: dict):
    LINEAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LINEAGE_PATH, "w") as f:
        json.dump(lineage, f, indent=2)
    log.info(f"Lineage written to {LINEAGE_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    parser = argparse.ArgumentParser(description="Build ChromaDB vector indexes.")
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Test mode: encode only 500 reviews, write to chromadb_test/ "
            "(never overwrites the production index)."
        ),
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    if args.test:
        max_reviews = 500
        chroma_path = str(DATA_PIPELINE_DIR / "processed" / "chromadb_test")
        log.info("TEST MODE — max_reviews=500, path=%s", chroma_path)
    else:
        max_reviews = int(os.getenv("MAX_REVIEWS", MAX_REVIEWS_DEFAULT))
        chroma_path = os.getenv(
            "CHROMA_PATH",
            str(DATA_PIPELINE_DIR / "processed" / "chromadb")
        )

    snapshot  = latest_snapshot()
    run_start = datetime.now(timezone.utc).isoformat()
    lineage   = {
        "run_started_at": run_start,
        "encoder":        ENCODER_MODEL,
        "max_reviews":    max_reviews,
        "test_mode":      args.test,
    }

    # Load encoder once — reused for both indexes
    encoder = load_encoder()
    client  = get_chroma_client(chroma_path)

    lineage["review_db"]                = build_review_index(client, encoder, snapshot, max_reviews)
    lineage["neighbourhood_context_db"] = build_wiki_index(client, encoder, snapshot)

    lineage["run_finished_at"] = datetime.now(timezone.utc).isoformat()
    write_lineage(lineage)
    log.info("Vector index build complete.")


if __name__ == "__main__":
    run()
