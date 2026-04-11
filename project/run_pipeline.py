"""
run_pipeline.py
---------------
Runs the full NYC Airbnb Polyglot Analytics Platform pipeline end-to-end.

Steps executed in order:
  1. docker-compose up        — start PostgreSQL, MongoDB, Ollama
  2. Ingestion                — fetch Airbnb, Places, Wikipedia, Weather, Noise
  3. transform_star_schema    — PySpark Bronze→Silver→Gold + PostgreSQL load
  4. load_mongodb             — GeoJSON + Places → MongoDB
  5. build_vector_index       — reviews + Wikipedia → ChromaDB

Usage:
    python run_pipeline.py           # full pipeline (50k reviews, ~7 hrs)
    python run_pipeline.py --test    # test mode  (500 reviews,  ~1 min)

Flags:
    --test          Build a small ChromaDB index (500 reviews) at
                    processed/chromadb_test/ instead of the full 50k index.
                    All other steps are identical.
    --skip-ingest   Skip ingestion (re-use previously downloaded raw data).
    --skip-docker   Skip `docker-compose up` (services already running).
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def _docker_compose_cmd() -> list[str]:
    """
    Return the correct docker compose command for this system.
    Docker Desktop v2+ uses 'docker compose' (plugin); older installs use
    'docker-compose' (standalone binary). Try plugin form first.
    """
    try:
        subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True, check=True
        )
        return ["docker", "compose"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ["docker-compose"]


DOCKER_COMPOSE = _docker_compose_cmd()


def run(cmd: list[str], **kwargs):
    """Run a command, streaming output. Exit on failure."""
    print(f"\n{'─'*60}")
    print(f"  {' '.join(cmd)}")
    print(f"{'─'*60}")
    result = subprocess.run(cmd, cwd=PROJECT_DIR, **kwargs)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed (exit {result.returncode}): {' '.join(cmd)}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full NYC Airbnb Polyglot Analytics Platform pipeline."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: 500-review ChromaDB index at processed/chromadb_test/",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingestion — re-use previously downloaded raw data.",
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip docker-compose up — services already running.",
    )
    args = parser.parse_args()

    mode = "TEST MODE (500 reviews)" if args.test else "PRODUCTION MODE (50k reviews)"
    print(f"\nNYC Airbnb Polyglot Analytics Platform — {mode}\n")

    # ── Step 1: Docker ─────────────────────────────────────────────────────────
    if not args.skip_docker:
        run(DOCKER_COMPOSE + ["up", "-d"])

    # ── Step 2: Ingestion ──────────────────────────────────────────────────────
    if not args.skip_ingest:
        ingestion_scripts = [
            "data_pipeline/ingestion/fetch_airbnb.py",
            "data_pipeline/ingestion/fetch_places.py",
            "data_pipeline/ingestion/fetch_wikipedia.py",
            "data_pipeline/ingestion/fetch_weather.py",
            "data_pipeline/ingestion/fetch_noise_complaints.py",
        ]
        for script in ingestion_scripts:
            run([sys.executable, script])

    # ── Step 3: Star schema (PySpark) ──────────────────────────────────────────
    run([sys.executable, "data_pipeline/transformation/transform_star_schema.py"])

    # ── Step 4: MongoDB ────────────────────────────────────────────────────────
    run([sys.executable, "data_pipeline/transformation/load_mongodb.py"])

    # ── Step 5: Vector index ───────────────────────────────────────────────────
    vector_cmd = [sys.executable, "data_pipeline/transformation/build_vector_index.py"]
    if args.test:
        vector_cmd.append("--test")
    run(vector_cmd)

    print(f"\n{'═'*60}")
    print(f"  Pipeline complete — {mode}")
    if args.test:
        print("  ChromaDB index at: data_pipeline/processed/chromadb_test/")
        print("  To run agent:  make agent-test")
    else:
        print("  ChromaDB index at: data_pipeline/processed/chromadb/")
        print("  To run agent:  make agent")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
