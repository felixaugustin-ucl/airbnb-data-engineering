"""
reload_postgres.py
------------------
Reloads the gold-layer Parquet files into PostgreSQL without re-running the
full Spark pipeline.

Use this when:
  - The PostgreSQL Docker volume was reset (docker-compose down -v)
  - The container was recreated with a fresh volume
  - You need to restore the star schema from the existing gold Parquet files

The gold Parquet files at project/data/gold/ are the source of truth.
This script reads them with pandas and writes to PostgreSQL — no Spark needed.

Runtime: ~30 seconds (no Spark startup overhead)

Usage:
    python3 data_pipeline/transformation/reload_postgres.py

Environment variables (from .env):
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
"""

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR       = DATA_PIPELINE_DIR.parent
REPO_ROOT         = PROJECT_DIR.parent
GOLD_DIR          = PROJECT_DIR / "data" / "gold"

# Load order matters — dimensions before facts (referential integrity)
TABLES = [
    "dim_date",
    "dim_neighbourhood",
    "dim_host",
    "dim_listing",
    "noise_borough_month",
    "fact_review",
    "fact_review_weather",
]


def get_engine():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db   = os.getenv("POSTGRES_DB",   "airbnb")
    user = os.getenv("POSTGRES_USER", "postgres")
    pw   = os.getenv("POSTGRES_PASSWORD", "postgres")
    url  = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"
    engine = create_engine(url, pool_pre_ping=True)
    log.info(f"Connected to PostgreSQL: {host}:{port}/{db}")
    return engine


def reload():
    load_dotenv(REPO_ROOT / ".env")
    engine = get_engine()

    for table in TABLES:
        parquet_dir = GOLD_DIR / table
        if not parquet_dir.exists():
            log.error(f"  Missing: {parquet_dir} — skipping")
            continue

        log.info(f"Loading {table} ...")
        df = pd.read_parquet(parquet_dir)

        # Spark collect_set() produces array columns — psycopg2 cannot serialize
        # numpy arrays. Convert any object columns containing lists/arrays to
        # JSON strings so they are stored as TEXT in PostgreSQL.
        import json, numpy as np
        for col in df.columns:
            if df[col].dtype == object:
                first_valid = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(first_valid, (list, np.ndarray)):
                    df[col] = df[col].apply(
                        lambda x: json.dumps(list(x)) if x is not None else None
                    )

        df.to_sql(table, engine, if_exists="replace", index=False,
                  method="multi", chunksize=10_000)
        log.info(f"  → {len(df):,} rows written to {table}")

    log.info("PostgreSQL reload complete.")


if __name__ == "__main__":
    reload()
