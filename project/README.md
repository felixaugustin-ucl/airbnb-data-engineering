# NYC Airbnb Hospitality Lakehouse

Multi-paradigm data engineering system for the UCL MSIN0166 individual assignment.
Three parallel pipelines ingest and transform NYC Airbnb data into a unified
lakehouse queryable by an MCP-powered agent.

---

## Architecture

```
Raw Sources → Ingestion → Bronze → Silver → Gold → PostgreSQL  (ETL, PySpark)
                       → MongoDB                               (ELT, geospatial)
                       → ChromaDB                             (ELT, vector)
                                  ↓
                            MCP Server
                                  ↓
                             LLM Agent
```

---

## Repository Structure

```
project/
├── data/                          # ⚠ gitignored — reproduced by pipeline
│   ├── bronze/                    #   Raw Parquet (immutable copy of source)
│   ├── silver/                    #   Cleaned, typed, deduplicated Parquet
│   └── gold/                      #   Kimball star schema Parquet
│
├── data_pipeline/
│   ├── ingestion/
│   │   ├── fetch_airbnb.py        # Inside Airbnb snapshot discovery + download
│   │   ├── fetch_places.py        # Google Places API — 5 POI types, 233 neighbourhoods
│   │   ├── fetch_wikipedia.py     # Wikipedia intro sections via MediaWiki API
│   │   ├── fetch_weather.py       # Open-Meteo historical daily weather
│   │   └── fetch_noise_complaints.py  # NYC 311 noise complaints via Socrata API
│   │
│   ├── transformation/
│   │   ├── transform_star_schema.py   # PySpark Bronze→Silver→Gold + PostgreSQL load
│   │   ├── load_mongodb.py            # GeoJSON neighbourhood boundaries + Places → MongoDB
│   │   └── build_vector_index.py      # Guest reviews + Wikipedia → ChromaDB
│   │
│   ├── raw/                       # ⚠ gitignored — populated by ingestion scripts
│   │   ├── airbnb/up_to_YYYY-MM-DD/   #   listings.csv.gz, reviews.csv.gz,
│   │   │                              #   reviews.csv, neighbourhoods.geojson
│   │   ├── places/                    #   per-neighbourhood Google Places JSON
│   │   ├── wikipedia/                 #   per-neighbourhood Wikipedia JSON
│   │   ├── weather/nyc_weather.json   #   6,119 daily records (2009–2026)
│   │   └── nyc_open_data/             #   1.8M NYC 311 noise complaints
│   │
│   └── processed/                 # ⚠ gitignored — populated by transformation scripts
│       ├── chromadb/              #   ChromaDB HNSW vector indexes
│       │   ├── review_db          #     56k guest reviews (stratified sample)
│       │   └── neighbourhood_context_db  # 598 Wikipedia paragraphs
│       ├── mongodb/lineage.json   #   MongoDB load lineage
│       ├── vectors/lineage.json   #   ChromaDB build lineage
│       └── star_schema/lineage.json   # PySpark transformation lineage
│
├── mcp_server/
│   └── server.py                  # MCP server — query_warehouse, query_geodata,
│                                  # search_reviews tools
├── agent/
│   └── agent.py                   # LLM agent orchestrating MCP tools
│
├── docker/
│   ├── postgres/01_readonly_role.sql  # Creates airbnb_readonly role on first start
│   └── mongodb/01_readonly_user.js    # Creates airbnb_readonly user on first start
│
├── docker-compose.yml             # PostgreSQL 15 + MongoDB 7.0 + Ollama service stack
├── Dockerfile                     # MCP server + agent container (Python 3.12-slim)
├── requirements.txt               # Full Python dependencies (ingestion + transformation)
├── requirements-agent.txt         # Lean subset for the agent container
└── README.md
```

---

## What is Gitignored and Why

| Path | Reason |
|------|--------|
| `data/bronze/`, `data/silver/`, `data/gold/` | Parquet files derived from raw sources — up to several GB. Reproduced by running `transform_star_schema.py`. |
| `data_pipeline/raw/` | Raw ingested data. The noise complaints file alone is ~400MB JSON. All data is re-fetchable via the ingestion scripts given valid API keys. |
| `data_pipeline/processed/chromadb/` | ChromaDB HNSW binary index files — not human-readable, not diffable, ~500MB. Reproduced by `build_vector_index.py`. |
| `data_pipeline/processed/*/lineage.json` | Runtime artefacts recording row counts and timestamps — not source code. |
| `.env` | API keys and database credentials (Google Places, Socrata, PostgreSQL, MongoDB). Never commit secrets. |

---

## Databases

| Store | Purpose | Docker service |
|-------|---------|----------------|
| PostgreSQL 15 | Kimball star schema — analytical queries | `airbnb_postgres` |
| MongoDB 7.0 | Geospatial operational store — neighbourhood boundaries, POI proximity | `airbnb_mongodb` |
| ChromaDB (local) | Vector indexes — semantic review search, neighbourhood context retrieval | no container (persistent client) |

---

## Prerequisites

### All platforms

- **Docker Desktop** — runs PostgreSQL, MongoDB, Ollama, and the agent container.
  Download from [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop).
  On Windows, enable WSL 2 backend in Docker Desktop settings.
- **Python 3.10–3.12** — required for ingestion and transformation scripts.
  Python 3.13 is **not supported** (`build_vector_index.py` uses PyTorch, which has
  no 3.13 wheel). Use [python.org](https://www.python.org/downloads/) or Anaconda.
- **Java 11+** — required by PySpark (`transform_star_schema.py` only).
  Download from [https://adoptium.net](https://adoptium.net).

### Windows-specific notes

- Use **Command Prompt**, **PowerShell**, or **Git Bash** — all work.
- Replace `python3` with `python` in all commands below if your PATH uses `python`.
- `JAVA_HOME` must point to your JDK installation and be on `PATH` for PySpark.
  Verify with: `java -version`
- If using Anaconda, activate your environment before running pipeline scripts:
  ```
  conda activate <your-env>
  python data_pipeline/transformation/build_vector_index.py --test
  ```
- The agent runs inside Docker (`docker-compose run --rm mcp-agent`) — no
  Windows-specific Python setup needed for the agent layer.

---

## Environment Setup

```bash
# 1. Copy the example env file and fill in your API keys
cp .env.example ../.env        # Mac/Linux
copy .env.example ..\.env      # Windows

# 2. Install Python dependencies (ingestion + transformation scripts)
pip install -r requirements.txt
```

Required keys — see `.env.example` for all variables:

| Key | Used by | Where to get it |
|-----|---------|-----------------|
| `GOOGLE_PLACES_API_KEY` | `fetch_places.py` | Google Cloud Console |
| `SOCRATA_APP_TOKEN` | `fetch_noise_complaints.py` | data.cityofnewyork.us |
| `GROQ_API_KEY` | `agent.py` (production) | console.groq.com — free tier |

---

## `build_vector_index.py` Modes

`build_vector_index.py` has two modes — choose based on your goal:

| | Demo mode | Production mode |
|---|---|---|
| **Flag** | `--test` | _(no flag)_ |
| **Reviews encoded** | 500 (stratified sample) | 50,000 (stratified sample) |
| **ChromaDB path** | `processed/chromadb_test/` | `processed/chromadb/` |
| **Overwrites production index** | No | Yes |
| **Runtime** | ~1 minute | ~7 hours |
| **Use when** | Verifying the pipeline end-to-end | Building the full index for deployment |

The demo mode exercises the full stratified sampling and encoding logic — it just caps at 500 reviews. All other pipeline steps are unaffected by this flag.

---

## Running the Pipeline

### Step 1 — Start databases

```bash
docker-compose up -d
```

This starts PostgreSQL, MongoDB, and Ollama. Database init scripts in
`docker/postgres/` and `docker/mongodb/` create the read-only roles
automatically on first start.

---

### Step 2 — Ingestion (run once, ~30 min total)

```bash
python data_pipeline/ingestion/fetch_airbnb.py
python data_pipeline/ingestion/fetch_places.py
python data_pipeline/ingestion/fetch_wikipedia.py
python data_pipeline/ingestion/fetch_weather.py
python data_pipeline/ingestion/fetch_noise_complaints.py
```

Scripts are idempotent — already-downloaded files are skipped on re-runs.

---

### Step 3 — Transformation

```bash
# Star schema (PySpark) — requires Java; ~5 min
python data_pipeline/transformation/transform_star_schema.py

# MongoDB load — ~1 min
python data_pipeline/transformation/load_mongodb.py

# Vector index — requires Python ≤ 3.12; demo mode ~1 min, full ~7 hrs
python data_pipeline/transformation/build_vector_index.py --test   # demo
python data_pipeline/transformation/build_vector_index.py          # full
```

---

### Step 4 — Run the agent

The agent runs inside the Docker container defined in `Dockerfile`.
This is the recommended path on all platforms (Mac, Windows, Linux) —
no local Python setup required for the agent layer.

```bash
# Pull the Ollama model (first time only — ~2 GB download)
docker-compose exec ollama ollama pull llama3.2

# Run the interactive agent
docker-compose run --rm mcp-agent
```

The container connects to PostgreSQL, MongoDB, and Ollama via the Docker
internal network. ChromaDB is mounted from
`data_pipeline/processed/chromadb/` (or `chromadb_test/` in demo mode).

> **Windows note:** if `chromadb_test/` was used in Step 3, update the
> volume mount in `docker-compose.yml` temporarily:
> `./data_pipeline/processed/chromadb_test:/app/chromadb`

---

## Stopping Services

```bash
docker-compose down          # stop containers, keep data volumes
docker-compose down -v       # ⚠ DESTRUCTIVE — deletes all database data
```

---
