# NYC Airbnb Polyglot Analytics Platform

Multi-paradigm data engineering system for the UCL MSIN0166 individual assignment.
Three parallel pipelines ingest and transform NYC Airbnb data into a polyglot
persistence layer (PostgreSQL, MongoDB, ChromaDB) queryable by an MCP-powered agent.

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
├── data/                                       # ⚠ gitignored — reproduced by pipeline
│   ├── bronze/                                 #   Raw Parquet (immutable copy of source)
│   ├── silver/                                 #   Cleaned, typed, deduplicated Parquet
│   └── gold/                                   #   Kimball star schema Parquet
│
├── data_pipeline/
│   ├── ingestion/
│   │   ├── fetch_airbnb.py                     # Inside Airbnb snapshot discovery + download
│   │   ├── fetch_places.py                     # Google Places API — 5 POI types, 233 neighbourhoods
│   │   ├── fetch_wikipedia.py                  # Wikipedia intro sections via MediaWiki API
│   │   ├── fetch_weather.py                    # Open-Meteo historical daily weather
│   │   └── fetch_noise_complaints.py           # NYC 311 noise complaints via Socrata API
│   │
│   ├── transformation/
│   │   ├── transform_star_schema.py            # PySpark Bronze→Silver→Gold + PostgreSQL load
│   │   ├── load_mongodb.py                     # GeoJSON neighbourhood boundaries + Places → MongoDB
│   │   └── build_vector_index.py               # Guest reviews + Wikipedia → ChromaDB
│   │
│   ├── raw/                                    # ⚠ gitignored — populated by ingestion scripts
│   │   ├── airbnb/up_to_YYYY-MM-DD/            #   listings.csv.gz, reviews.csv.gz,
│   │   │                                       #   reviews.csv, neighbourhoods.geojson
│   │   ├── places/                             #   per-neighbourhood Google Places JSON
│   │   ├── wikipedia/                          #   per-neighbourhood Wikipedia JSON
│   │   ├── weather/nyc_weather.json            #   6,119 daily records (2009–2026)
│   │   └── nyc_open_data/                      #   1.8M NYC 311 noise complaints
│   │
│   └── processed/                              # ⚠ gitignored — populated by transformation scripts
│       ├── chromadb/                           #   ChromaDB HNSW vector indexes
│       │   ├── review_db                       #     56k guest reviews (stratified sample)
│       │   └── neighbourhood_context_db        #   598 Wikipedia paragraphs
│       ├── mongodb/lineage.json                #   MongoDB load lineage
│       ├── vectors/lineage.json                #   ChromaDB build lineage
│       └── star_schema/lineage.json            #   PySpark transformation lineage
│
├── mcp_server/
│   └── server.py                               # MCP server — query_warehouse, query_geodata, search_reviews
├── agent/
│   └── agent.py                                # LLM agent orchestrating MCP tools
│
├── docker/
│   ├── postgres/01_readonly_role.sql           # Creates airbnb_readonly role on first start
│   └── mongodb/01_readonly_user.js             # Creates airbnb_readonly user on first start
│
├── docker-compose.yml                          # PostgreSQL 15 + MongoDB 7.0 + Ollama service stack
├── Dockerfile                                  # MCP server + agent container (Python 3.12-slim)
├── requirements.txt                            # Full Python dependencies (ingestion + transformation)
├── requirements-agent.txt                      # Lean subset for the agent container
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
  → [adoptium.net](https://adoptium.net) — download the LTS installer for your platform.
  Verify with `java -version` after installing. PySpark will silently fail without it.

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
# 1. Copy the example env file and fill in your API keys (run from repo root)
cp .env.example .env        # Mac/Linux
copy .env.example .env      # Windows

# 2. Install Python dependencies (ingestion + transformation scripts)
cd project
pip install -r requirements.txt
```

Required keys — see `project/.env.example` for all variables:

| Key | Used by | Where to get it |
|-----|---------|-----------------|
| `GOOGLE_PLACES_API_KEY` | `fetch_places.py` | → [Google Cloud Console](https://console.cloud.google.com/apis/credentials) |
| `SOCRATA_APP_TOKEN` | `fetch_noise_complaints.py` | → [data.cityofnewyork.us](https://data.cityofnewyork.us/profile/edit/developer_settings) |
| `GROQ_API_KEY` | `agent.py` | → [console.groq.com](https://console.groq.com/keys) — free tier |

---

## `build_vector_index.py` Modes

| | Demo mode | Production mode |
|---|---|---|
| **Flag** | `--test` | _(no flag)_ |
| **Reviews encoded** | 500 (stratified sample) | 50,000 (stratified sample) |
| **ChromaDB path** | `processed/chromadb_test/` | `processed/chromadb/` |
| **Overwrites production index** | No | Yes |
| **Runtime** | ~1 minute | ~7 hours |

---

## Running the Pipeline

All commands run from the `project/` directory.

### Quick start

```bash
# Test mode — full pipeline, 500-review index, ~1 min vector build
python run_pipeline.py --test
make agent-test
```

```bash
# Production — full 50k-review index (~7 hrs vector build)
python run_pipeline.py
make agent
```

`run_pipeline.py` runs every step in order: starts Docker, ingestion,
PySpark transformation, MongoDB load, ChromaDB index build.

The agent uses Groq by default (`GROQ_API_KEY` in `.env`). If no key is set it
falls back to Ollama — pull the model first in that case:
`docker-compose exec ollama ollama pull llama3.2`

---

### Step-by-step (finer control)

```bash
make up

python data_pipeline/ingestion/fetch_airbnb.py
python data_pipeline/ingestion/fetch_places.py
python data_pipeline/ingestion/fetch_wikipedia.py
python data_pipeline/ingestion/fetch_weather.py
python data_pipeline/ingestion/fetch_noise_complaints.py

python data_pipeline/transformation/transform_star_schema.py         # ~5 min (PySpark)
python data_pipeline/transformation/load_mongodb.py                  # ~1 min
python data_pipeline/transformation/build_vector_index.py --test     # ~1 min (test)
python data_pipeline/transformation/build_vector_index.py            # ~7 hrs (full)

make agent-test    # agent against test index
make agent         # agent against full index
```

Ingestion scripts are idempotent — already-downloaded files are skipped on re-runs.

---

## Stopping Services

```bash
make down                    # stop containers, keep data volumes
docker-compose down -v       # ⚠ DESTRUCTIVE — deletes all database data
```

---