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
├── docker-compose.yml             # PostgreSQL 15 + MongoDB 7.0 service stack
├── requirements.txt               # Python dependencies
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

Start all services: `docker-compose up -d`

---

## Interpreter Requirements

| Script | Interpreter | Reason |
|--------|------------|--------|
| All ingestion scripts | `/usr/local/bin/python3` (3.13) | Standard |
| `transform_star_schema.py` | `/usr/local/bin/python3` (3.13) | PySpark |
| `load_mongodb.py` | `/usr/local/bin/python3` (3.13) | Standard |
| `build_vector_index.py` | `/opt/anaconda3/bin/python3.12` | PyTorch has no wheel for Python 3.13 |

---

## Running the Pipeline

```bash
# 1. Start databases
docker-compose up -d

# 2. Ingestion (run once — takes ~30 min total)
python3 data_pipeline/ingestion/fetch_airbnb.py
python3 data_pipeline/ingestion/fetch_places.py
python3 data_pipeline/ingestion/fetch_wikipedia.py
python3 data_pipeline/ingestion/fetch_weather.py
python3 data_pipeline/ingestion/fetch_noise_complaints.py

# 3. Transformation
python3 data_pipeline/transformation/transform_star_schema.py  # ~5 min
python3 data_pipeline/transformation/load_mongodb.py           # ~1 min
/opt/anaconda3/bin/python3.12 data_pipeline/transformation/build_vector_index.py  # ~7 hrs
```
