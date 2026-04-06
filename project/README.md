# Project

## Structure

Follows the **medallion architecture** (bronze → silver → gold).

```
project/
├── data/
│   ├── bronze/              # Raw ingested data (immutable)
│   ├── silver/              # Cleaned / validated data
│   └── gold/                # Aggregated, query-ready data
├── data_pipeline/
│   ├── ingestion/
│   │   ├── ingest_sql.py    # Ingest from SQL source
│   │   └── ingest_api.py    # Ingest from external API
│   ├── transformation/
│   │   └── build_vector_index.py  # Build vector index
│   ├── raw/                 # Staging area for raw files
│   └── processed/           # Staging area for processed files
├── mcp_server/
│   └── server.py            # MCP server — all tools live here
├── agent/
│   └── agent.py             # LLM agent that calls MCP tools
├── docker-compose.yml
└── README.md
```
