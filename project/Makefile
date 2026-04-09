# ── NYC Airbnb Hospitality Lakehouse ──────────────────────────────────────────

# Run the full pipeline (production — 50k reviews, ~7 hrs total)
pipeline:
	python run_pipeline.py

# Run the pipeline in test mode (500 reviews, ~1 min total)
pipeline-test:
	python run_pipeline.py --test

# Launch the agent (production ChromaDB index)
agent:
	docker-compose run --rm mcp-agent

# Launch the agent against the test ChromaDB index
agent-test:
	docker-compose run --rm \
		-e CHROMA_PATH=/app/chromadb \
		-v $$(pwd)/data_pipeline/processed/chromadb_test:/app/chromadb \
		mcp-agent

# Start databases only
up:
	docker-compose up -d

# Stop containers (keep data volumes)
down:
	docker-compose down

.PHONY: pipeline pipeline-test agent agent-test up down
