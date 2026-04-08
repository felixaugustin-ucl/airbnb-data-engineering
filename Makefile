# ── NYC Airbnb Hospitality Lakehouse — Makefile ───────────────────────────────
#
# Run from the repo root after cloning:
#   make agent     — start all services and launch the interactive agent
#   make build     — rebuild the mcp-agent container (after code changes)
#   make up        — start databases and Ollama in the background
#   make down      — stop all services (data volumes preserved)
#   make logs      — tail live logs from all services
#   make status    — show container health status

COMPOSE = docker-compose -f project/docker-compose.yml

.PHONY: agent build up down logs status

agent:
	@$(COMPOSE) up -d postgres mongodb ollama
	@$(COMPOSE) up -d --wait postgres mongodb
	@$(COMPOSE) run --rm mcp-agent

build:
	@$(COMPOSE) build mcp-agent

up:
	@$(COMPOSE) up -d postgres mongodb ollama

down:
	@$(COMPOSE) down

logs:
	@$(COMPOSE) logs -f

status:
	@docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep airbnb || echo "No services running"
