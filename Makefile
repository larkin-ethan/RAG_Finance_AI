# Docker Desktop's CLI (and docker-credential-desktop) live inside the app bundle.
# Use the docker on PATH if there is one, else the bundled one. PATH is also
# exported so docker can find its credential helper when pulling images.
DOCKER_BIN := /Applications/Docker.app/Contents/Resources/bin
DOCKER := $(shell command -v docker 2>/dev/null || echo $(DOCKER_BIN)/docker)
UV := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
export PATH := $(DOCKER_BIN):$(HOME)/.local/bin:$(PATH)

.PHONY: install test lint db db-down fetch parse chunk load api ui eval sweep

install:
	$(UV) sync --extra dev --extra ml

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check src tests ui

db:
	$(DOCKER) compose -f docker/docker-compose.yml up -d

db-down:
	$(DOCKER) compose -f docker/docker-compose.yml down

fetch:
	$(UV) run python -m finrag.ingest.fetch

parse:
	$(UV) run python -m finrag.ingest.parse

chunk:
	$(UV) run python -m finrag.chunking.run

load:
	$(UV) run python -m finrag.store.load --strategy section-512

api:
	$(UV) run uvicorn finrag.api.main:app --reload --port 8000

ui:
	$(UV) run streamlit run ui/Home.py

eval:
	$(UV) run python -m finrag.evals.run --name dev-run --split dev

sweep:
	$(UV) run python -m finrag.evals.experiments
