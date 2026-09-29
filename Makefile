.PHONY: install run test lint docker-up docker-down eval pull-models

PY := .venv/bin/python

install:  ## Create venv and install dependencies
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt

run:  ## Run the API locally with reload
	$(PY) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:  ## Run the test suite
	$(PY) -m pytest -q

lint:  ## Lint with ruff
	.venv/bin/ruff check .

docker-up:  ## Build and start ollama + api
	docker compose up --build -d

docker-down:  ## Stop containers
	docker compose down

eval:  ## Run the RAG smoke eval (needs Ollama running)
	$(PY) scripts/eval_rag.py

pull-models:  ## Pull Ollama chat + embedding models
	bash scripts/pull_models.sh
