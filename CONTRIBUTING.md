# Contributing to ollama-rag-api

Thanks for contributing! A few ground rules keep the project healthy:

## Getting started

```bash
make install   # venv + deps
make test      # full suite (Ollama is mocked; no server needed)
make lint      # ruff must be clean
```

## Pull requests

- Keep changes focused; one concern per PR.
- Add or update tests for behaviour changes (`tests/`).
- Keep `ruff check .` and `pytest -q` green — CI enforces both plus a Docker build.
- Update `README.md` when you change the API, config, or deployment story.
- Never commit secrets: `.env` is gitignored; use `.env.example` for new keys.

## Code style

- Python 3.11+, type hints on public functions, module docstrings.
- Structured JSON logging; never log API keys, request bodies, or document contents.
- Errors: uniform `{"error": {"code", "message"}}` envelope via the handlers in
  `app/main.py` — don't invent new shapes.

## Reporting issues

Include: what you ran, the full JSON error body, relevant log lines
(`request_id` helps), and your config (redact `API_KEY`).
