# Repository Guidelines

## Project Structure & Module Organization
- Root: data artifacts (`dfg_relationships.csv`, reports) and the `pmchatbot/` package.
- `pmchatbot/pyproject.toml`: packaging, deps, dev tools.
- `pmchatbot/src/`: main source code (modular by domain):
  - `data/`, `chunking/`, `embeddings/`, `retrieval/`, `database/`, `llm/`, `utils/`, `config/`.
  - `streamlit_integration/`: UI, state, handlers, assets.
  - Entrypoints: `src/main.py` (CLI loop) and `src/streamlit_integration/streamlit_app.py` (UI).
- Tests (current): `pmchatbot/src/test_files/` (pytest-style `test_*.py`).

## Build, Test, and Development Commands
- Create env (PowerShell): `python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1`
- Install (dev): `cd pmchatbot && pip install -e .[dev]`
- Lint/format: `black pmchatbot/src && flake8 pmchatbot/src`
- Run tests: `pytest pmchatbot/src/test_files -q`
- Run CLI: `python pmchatbot/src/main.py`
- Run UI: `streamlit run pmchatbot/src/streamlit_integration/streamlit_app.py`

## Coding Style & Naming Conventions
- Python: 4-space indent, max line length 100 (Black configured).
- Naming: modules/functions `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE`.
- Imports: prefer absolute (`pmchatbot.src...`). Keep public APIs in package `__init__.py` minimal.
- Formatting/linting: run Black and Flake8 before pushing.

## Testing Guidelines
- Frameworks: `pytest`, `pytest-asyncio` (for async pieces).
- Location: keep or migrate to `pmchatbot/tests` over time; for now use `pmchatbot/src/test_files`.
- Naming: files `test_*.py`, tests `test_*`.
- Scope: unit-test data loaders, chunkers, retrievers, and config parsing. Use small CSV fixtures.
- Command: `pytest -q` (or target the directory above).

## Commit & Pull Request Guidelines
- Commits: follow Conventional Commits (e.g., `feat: add variant chunker`, `fix: guard Neo4j connect`).
- PRs: clear description, link issues, include before/after screenshots for Streamlit changes, and note config/env expectations. Update docs when behavior changes.

## Security & Configuration Tips
- Config: `.env` is loaded from `pmchatbot/.env` (see `src/config/settings.py`). Example keys: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `CSV_FILE_PATH`, `LLM_TYPE`, `OPENAI_API_KEY`, `OLLAMA_BASE_URL`.
- Secrets: never commit `.env` or credentials. Use placeholders in docs and PRs.
- GPU/Models: embeddings/reranker can use CPU/GPU; ensure local model paths/versions align with `.env`.
