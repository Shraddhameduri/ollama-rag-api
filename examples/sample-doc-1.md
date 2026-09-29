# Sample document 1: project overview (used by scripts/eval_rag.py)

# ollama-rag-api

**ollama-rag-api** is a production-grade Chat API with Retrieval-Augmented
Generation (RAG), powered entirely by free, locally-run Ollama models. It
exposes an OpenAI-compatible `POST /v1/chat/completions` endpoint (including
SSE streaming), a document ingestion API for PDF/TXT/Markdown files, and
operations endpoints for health, readiness, and Prometheus metrics.

## Key features

- OpenAI-compatible chat completions with `sources` (citations) in every response.
- RAG pipeline: recursive character chunking, Ollama embeddings, ChromaDB vector store.
- Per-session conversation memory with a configurable history window.
- API-key authentication (constant-time compare), per-key/IP rate limiting.
- Structured JSON logging with request IDs, Prometheus metrics, readiness probes.
- Docker + docker-compose for one-command deployment alongside Ollama.

## Architecture

Requests flow through FastAPI middleware (request ID, access logging, CORS),
then to the chat route. The last user message is embedded with the configured
Ollama embedding model and used to query ChromaDB for the top-K chunks. Those
chunks are injected into the system prompt as numbered context with citation
instructions, recent session history is prepended, and the Ollama chat model
generates the answer.
