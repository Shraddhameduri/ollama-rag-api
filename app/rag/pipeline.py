"""RAG ingest + retrieve pipeline tying chunking, embeddings and the store."""

from __future__ import annotations

import logging
import time
import uuid

from .chunking import split_text
from .store import DocumentStore

log = logging.getLogger(__name__)

# Cosine similarity below this is treated as "nothing useful retrieved".
MIN_SIMILARITY = 0.15
EXCERPT_LEN = 220
_EMBED_BATCH = 32


class RagPipeline:
    """High-level RAG operations used by the API routes."""

    def __init__(
        self,
        store: DocumentStore,
        ollama_client,  # app.ollama_client.OllamaClient (duck-typed for tests)
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        top_k: int = 4,
    ) -> None:
        self.store = store
        self.ollama = ollama_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

    def ingest_text(
        self, text: str, filename: str, doc_id: str | None = None
    ) -> dict:
        """Chunk, embed and store a document. Returns the document record."""
        chunks = split_text(text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            raise ValueError("No text to ingest")
        embeddings: list[list[float]] = []
        for i in range(0, len(chunks), _EMBED_BATCH):
            embeddings.extend(self.ollama.embed(chunks[i : i + _EMBED_BATCH]))
        doc_id = doc_id or uuid.uuid4().hex[:12]
        self.store.add_chunks(doc_id, filename, chunks, embeddings)
        log.info(
            "ingested document",
            extra={"doc_id": doc_id, "chunks": len(chunks), "chars": len(text)},
        )
        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks": len(chunks),
            "chars": len(text),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return ranked, excerpted chunks relevant to ``query``.

        Each result has ``rank, doc_id, filename, chunk_index, score,
        excerpt, text``. Low-similarity hits (< ``MIN_SIMILARITY``) are
        dropped as "nothing useful".
        """
        if not query or not query.strip():
            return []
        k = self.top_k if top_k is None else top_k
        if k <= 0:
            return []
        (query_vec,) = self.ollama.embed([query])
        hits = self.store.query(query_vec, top_k=k)
        results: list[dict] = []
        for hit in hits:
            if hit["score"] < MIN_SIMILARITY:
                continue
            text = hit["text"]
            excerpt = text[:EXCERPT_LEN] + ("…" if len(text) > EXCERPT_LEN else "")
            results.append(
                {
                    "doc_id": hit["doc_id"],
                    "filename": hit["filename"],
                    "chunk_index": hit["chunk_index"],
                    "score": round(hit["score"], 4),
                    "excerpt": excerpt,
                    "text": text,
                }
            )
        for rank, result in enumerate(results, start=1):
            result["rank"] = rank
        return results
