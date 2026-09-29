"""ChromaDB-backed vector store for document chunks."""

from __future__ import annotations

import logging
import time

import chromadb

log = logging.getLogger(__name__)


class DocumentStore:
    """Persistent vector store; one Chroma collection holds all chunks.

    Chunk ids are ``"<doc_id>:<chunk_index>"`` and every record carries
    ``doc_id`` / ``filename`` / ``chunk_index`` / ``created_at`` metadata so
    documents can be listed and deleted as a unit.
    """

    def __init__(self, persist_dir: str, collection_name: str = "documents") -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        doc_id: str,
        filename: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Persist chunk texts with their precomputed embeddings."""
        if not chunks:
            return
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._collection.add(
            ids=[f"{doc_id}:{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": i,
                    "created_at": created_at,
                }
                for i in range(len(chunks))
            ],
        )

    def query(self, embedding: list[float], top_k: int = 4) -> list[dict]:
        """Return the top-k most similar chunks as ranked hit dicts."""
        if self._collection.count() == 0:
            return []
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict] = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
        ):
            hits.append(
                {
                    "doc_id": meta["doc_id"],
                    "filename": meta["filename"],
                    "chunk_index": meta["chunk_index"],
                    "score": max(0.0, 1.0 - float(dist)),  # cosine distance -> similarity
                    "text": doc,
                }
            )
        return hits

    def delete_doc(self, doc_id: str) -> bool:
        """Delete every chunk of a document. Returns False when unknown."""
        existing = self._collection.get(where={"doc_id": doc_id})
        ids = existing["ids"]
        if not ids:
            return False
        self._collection.delete(ids=ids)
        return True

    def list_docs(self) -> list[dict]:
        """Aggregate chunk records into per-document summaries."""
        data = self._collection.get(include=["metadatas"])
        docs: dict[str, dict] = {}
        for meta in data["metadatas"]:
            entry = docs.setdefault(
                meta["doc_id"],
                {
                    "doc_id": meta["doc_id"],
                    "filename": meta["filename"],
                    "chunks": 0,
                    "created_at": meta.get("created_at", ""),
                },
            )
            entry["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["created_at"], reverse=True)

    def ping(self) -> bool:
        """True when the store answers (used by /ready)."""
        try:
            self._collection.count()
            return True
        except Exception:  # noqa: BLE001 — ping is best-effort by design
            return False
