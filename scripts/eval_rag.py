#!/usr/bin/env python3
"""Smoke eval for ollama-rag-api.

Ingests ``examples/`` documents, asks three questions through the RAG
pipeline + Ollama chat, and asserts each answer carries citations ([1] style)
and non-empty sources.

Requires a running Ollama server with the configured models pulled
(see scripts/pull_models.sh). Exits non-zero on any failure.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.ollama_client import OllamaClient  # noqa: E402
from app.rag.pipeline import RagPipeline  # noqa: E402
from app.rag.prompts import build_rag_prompt  # noqa: E402
from app.rag.store import DocumentStore  # noqa: E402

QUESTIONS = [
    "What is ollama-rag-api and what does it do?",
    "How do I run ollama-rag-api with Docker?",
    "Which vector store does ollama-rag-api use and where is data persisted?",
]

CITATION_RE = re.compile(r"\[\d+\]")


def main() -> int:
    settings = Settings()
    ollama = OllamaClient(
        host=settings.OLLAMA_HOST,
        chat_model=settings.OLLAMA_MODEL,
        embed_model=settings.OLLAMA_EMBED_MODEL,
    )
    if not ollama.ping():
        print(f"FAIL: Ollama not reachable at {settings.OLLAMA_HOST}")
        return 1

    store = DocumentStore(persist_dir=str(Path("./data/eval-chroma").resolve()))
    pipeline = RagPipeline(
        store=store,
        ollama_client=ollama,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        top_k=settings.TOP_K,
    )

    for doc in sorted((ROOT / "examples").glob("*.md")):
        record = pipeline.ingest_text(doc.read_text(encoding="utf-8"), doc.name)
        print(f"ingested {doc.name}: {record['chunks']} chunks")

    failures = 0
    for question in QUESTIONS:
        chunks = pipeline.retrieve(question, top_k=settings.TOP_K)
        prompt = build_rag_prompt(chunks)
        result = ollama.chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
        )
        answer = result["content"]
        has_citation = bool(CITATION_RE.search(answer))
        has_sources = bool(chunks)
        status = "PASS" if (has_citation and has_sources) else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"[{status}] Q: {question}")
        print(f"        sources={len(chunks)} citations={'yes' if has_citation else 'no'}")
        print(f"        A: {answer[:200]}...")

    print(f"\n{len(QUESTIONS) - failures}/{len(QUESTIONS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
