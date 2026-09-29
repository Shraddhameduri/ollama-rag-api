"""Document management: upload (.pdf/.txt/.md), list, delete."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pypdf import PdfReader

from .. import metrics
from ..limits import rate_limit

router = APIRouter(tags=["documents"])

ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}


def _extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from an uploaded file by extension."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="replace")


@router.post("/documents")
@rate_limit()
async def upload_document(request: Request, file: UploadFile = File(...)):  # noqa: B008
    settings = request.app.state.settings
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: .pdf, .txt, .md",
        )
    data = await file.read()
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.MAX_UPLOAD_MB} MB upload limit.",
        )
    try:
        text = _extract_text(filename, data)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail="Could not extract text from file."
        ) from exc
    if not text.strip():
        raise HTTPException(
            status_code=400, detail="No extractable text found in file."
        )

    doc = request.app.state.pipeline.ingest_text(text, filename)
    metrics.documents_ingested_total.inc()
    return {
        "doc_id": doc["doc_id"],
        "filename": doc["filename"],
        "chunks": doc["chunks"],
        "chars": doc["chars"],
    }


@router.get("/documents")
def list_documents(request: Request) -> dict:
    docs = request.app.state.pipeline.store.list_docs()
    return {"documents": docs}


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str, request: Request) -> dict:
    deleted = request.app.state.pipeline.store.delete_doc(doc_id)
    if not deleted:
        raise HTTPException(
            status_code=404, detail=f"Document '{doc_id}' not found."
        )
    return {"deleted": doc_id}
