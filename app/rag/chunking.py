"""Recursive character text splitter (no langchain dependency).

Splits text on paragraph, then line, then word boundaries, falling back to
hard character splits. Pieces are merged greedily with ``" "`` into chunks of
at most ``chunk_size`` characters; the trailing ``chunk_overlap`` characters
of each chunk are carried into the next one.
"""

from __future__ import annotations

_DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]
_JOINER = " "


def _split_recursive(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Recursively split until every piece is at most ``chunk_size`` chars.

    Only pieces still exceeding ``chunk_size`` are split further, so
    paragraph/line structure is preserved whenever it already fits.
    """
    if len(text) <= chunk_size:
        return [text]
    for i, sep in enumerate(separators):
        if sep == "":
            return [text[j : j + chunk_size] for j in range(0, len(text), chunk_size)]
        if sep in text:
            pieces: list[str] = []
            rest = separators[i + 1 :] or [""]
            for part in text.split(sep):
                if not part:
                    continue
                if len(part) > chunk_size:
                    pieces.extend(_split_recursive(part, chunk_size, rest))
                else:
                    pieces.append(part)
            return pieces
    # No separator present and too long: hard split.
    return [text[j : j + chunk_size] for j in range(0, len(text), chunk_size)]


def split_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    separators: list[str] | None = None,
) -> list[str]:
    """Split ``text`` into overlapping chunks of at most ``chunk_size`` chars.

    Args:
        text: Input text (any length).
        chunk_size: Maximum characters per chunk (hard bound).
        chunk_overlap: Trailing characters carried from each chunk into the
            next. Best-effort: dropped when it would break the size bound.
        separators: Splitting hierarchy, tried in order.

    Returns:
        List of chunk strings. Empty/blank input returns ``[]``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
    if not text or not text.strip():
        return []

    separators = separators or list(_DEFAULT_SEPARATORS)
    # Every piece is guaranteed <= chunk_size by _split_recursive.
    pieces = [p.strip() for p in _split_recursive(text, chunk_size, separators)]
    pieces = [p for p in pieces if p]

    chunks: list[str] = []
    buf = ""

    def carry_overlap() -> str:
        return buf[-chunk_overlap:] if chunk_overlap > 0 else ""

    for piece in pieces:
        if not buf:
            buf = piece
            continue
        candidate = buf + _JOINER + piece
        if len(candidate) <= chunk_size:
            buf = candidate
            continue
        # Flush the full buffer, then start the next chunk with overlap.
        chunks.append(buf)
        tail = carry_overlap()
        if tail and len(tail) + len(_JOINER) + len(piece) <= chunk_size:
            buf = tail + _JOINER + piece
        else:
            buf = piece  # overlap dropped to respect the size bound

    if buf:
        chunks.append(buf)
    return chunks
