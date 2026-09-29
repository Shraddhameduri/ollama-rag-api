"""Unit tests for the recursive character splitter."""

from __future__ import annotations

import pytest

from app.rag.chunking import split_text


def test_empty_and_blank_input():
    assert split_text("") == []
    assert split_text("   \n  ") == []


def test_short_text_single_chunk():
    assert split_text("hello world", chunk_size=800) == ["hello world"]


def test_chunk_size_bound():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)  # hard bound


def test_overlap_carries_text_forward():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = split_text(text, chunk_size=100, chunk_overlap=50)
    assert len(chunks) > 1
    # The next chunk must start with the tail of the previous chunk.
    assert chunks[1].startswith(chunks[0][-50:])


def test_zero_overlap():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = split_text(text, chunk_size=100, chunk_overlap=0)
    assert len(chunks) > 1
    assert not chunks[1].startswith(chunks[0][-10:])


def test_paragraph_boundaries_preferred():
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    chunks = split_text(text, chunk_size=30, chunk_overlap=0)
    assert len(chunks) == 3
    assert "First paragraph" in chunks[0]
    assert "Third paragraph" in chunks[2]


def test_long_word_hard_split():
    text = "a" * 250
    chunks = split_text(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(len(c) <= 110 for c in chunks)
    assert all(set(c) <= {"a", " "} for c in chunks)  # " " is the piece joiner
    assert chunks[0].startswith("a" * 50)


def test_invalid_params():
    with pytest.raises(ValueError):
        split_text("x", chunk_size=0)
    with pytest.raises(ValueError):
        split_text("x", chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        split_text("x", chunk_size=100, chunk_overlap=-1)


def test_no_empty_chunks():
    text = "\n\n\nLots\n\n\nof\n\n\nblank\n\n\nlines\n\n\n"
    chunks = split_text(text, chunk_size=800)
    assert chunks and all(c.strip() for c in chunks)
