import anyio
from io import BytesIO
from unittest.mock import MagicMock

import fitz
import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

import backend.app.routes.papers as papers_route
from backend.app.routes.papers import upload_paper


def _pdf_bytes() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Research paper text.")
    content = document.tobytes()
    document.close()
    return content


def _upload(content: bytes, filename: str = "paper.pdf", **params):
    file = UploadFile(filename=filename, file=BytesIO(content))
    return anyio.run(
        upload_paper,
        file,
        params.get("strategy", "recursive"),
        params.get("chunk_size", 500),
        params.get("overlap", 50),
    )


def test_upload_pdf_returns_pages_and_chunks(monkeypatch):
    """
    Happy-path upload test.

    Patches get_default_vector_index() so the test does NOT load the
    sentence-transformer model (which takes ~2 min cold in CI/sandbox).
    The patch verifies that add_chunks() is called with the produced chunks.
    """
    mock_index = MagicMock()
    monkeypatch.setattr(papers_route, "get_default_vector_index", lambda: mock_index)

    result = _upload(_pdf_bytes())

    assert result["filename"] == "paper.pdf"
    assert result["total_pages"] == 1
    assert result["pages"][0]["text"].strip() == "Research paper text."
    assert result["chunks"]
    assert len(result["chunks"]) == len(set(
        chunk["chunk_id"] for chunk in result["chunks"]
    ))
    # Confirm chunks were submitted to the vector index
    mock_index.add_chunks.assert_called_once_with(result["chunks"])


def test_upload_rejects_files_over_size_limit():
    with pytest.raises(HTTPException) as error:
        _upload(b"%PDF-" + b"x" * (20 * 1024 * 1024))

    assert error.value.status_code == 413
    assert error.value.detail == "PDF must not exceed 20 MB."


def test_upload_rejects_malformed_pdf():
    with pytest.raises(HTTPException) as error:
        _upload(b"%PDF-not a valid document")

    assert error.value.status_code == 400
    assert error.value.detail == "Could not process this PDF."


def test_upload_rejects_overlap_not_smaller_than_chunk_size():
    with pytest.raises(HTTPException) as error:
        _upload(_pdf_bytes(), chunk_size=10, overlap=10)

    assert error.value.status_code == 422
    assert error.value.detail == "overlap must be smaller than chunk_size."
