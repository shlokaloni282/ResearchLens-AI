import fitz
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from backend.app.services.chunking import chunk_pages
from backend.app.services.retrieval import get_default_vector_index


router = APIRouter(
    prefix="/papers",
    tags=["Research Papers"],
)

MAX_FILE_SIZE = 20 * 1024 * 1024


@router.post("/upload")
async def upload_paper(
    file: UploadFile = File(...),
    strategy: str = Query(
        default="recursive",
        pattern="^(fixed|recursive|semantic)$",
    ),
    chunk_size: int = Query(default=500, ge=1, le=5000),
    overlap: int = Query(default=50, ge=0, le=4999),
):
    try:
        if overlap >= chunk_size:
            raise HTTPException(
                status_code=422,
                detail="overlap must be smaller than chunk_size.",
            )

        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported.",
            )

        content = await file.read(MAX_FILE_SIZE + 1)

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="PDF must not exceed 20 MB.",
            )

        if not content.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=400,
                detail="Invalid PDF file.",
            )

        try:
            document = fitz.open(stream=content, filetype="pdf")
        except (fitz.FileDataError, RuntimeError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Could not process this PDF.",
            ) from exc

        try:
            if document.is_encrypted:
                raise HTTPException(
                    status_code=400,
                    detail="Password-protected PDFs are not supported.",
                )

            pages = [
                {
                    "page_number": page_number,
                    "text": page.get_text("text"),
                }
                for page_number, page in enumerate(document, start=1)
            ]
            total_pages = len(document)
        finally:
            document.close()

        paper_id = str(uuid4())
        chunks = chunk_pages(
            pages=pages,
            paper_id=paper_id,
            source_filename=file.filename,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        # Index chunks into default in-memory vector store for retrieval & QA
        get_default_vector_index().add_chunks(chunks)

        return {
            "filename": file.filename,
            "total_pages": total_pages,
            "pages": pages,
            "chunks": chunks,
            "paper_id": paper_id,
            "chunking_method": strategy,
            "chunk_size": chunk_size,
            "overlap": overlap,
        }
    finally:
        await file.close()
