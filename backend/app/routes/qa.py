"""
FastAPI Routes for Research Paper Question Answering.
"""

from fastapi import APIRouter, HTTPException

from backend.app.schemas.qa import AnswerResponse, QuestionRequest
from backend.app.services.qa_service import LLMError, answer_research_question


router = APIRouter(
    prefix="/qa",
    tags=["Question Answering"],
)


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Asks a question grounded in uploaded research paper content.

    - Retrieves top-k relevant chunks matching the question.
    - Prompts the LLM with strict grounding constraints.
    - Returns verified citations (paper and page numbers).
    - Abstains if evidence is insufficient.
    """
    try:
        response = await answer_research_question(
            question=request.question,
            top_k=request.top_k,
            paper_id=request.paper_id,
            paper_ids=request.paper_ids,
            source_filename=request.source_filename,
        )
        return response
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {exc}")
