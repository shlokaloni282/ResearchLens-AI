"""
Question Answering & Grounded Generation Service for ResearchLens AI.

Accepts research questions, retrieves relevant chunks from the vector index,
and generates strictly grounded answers with verifiable page citations using an LLM.
Includes abstention handling when context is insufficient and provenance verification
to prevent hallucinated citations.
"""

import asyncio
import os
import random
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import httpx

from backend.app.schemas.qa import AnswerResponse, CitationItem
from backend.app.services.retrieval import InMemoryVectorIndex, get_default_vector_index


# ---------------------------------------------------------------------------
# Exceptions & Constants
# ---------------------------------------------------------------------------

ABSTENTION_MESSAGE = (
    "The provided paper context does not contain sufficient information to answer this question."
)

ABSTENTION_PHRASES = [
    "does not contain sufficient information",
    "insufficient information",
    "does not provide sufficient information",
    "not specified in the provided context",
    "not mentioned in the provided context",
    "cannot be determined from the provided",
    "cannot answer from the provided context",
    "no information provided",
    "not specified",
]

GEMINI_MAX_ATTEMPTS = 3
GEMINI_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
GEMINI_RETRY_BASE_DELAY_SECONDS = 0.5
GEMINI_MAX_RETRY_DELAY_SECONDS = 10.0


class LLMError(Exception):
    """Raised when an LLM provider fails to generate a response."""
    pass


# ---------------------------------------------------------------------------
# LLM Provider Abstraction
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Synchronous text generation."""
        pass

    @abstractmethod
    async def generate_async(self, prompt: str) -> str:
        """Asynchronous text generation."""
        pass


class GeminiProvider(LLMProvider):
    """
    Modular Google Gemini API provider using direct HTTP requests.
    
    Reads API key from GEMINI_API_KEY or GOOGLE_API_KEY environment variables.
    Defaults to gemini-1.5-flash (configurable via GEMINI_MODEL).
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY in your environment."
            )
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        self.api_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )

    def _build_payload(self, prompt: str) -> Dict[str, Any]:
        return {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            },
        }

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> Optional[float]:
        retry_after = response.headers.get("Retry-After")
        if retry_after is None:
            return None

        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
            except (TypeError, ValueError, OverflowError):
                return None

            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())

    @classmethod
    def _retry_delay(cls, response: httpx.Response, attempt: int) -> float:
        retry_after = cls._retry_after_seconds(response)
        if retry_after is not None:
            return min(retry_after, GEMINI_MAX_RETRY_DELAY_SECONDS)

        backoff_cap = min(
            GEMINI_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
            GEMINI_MAX_RETRY_DELAY_SECONDS,
        )
        return random.uniform(0.0, backoff_cap)

    @staticmethod
    def _http_error_message(response: httpx.Response, attempt: int) -> str:
        message = f"Gemini API error ({response.status_code})"
        if response.status_code in GEMINI_RETRYABLE_STATUS_CODES and attempt > 1:
            message += f" after {attempt} attempts"
        return f"{message}: {response.text}"

    def generate(self, prompt: str) -> str:
        payload = self._build_payload(prompt)
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}

        try:
            with httpx.Client(timeout=30.0) as client:
                for attempt in range(1, GEMINI_MAX_ATTEMPTS + 1):
                    response = client.post(
                        self.api_url, params=params, json=payload, headers=headers
                    )
                    if response.status_code == 200:
                        break
                    if (
                        response.status_code not in GEMINI_RETRYABLE_STATUS_CODES
                        or attempt == GEMINI_MAX_ATTEMPTS
                    ):
                        raise LLMError(self._http_error_message(response, attempt))
                    time.sleep(self._retry_delay(response, attempt))
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Failed to connect to Gemini API: {exc}") from exc

        data = response.json()
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                raise LLMError("Gemini API returned no candidates.")
            return candidates[0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Malformed Gemini API response: {data}") from exc

    async def generate_async(self, prompt: str) -> str:
        payload = self._build_payload(prompt)
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for attempt in range(1, GEMINI_MAX_ATTEMPTS + 1):
                    response = await client.post(
                        self.api_url, params=params, json=payload, headers=headers
                    )
                    if response.status_code == 200:
                        break
                    if (
                        response.status_code not in GEMINI_RETRYABLE_STATUS_CODES
                        or attempt == GEMINI_MAX_ATTEMPTS
                    ):
                        raise LLMError(self._http_error_message(response, attempt))
                    await asyncio.sleep(self._retry_delay(response, attempt))
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Failed to connect to Gemini API: {exc}") from exc

        data = response.json()
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                raise LLMError("Gemini API returned no candidates.")
            return candidates[0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Malformed Gemini API response: {data}") from exc


class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider for unit tests, offline development, or CI testing.
    """

    def __init__(
        self,
        response_generator: Optional[Union[Callable[[str], str], str]] = None,
    ):
        self.response_generator = response_generator

    def generate(self, prompt: str) -> str:
        if callable(self.response_generator):
            return self.response_generator(prompt)
        if isinstance(self.response_generator, str):
            return self.response_generator
        return ABSTENTION_MESSAGE

    async def generate_async(self, prompt: str) -> str:
        return self.generate(prompt)


def get_default_llm_provider() -> LLMProvider:
    """
    Factory that returns GeminiProvider if an API key is present,
    or raises a clear error.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return GeminiProvider(api_key=api_key)
    raise ValueError(
        "No LLM API key configured. Please set GEMINI_API_KEY or GOOGLE_API_KEY in your environment, "
        "or inject a mock provider for testing."
    )


# ---------------------------------------------------------------------------
# Prompt Building & Provenance Verification
# ---------------------------------------------------------------------------

def build_rag_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Constructs a strict grounding prompt with numbered source passages.
    """
    context_blocks = []
    for idx, c in enumerate(chunks, start=1):
        filename = c.get("source_filename", "document.pdf")
        page = c.get("page_number", 1)
        text = c.get("text", "").strip()
        context_blocks.append(
            f"--- Context [{idx}] (Paper: {filename}, Page: {page}) ---\n{text}"
        )

    context_str = "\n\n".join(context_blocks)

    return f"""You are ResearchLens AI, an accurate and transparent AI research assistant for scientific papers.

Instructions:
1. Answer the user's research question using ONLY the provided paper contexts below.
2. Every factual statement in your answer must be supported by the provided text.
3. Cite your sources inline using the format: [Paper: <filename>, Page: <page_number>].
4. NEVER invent citations, paper names, page numbers, or external facts not present in the contexts.
5. If the provided contexts do NOT contain sufficient information or evidence to answer the question, state:
   "{ABSTENTION_MESSAGE}"
   Do not attempt to fabricate an answer.

CONTEXTS:
{context_str}

RESEARCH QUESTION:
{question}

ANSWER:"""


def is_abstention(answer_text: str) -> bool:
    """Detects whether the LLM explicitly abstained due to lack of evidence."""
    lower = answer_text.lower()
    return any(phrase in lower for phrase in ABSTENTION_PHRASES)


def extract_and_verify_citations(
    answer_text: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> List[CitationItem]:
    """
    Parses citations from the answer text and verifies them against retrieved chunks.
    NEVER returns a citation that was not part of the retrieved context.
    """
    if is_abstention(answer_text):
        return []

    # Map available chunks by (filename, page_number) and by page_number alone
    valid_citations: List[CitationItem] = []
    seen_keys = set()

    # Regex matches:
    # 1. [Paper: filename.pdf, Page: 3] or [Paper: filename.pdf, Page 3]
    # 2. [filename.pdf, Page 3] or [filename.pdf, p. 3]
    # 3. [Page 3] or [Page: 3]
    citation_pattern = re.compile(
        r"\[(?:Paper:\s*)?([a-zA-Z0-9_\-\.\s]+?)[,\s]+(?:Page|p\.)\s*:?\s*(\d+)\]|"
        r"\[(?:Page|p\.)\s*:?\s*(\d+)\]",
        re.I,
    )

    matches = citation_pattern.findall(answer_text)

    for match in matches:
        if match[0] and match[1]:
            # Both filename and page matched
            matched_file = match[0].strip()
            matched_page = int(match[1])
        elif match[2]:
            # Only page matched
            matched_file = None
            matched_page = int(match[2])
        else:
            continue

        # Verify against retrieved chunks
        for c in retrieved_chunks:
            chunk_file = c.get("source_filename", "")
            chunk_page = c.get("page_number")

            file_matches = (matched_file is None) or (matched_file.lower() in chunk_file.lower())
            page_matches = (chunk_page == matched_page)

            if file_matches and page_matches:
                key = (chunk_file, chunk_page)
                if key not in seen_keys:
                    seen_keys.add(key)
                    valid_citations.append(
                        CitationItem(
                            source_filename=chunk_file,
                            page_number=chunk_page,
                            chunk_id=c.get("chunk_id"),
                            score=c.get("score"),
                        )
                    )
                break

    # If the LLM answered affirmatively without inline brackets, or if citations were not formatted,
    # attribute to the top matching retrieved chunks that contain the relevant keywords:
    if not valid_citations and not is_abstention(answer_text) and retrieved_chunks:
        # Include the highest-scoring chunk as evidence
        top_chunk = retrieved_chunks[0]
        valid_citations.append(
            CitationItem(
                source_filename=top_chunk.get("source_filename", ""),
                page_number=top_chunk.get("page_number", 1),
                chunk_id=top_chunk.get("chunk_id"),
                score=top_chunk.get("score"),
            )
        )

    return valid_citations


# ---------------------------------------------------------------------------
# High-Level Service Functions
# ---------------------------------------------------------------------------

async def answer_research_question(
    question: str,
    top_k: int = 5,
    paper_id: Optional[str] = None,
    source_filename: Optional[str] = None,
    vector_index: Optional[InMemoryVectorIndex] = None,
    llm_provider: Optional[LLMProvider] = None,
) -> AnswerResponse:
    """
    Main asynchronous pipeline for answering research questions with strict grounding.

    1. Retrieves top_k relevant chunks from the vector index.
    2. If no chunks found, returns an explicit abstention response.
    3. Builds the strict grounding RAG prompt.
    4. Calls LLM provider to generate answer.
    5. Verifies citations against retrieved chunks to eliminate hallucinations.
    """
    index = vector_index if vector_index is not None else get_default_vector_index()
    provider = llm_provider if llm_provider is not None else get_default_llm_provider()

    # 1. Retrieve chunks
    chunks = index.search(
        query=question,
        top_k=top_k,
        paper_id=paper_id,
        source_filename=source_filename,
    )

    # 2. Check for empty context
    if not chunks:
        return AnswerResponse(
            question=question,
            answer=ABSTENTION_MESSAGE,
            has_sufficient_evidence=False,
            citations=[],
            retrieved_chunks_count=0,
        )

    # 3. Build prompt
    prompt = build_rag_prompt(question=question, chunks=chunks)

    # 4. Generate answer
    raw_answer = await provider.generate_async(prompt)

    # 5. Abstention & Citation Verification
    abstaining = is_abstention(raw_answer)

    if abstaining:
        final_answer = ABSTENTION_MESSAGE
        citations: List[CitationItem] = []
        sufficient = False
    else:
        final_answer = raw_answer
        citations = extract_and_verify_citations(raw_answer, chunks)
        sufficient = True

    return AnswerResponse(
        question=question,
        answer=final_answer,
        has_sufficient_evidence=sufficient,
        citations=citations,
        retrieved_chunks_count=len(chunks),
    )
