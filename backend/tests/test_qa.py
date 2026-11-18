"""
Unit and Integration Tests for Stage 3 Question Answering & Grounded Generation.
"""

import asyncio
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.qa import QuestionRequest
from backend.app.services.qa_service import (
    ABSTENTION_MESSAGE,
    GeminiProvider,
    LLMError,
    MockLLMProvider,
    answer_research_question,
    build_rag_prompt,
    extract_and_verify_citations,
    is_abstention,
)
from backend.app.services.retrieval import InMemoryVectorIndex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class MockVectorIndex:
    """Fast, lightweight mock vector index for QA service unit tests."""
    def __init__(self, chunks=None):
        self._chunks = chunks or []

    def search(self, query: str, top_k: int = 5, paper_id=None, source_filename=None):
        if not self._chunks:
            return []
        results = []
        for c in self._chunks:
            if paper_id is not None and c.get("paper_id") != paper_id:
                continue
            if source_filename is not None and c.get("source_filename") != source_filename:
                continue
            item = dict(c)
            item["score"] = 0.95
            results.append(item)
        return results[:top_k]


@pytest.fixture
def populated_index():
    chunks = [
        {
            "chunk_id": "chunk_p3",
            "paper_id": "paper_rng",
            "source_filename": "rng_paper.pdf",
            "page_number": 3,
            "chunking_method": "semantic",
            "text": "The power dissipation is 23.4 mW at maximal clock rate = 100 MHz. The core area is 37x37 um2.",
        },
        {
            "chunk_id": "chunk_p4",
            "paper_id": "paper_rng",
            "source_filename": "rng_paper.pdf",
            "page_number": 4,
            "chunking_method": "semantic",
            "text": "Monobit Test: The test is passed if 9,654 < X < 10,346 in a 20,000 bit stream.",
        },
    ]
    return MockVectorIndex(chunks)


# ---------------------------------------------------------------------------
# Unit Tests: Prompt Building & Citation Verification
# ---------------------------------------------------------------------------

def test_build_rag_prompt():
    chunks = [
        {"source_filename": "test.pdf", "page_number": 1, "text": "Sample text here."}
    ]
    prompt = build_rag_prompt("What is this?", chunks)
    assert "test.pdf" in prompt
    assert "Page: 1" in prompt
    assert "What is this?" in prompt
    assert "NEVER invent citations" in prompt


def test_extract_and_verify_valid_citations():
    retrieved_chunks = [
        {"source_filename": "rng_paper.pdf", "page_number": 3, "chunk_id": "c1", "score": 0.85},
        {"source_filename": "rng_paper.pdf", "page_number": 4, "chunk_id": "c2", "score": 0.72},
    ]
    answer = "The power is 23.4 mW [Paper: rng_paper.pdf, Page: 3], and the test is passed [rng_paper.pdf, Page 4]."
    citations = extract_and_verify_citations(answer, retrieved_chunks)

    assert len(citations) == 2
    pages = {c.page_number for c in citations}
    assert pages == {3, 4}
    assert all(c.source_filename == "rng_paper.pdf" for c in citations)


def test_hallucinated_citations_are_rejected():
    retrieved_chunks = [
        {"source_filename": "rng_paper.pdf", "page_number": 3, "chunk_id": "c1"},
    ]
    # The LLM hallucinates Page 99 which was never retrieved
    answer = "The battery lifetime is 10 hours [Paper: rng_paper.pdf, Page: 99]."
    citations = extract_and_verify_citations(answer, retrieved_chunks)

    # Page 99 must NOT be in the verified citations list!
    assert not any(c.page_number == 99 for c in citations)


def test_is_abstention_detector():
    assert is_abstention("The provided paper context does not contain sufficient information to answer this question.")
    assert is_abstention("This is not specified in the provided context.")
    assert is_abstention("I cannot answer from the provided context.")
    assert not is_abstention("The power dissipation is 23.4 mW at 100 MHz [Page 3].")


# ---------------------------------------------------------------------------
# Core Service Tests: answer_research_question
# ---------------------------------------------------------------------------

def test_answer_generation_grounded_success(populated_index):
    mock_llm = MockLLMProvider(
        response_generator="The power dissipation is 23.4 mW at 100 MHz [Paper: rng_paper.pdf, Page: 3]."
    )

    response = asyncio.run(
        answer_research_question(
            question="What is the power dissipation?",
            vector_index=populated_index,
            llm_provider=mock_llm,
        )
    )

    assert response.has_sufficient_evidence is True
    assert "23.4 mW" in response.answer
    assert len(response.citations) >= 1
    assert response.citations[0].page_number == 3
    assert response.citations[0].source_filename == "rng_paper.pdf"
    assert response.retrieved_chunks_count > 0


def test_missing_evidence_empty_index_abstains():
    empty_index = MockVectorIndex([])
    # LLM should not even be called
    mock_llm = MockLLMProvider(response_generator="I should not be called")

    response = asyncio.run(
        answer_research_question(
            question="What is the power dissipation?",
            vector_index=empty_index,
            llm_provider=mock_llm,
        )
    )

    assert response.has_sufficient_evidence is False
    assert response.answer == ABSTENTION_MESSAGE
    assert response.citations == []
    assert response.retrieved_chunks_count == 0


def test_unanswerable_question_llm_abstains(populated_index):
    # LLM determines the question cannot be answered from the retrieved context
    mock_llm = MockLLMProvider(
        response_generator="The provided paper context does not contain sufficient information to answer this question."
    )

    response = asyncio.run(
        answer_research_question(
            question="What is the measured battery lifetime in hours?",
            vector_index=populated_index,
            llm_provider=mock_llm,
        )
    )

    assert response.has_sufficient_evidence is False
    assert ABSTENTION_MESSAGE in response.answer
    assert response.citations == []


# ---------------------------------------------------------------------------
# Gemini Provider Tests
# ---------------------------------------------------------------------------

def test_gemini_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Gemini API key not found"):
        GeminiProvider(api_key=None)


def test_gemini_provider_mocked_network(monkeypatch):
    # Mock httpx.Client to simulate successful Gemini REST API response
    class MockResponse:
        status_code = 200
        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Grounded answer from Gemini."}]
                        }
                    }
                ]
            }

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, **kwargs):
            return MockResponse()

    monkeypatch.setattr("httpx.Client", MockClient)

    provider = GeminiProvider(api_key="fake-test-key")
    result = provider.generate("Test prompt")
    assert result == "Grounded answer from Gemini."


def test_gemini_provider_error_handling(monkeypatch):
    class ErrorResponse:
        status_code = 403
        text = "API key not valid."
        headers = {}

    class MockClient:
        calls = 0

        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, **kwargs):
            type(self).calls += 1
            return ErrorResponse()

    monkeypatch.setattr("httpx.Client", MockClient)

    provider = GeminiProvider(api_key="bad-key")
    with pytest.raises(LLMError, match="Gemini API error"):
        provider.generate("Test prompt")
    assert MockClient.calls == 1


def test_gemini_provider_retries_503_then_succeeds(monkeypatch):
    class MockResponse:
        def __init__(self, status_code, headers=None):
            self.status_code = status_code
            self.headers = headers or {}
            self.text = "temporarily unavailable"

        def json(self):
            return {
                "candidates": [
                    {"content": {"parts": [{"text": "Recovered answer."}]}}
                ]
            }

    class MockClient:
        calls = 0

        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, **kwargs):
            type(self).calls += 1
            if type(self).calls == 1:
                return MockResponse(503, {"Retry-After": "120"})
            return MockResponse(200)

    delays = []
    monkeypatch.setattr("httpx.Client", MockClient)
    monkeypatch.setattr(
        "backend.app.services.qa_service.time.sleep", delays.append
    )

    provider = GeminiProvider(api_key="fake-test-key")
    assert provider.generate("Test prompt") == "Recovered answer."
    assert MockClient.calls == 2
    assert delays == [10.0]


def test_gemini_provider_retry_exhaustion(monkeypatch):
    class ErrorResponse:
        status_code = 503
        text = "temporarily unavailable"
        headers = {}

    class MockClient:
        calls = 0

        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, **kwargs):
            type(self).calls += 1
            return ErrorResponse()

    delays = []
    monkeypatch.setattr("httpx.Client", MockClient)
    monkeypatch.setattr(
        "backend.app.services.qa_service.time.sleep", delays.append
    )

    provider = GeminiProvider(api_key="fake-test-key")
    with pytest.raises(LLMError, match=r"Gemini API error \(503\) after 3 attempts"):
        provider.generate("Test prompt")
    assert MockClient.calls == 3
    assert len(delays) == 2
    assert 0.0 <= delays[0] <= 0.5
    assert 0.0 <= delays[1] <= 1.0


def test_gemini_provider_async_retries_503_then_succeeds(monkeypatch):
    class MockResponse:
        def __init__(self, status_code):
            self.status_code = status_code
            self.headers = {}
            self.text = "temporarily unavailable"

        def json(self):
            return {
                "candidates": [
                    {"content": {"parts": [{"text": "Recovered async answer."}]}}
                ]
            }

    class MockClient:
        calls = 0

        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def post(self, url, **kwargs):
            type(self).calls += 1
            if type(self).calls == 1:
                return MockResponse(503)
            return MockResponse(200)

    delays = []

    async def record_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("httpx.AsyncClient", MockClient)
    monkeypatch.setattr(
        "backend.app.services.qa_service.asyncio.sleep", record_sleep
    )

    provider = GeminiProvider(api_key="fake-test-key")
    assert asyncio.run(provider.generate_async("Test prompt")) == "Recovered async answer."
    assert MockClient.calls == 2
    assert len(delays) == 1
    assert 0.0 <= delays[0] <= 0.5


def test_gemini_provider_async_retry_exhaustion(monkeypatch):
    class ErrorResponse:
        status_code = 503
        text = "temporarily unavailable"
        headers = {}

    class MockClient:
        calls = 0

        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def post(self, url, **kwargs):
            type(self).calls += 1
            return ErrorResponse()

    delays = []

    async def record_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("httpx.AsyncClient", MockClient)
    monkeypatch.setattr(
        "backend.app.services.qa_service.asyncio.sleep", record_sleep
    )

    provider = GeminiProvider(api_key="fake-test-key")
    with pytest.raises(LLMError, match=r"Gemini API error \(503\) after 3 attempts"):
        asyncio.run(provider.generate_async("Test prompt"))
    assert MockClient.calls == 3
    assert len(delays) == 2


# ---------------------------------------------------------------------------
# API Route Tests (FastAPI TestClient)
# ---------------------------------------------------------------------------

def test_api_qa_ask_success(populated_index, monkeypatch):
    # Dependency injection: patch get_default_vector_index & get_default_llm_provider
    mock_llm = MockLLMProvider(
        response_generator="The power dissipation is 23.4 mW [Paper: rng_paper.pdf, Page: 3]."
    )
    monkeypatch.setattr(
        "backend.app.services.qa_service.get_default_vector_index",
        lambda: populated_index
    )
    monkeypatch.setattr(
        "backend.app.services.qa_service.get_default_llm_provider",
        lambda: mock_llm
    )

    client = TestClient(app)
    response = client.post(
        "/qa/ask",
        json={"question": "What is the power dissipation?", "top_k": 3}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["has_sufficient_evidence"] is True
    assert "23.4 mW" in data["answer"]
    assert len(data["citations"]) >= 1
    assert data["citations"][0]["page_number"] == 3


def test_api_qa_ask_validation_error():
    client = TestClient(app)
    # Question is too short (<3 chars)
    response = client.post("/qa/ask", json={"question": "a"})
    assert response.status_code == 422


def test_api_qa_ask_missing_api_key_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    client = TestClient(app)
    response = client.post(
        "/qa/ask",
        json={"question": "What is the power dissipation?"}
    )
    # When no key is configured, service returns 500 with configuration message
    assert response.status_code == 500
    assert "Gemini API key not found" in response.json()["detail"] or "No LLM API key configured" in response.json()["detail"]
