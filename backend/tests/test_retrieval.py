import pytest
import numpy as np
from backend.app.services.retrieval import InMemoryVectorIndex


class AccuracyComparisonEmbeddingModel:
    """Small deterministic embedding model for cross-paper ranking tests."""

    def encode(self, texts, normalize_embeddings=True):
        vectors = []
        for text in texts:
            text = text.lower()
            vectors.append([
                1.0 if "accuracy" in text or "reported" in text or "study" in text else 0.0,
                1.0 if "machine learning" in text else 0.0,
                1.0 if "dataset" in text else 0.0,
                1.0 if "biology" in text or "plants" in text else 0.0,
            ])
        embeddings = np.asarray(vectors, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = np.divide(
                embeddings,
                norms,
                out=np.zeros_like(embeddings),
                where=norms != 0,
            )
        return embeddings


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "c1",
            "paper_id": "paper_100mhz",
            "source_filename": "rng_paper.pdf",
            "page_number": 3,
            "chunking_method": "semantic",
            "text": "The power dissipation is 23.4 mW at maximal clock rate = 100 MHz. Entire chip area is 116x84 um2.",
        },
        {
            "chunk_id": "c2",
            "paper_id": "paper_100mhz",
            "source_filename": "rng_paper.pdf",
            "page_number": 4,
            "chunking_method": "semantic",
            "text": "Monobit Test: Count the number of ones in the 20,000 bit stream. Passed if 9,654 < X < 10,346.",
        },
        {
            "chunk_id": "c3",
            "paper_id": "paper_ai_health",
            "source_filename": "ai_health.pdf",
            "page_number": 1,
            "chunking_method": "recursive",
            "text": "Deep learning models improve disease detection in medical MRI scans significantly.",
        },
    ]


def test_empty_index_returns_empty():
    index = InMemoryVectorIndex()
    assert len(index) == 0
    assert index.search("anything") == []
    assert index.get_paper_ids() == []


def test_add_chunks_stores_metadata(sample_chunks):
    index = InMemoryVectorIndex()
    added = index.add_chunks(sample_chunks)

    assert added == 3
    assert len(index) == 3
    assert "paper_100mhz" in index.get_paper_ids()
    assert "paper_ai_health" in index.get_paper_ids()
    assert "rng_paper.pdf" in index.get_source_filenames()


def test_search_ranks_relevant_chunk_first(sample_chunks):
    index = InMemoryVectorIndex()
    index.add_chunks(sample_chunks)

    results = index.search("What is the power dissipation and clock rate?", top_k=2)

    assert len(results) == 2
    top_hit = results[0]
    assert top_hit["chunk_id"] == "c1"
    assert top_hit["page_number"] == 3
    assert "23.4 mW" in top_hit["text"]
    assert top_hit["score"] > 0.0


def test_paper_filtering_by_paper_id(sample_chunks):
    index = InMemoryVectorIndex()
    index.add_chunks(sample_chunks)

    # Search specifically for medical MRI topic, but filter to paper_100mhz
    results = index.search(
        query="disease detection in medical scans",
        top_k=5,
        paper_id="paper_100mhz",
    )

    # None of the results should be from paper_ai_health
    assert all(r["paper_id"] == "paper_100mhz" for r in results)


def test_paper_filtering_by_source_filename(sample_chunks):
    index = InMemoryVectorIndex()
    index.add_chunks(sample_chunks)

    results = index.search(
        query="20,000 bit stream",
        top_k=5,
        source_filename="rng_paper.pdf",
    )

    assert len(results) > 0
    assert all(r["source_filename"] == "rng_paper.pdf" for r in results)


def test_paper_filtering_by_selected_paper_ids(sample_chunks):
    index = InMemoryVectorIndex()
    index.add_chunks(sample_chunks)

    results = index.search(
        query="medical MRI disease detection",
        top_k=5,
        paper_ids=["paper_ai_health"],
    )

    assert len(results) == 1
    assert all(result["paper_id"] == "paper_ai_health" for result in results)


def test_paper_filtering_excludes_unselected_papers(sample_chunks):
    index = InMemoryVectorIndex()
    index.add_chunks(sample_chunks)

    results = index.search(
        query="research study",
        top_k=5,
        paper_ids=["paper_100mhz"],
    )

    assert results
    assert all(result["paper_id"] == "paper_100mhz" for result in results)


def test_empty_paper_ids_returns_no_results(sample_chunks):
    index = InMemoryVectorIndex()
    index.add_chunks(sample_chunks)

    assert index.search("power dissipation", top_k=5, paper_ids=[]) == []


def test_paper_ids_combines_with_source_filename_filter(sample_chunks):
    index = InMemoryVectorIndex()
    index.add_chunks(sample_chunks)

    results = index.search(
        query="Monobit test",
        top_k=5,
        paper_ids=["paper_100mhz", "paper_ai_health"],
        source_filename="rng_paper.pdf",
    )

    assert results
    assert all(
        result["paper_id"] == "paper_100mhz"
        and result["source_filename"] == "rng_paper.pdf"
        for result in results
    )


def test_paper_ids_combines_with_single_paper_id_filter(sample_chunks):
    index = InMemoryVectorIndex()
    index.add_chunks(sample_chunks)

    selected_and_single = index.search(
        query="power dissipation",
        top_k=5,
        paper_id="paper_100mhz",
        paper_ids=["paper_100mhz", "paper_ai_health"],
    )
    excluded_by_single = index.search(
        query="power dissipation",
        top_k=5,
        paper_id="paper_100mhz",
        paper_ids=["paper_ai_health"],
    )

    assert selected_and_single
    assert all(result["paper_id"] == "paper_100mhz" for result in selected_and_single)
    assert excluded_by_single == []


def test_cross_paper_search(sample_chunks):
    index = InMemoryVectorIndex()
    index.add_chunks(sample_chunks)

    # Without filter, can retrieve from either paper
    medical_results = index.search("medical MRI disease detection", top_k=1)
    assert medical_results[0]["paper_id"] == "paper_ai_health"

    hardware_results = index.search("Monobit test 20,000 bits", top_k=1)
    assert hardware_results[0]["paper_id"] == "paper_100mhz"


def test_cross_paper_search_returns_relevant_chunks_from_multiple_papers():
    chunks = [
        {
            "chunk_id": "a1",
            "paper_id": "paper_a",
            "source_filename": "study_a.pdf",
            "page_number": 2,
            "text": "Study A reported accuracy using machine learning.",
        },
        {
            "chunk_id": "b1",
            "paper_id": "paper_b",
            "source_filename": "study_b.pdf",
            "page_number": 5,
            "text": "Study B reported accuracy using its dataset.",
        },
        {
            "chunk_id": "c1",
            "paper_id": "paper_c",
            "source_filename": "study_c.pdf",
            "page_number": 1,
            "text": "Plant biology describes leaf structures.",
        },
    ]
    index = InMemoryVectorIndex(model=AccuracyComparisonEmbeddingModel())
    index.add_chunks(chunks)

    results = index.search("Compare reported accuracy between the studies", top_k=2)

    assert {result["paper_id"] for result in results} == {"paper_a", "paper_b"}
    assert results[0]["score"] == results[1]["score"]
    assert results == sorted(results, key=lambda result: result["score"], reverse=True)


def test_cross_paper_search_preserves_similarity_ranking_without_forced_paper_diversity():
    chunks = [
        {
            "chunk_id": "a1",
            "paper_id": "paper_a",
            "source_filename": "study_a.pdf",
            "page_number": 2,
            "text": "Study A reported accuracy using machine learning.",
        },
        {
            "chunk_id": "a2",
            "paper_id": "paper_a",
            "source_filename": "study_a.pdf",
            "page_number": 3,
            "text": "Study A accuracy was reported for a machine learning model.",
        },
        {
            "chunk_id": "b1",
            "paper_id": "paper_b",
            "source_filename": "study_b.pdf",
            "page_number": 5,
            "text": "Study B reported accuracy using its dataset.",
        },
    ]
    index = InMemoryVectorIndex(model=AccuracyComparisonEmbeddingModel())
    index.add_chunks(chunks)

    results = index.search("Study machine learning accuracy", top_k=2)

    assert [result["paper_id"] for result in results] == ["paper_a", "paper_a"]
    assert results[0]["score"] >= results[1]["score"]
    assert all(result["paper_id"] == "paper_a" for result in results)


def test_top_k_limit(sample_chunks):
    index = InMemoryVectorIndex()
    index.add_chunks(sample_chunks)

    results = index.search("RNG or MRI test", top_k=2)
    assert len(results) == 2


def test_clear_index(sample_chunks):
    index = InMemoryVectorIndex()
    index.add_chunks(sample_chunks)
    assert len(index) == 3

    index.clear()
    assert len(index) == 0
    assert index.search("power dissipation") == []
