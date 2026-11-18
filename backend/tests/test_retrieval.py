import pytest
from backend.app.services.retrieval import InMemoryVectorIndex


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


def test_cross_paper_search(sample_chunks):
    index = InMemoryVectorIndex()
    index.add_chunks(sample_chunks)

    # Without filter, can retrieve from either paper
    medical_results = index.search("medical MRI disease detection", top_k=1)
    assert medical_results[0]["paper_id"] == "paper_ai_health"

    hardware_results = index.search("Monobit test 20,000 bits", top_k=1)
    assert hardware_results[0]["paper_id"] == "paper_100mhz"


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
