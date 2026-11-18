import pytest

from backend.app.services.chunking import (
    fixed_size_chunking,
    recursive_chunking,
    semantic_chunking,
    chunk_pages,
    split_sentences,
    strip_footer,
)


sample_text = """
Artificial intelligence is transforming healthcare.
Machine learning helps doctors analyze medical images.
Deep learning models improve disease detection.

Climate change affects global temperatures.
Rising temperatures contribute to glacier melting.
Carbon emissions accelerate global warming.
"""

FOOTER = (
    "Authorized licensed use limited to: Some College. "
    "Downloaded on October 26,2026 at 04:05:03 UTC from IEEE Xplore.  "
    "Restrictions apply."
)


# ---------------------------------------------------------------------------
# Original tests
# ---------------------------------------------------------------------------

def test_recursive_chunking():
    chunks = recursive_chunking(sample_text, chunk_size=100, overlap=20)

    assert chunks is not None
    assert len(chunks) > 0
    assert all(len(chunk) <= 100 for chunk in chunks)

    print("\nRecursive chunks:")
    for chunk in chunks:
        print(chunk)
        print("---")


def test_fixed_chunking():
    chunks = fixed_size_chunking(sample_text, chunk_size=100, overlap=20)

    assert chunks is not None
    assert len(chunks) > 0
    assert all(len(chunk) <= 100 for chunk in chunks)

    print("\nFixed-size chunks:")
    for chunk in chunks:
        print(chunk)


def test_semantic_chunking():
    chunks = semantic_chunking(sample_text)

    assert chunks is not None
    assert len(chunks) > 0

    print("\nSemantic chunks:")
    for chunk in chunks:
        print(chunk)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "func", [fixed_size_chunking, recursive_chunking, semantic_chunking]
)
def test_invalid_overlap_raises(func):
    with pytest.raises(ValueError):
        func(sample_text, chunk_size=100, overlap=100)


@pytest.mark.parametrize(
    "func", [fixed_size_chunking, recursive_chunking, semantic_chunking]
)
def test_empty_text_returns_empty(func):
    assert func("", chunk_size=100, overlap=20) == []


# ---------------------------------------------------------------------------
# Semantic regressions (issues seen in the first PDF run)
# ---------------------------------------------------------------------------

def test_semantic_keeps_decimals_and_abbreviations():
    text = (
        "The power dissipation is 23.4 mW at maximal clock rate = 100 MHz. "
        "Fig. 6 is the layout, where the chip area is 116x84 um2."
    )
    joined = " ".join(semantic_chunking(text, chunk_size=500, overlap=50))

    assert "23.4 mW" in joined
    assert "Fig. 6" in joined


def test_semantic_does_not_split_emails():
    text = (
        "Contact: ccwang@ee.nsysu.edu.tw for details. "
        "The proposed RNG passes the monobit test."
    )
    joined = " ".join(semantic_chunking(text, chunk_size=500, overlap=50))

    assert "ccwang@ee.nsysu.edu.tw" in joined


def test_semantic_respects_chunk_size():
    chunks = semantic_chunking(sample_text * 5, chunk_size=200, overlap=30)

    assert len(chunks) > 0
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_semantic_no_tiny_chunks():
    chunks = semantic_chunking(sample_text * 5, chunk_size=300, overlap=30)

    # allow at most one short chunk (a leftover that could not be merged)
    short = [c for c in chunks if len(c) < 40]
    assert len(short) <= 1


def test_semantic_removes_ieee_footer():
    text = "The RNG passes the long run test. " + FOOTER
    joined = " ".join(semantic_chunking(text, chunk_size=500, overlap=50))

    assert "Authorized licensed use" not in joined
    assert "Restrictions apply" not in joined


# ---------------------------------------------------------------------------
# chunk_pages
# ---------------------------------------------------------------------------

def _pages():
    return [
        {
            "page_number": 1,
            "text": "The proposed RNG uses SR latches. It runs at 100 MHz.\n" + FOOTER,
        },
        {
            "page_number": 2,
            "text": (
                "Conclusion text about metastability.\n"
                "References\n"
                "[1] C. Chien, Digital radio systems on a chip, 2001.\n" + FOOTER
            ),
        },
        {
            "page_number": 3,
            "text": "[2] A. Author, Another reference entry.",
        },
    ]


def test_chunk_pages_metadata_and_excludes_references():
    chunks = chunk_pages(
        _pages(),
        paper_id="paper-1",
        source_filename="paper.pdf",
        strategy="recursive",
        chunk_size=200,
        overlap=20,
    )

    assert len(chunks) > 0
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    assert all(c["paper_id"] == "paper-1" for c in chunks)
    assert all(c["chunking_method"] == "recursive" for c in chunks)

    joined = " ".join(c["text"] for c in chunks)
    assert "Chien" not in joined
    assert "Another reference" not in joined
    assert "Authorized licensed use" not in joined
    assert {c["page_number"] for c in chunks} <= {1, 2}


def test_chunk_pages_can_include_references():
    chunks = chunk_pages(
        _pages(),
        paper_id="paper-1",
        source_filename="paper.pdf",
        strategy="recursive",
        chunk_size=200,
        overlap=20,
        include_references=True,
    )

    joined = " ".join(c["text"] for c in chunks)
    assert "Chien" in joined


def test_chunk_pages_unsupported_strategy():
    with pytest.raises(ValueError):
        chunk_pages(_pages(), "p", "f.pdf", strategy="nope")


# ---------------------------------------------------------------------------
# Regressions from the second PDF run (no model needed for these)
# ---------------------------------------------------------------------------

def test_split_sentences_keeps_fig_label_with_number():
    text = (
        "The functions of the buffers at the outputs is to extend the path "
        "and delay thereof, if necessary. By contrast, Fig. 2 is another TERO "
        "composed of a NOR-based SR latch and inverters. "
        "By using Monte- Carlo simulation, Fig. 3 and Fig. 4 show the outputs."
    )
    sentences = split_sentences(text)

    assert not any(s.rstrip().endswith("Fig.") for s in sentences)
    assert any("Fig. 2 is another TERO" in s for s in sentences)
    assert any("Fig. 3 and Fig. 4 show" in s for s in sentences)


def test_split_sentences_moves_section_numeral_to_heading():
    text = (
        "Fig. 7. Output waveform at SS corner III. "
        "Simulation and Validation To manifest the randomness superiority."
    )
    sentences = split_sentences(text)

    assert not any(s.rstrip().endswith("III.") for s in sentences)
    assert any(s.startswith("III. Simulation") for s in sentences)


def test_split_sentences_keeps_table_numeral():
    text = "The output Q is undefined, as shown in Table II. Namely, it is unpredictable."
    assert any("Table II." in s for s in split_sentences(text))


def test_strip_footer_removes_conference_line_and_page_number():
    page = (
        "due to coupling between adjacent structures.\n"
        "The 2020 IEEE Asia Pacific Conference on Circuits and Systems (APCCAS)\n"
        "        978-1-7281-9396-0/20/$31.00 \u00a92020 IEEE\n"
        "98\n"
        "Authorized licensed use limited to: Some College. "
        "Downloaded on October 26,2026 at 04:05:03 UTC from IEEE Xplore.  "
        "Restrictions apply. \n"
    )
    cleaned = strip_footer(page)

    assert "coupling between adjacent structures." in cleaned
    assert "APCCAS" not in cleaned
    assert "978-" not in cleaned
    assert "98" not in cleaned
    assert "Authorized" not in cleaned


def test_strip_footer_keeps_table_numbers():
    table = "clock (MHz)\n33\n1\n0.5\n0.001\n100\narea (mm2)\n"
    assert strip_footer(table) == table