import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import re
from functools import lru_cache
from uuid import uuid4

import numpy as np
import pysbd
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FOOTER_RE = re.compile(
    r"(?:The \d{4} IEEE [^\n]*\n\s*978-[\d\-/$.]+\s*©\d{4} IEEE\s*)?"  # conference line
    r"(?:\n?\s*\d{1,3}\s*\n)?"                                          # page number
    r"\s*Authorized licensed use limited to:.*?Restrictions apply\.",
    re.S,
)
REF_RE = re.compile(r"\n\s*References\s*\n", re.I)

# A sentence ending in one of these labels was split too early ("Fig." | "2 is ...")
LABEL_END_RE = re.compile(r"\b(?:Fig|Figs|Eq|Eqs|Ref|Refs|vs)\.$", re.I)
# Section numeral stuck to the end of the previous sentence ("... corner III.")
HEADING_TAIL_RE = re.compile(
    r"^(.*\S)\s+((?<!Table )(?<!Section )[IVX]{2,}\.)$", re.S
)

_segmenter = pysbd.Segmenter(language="en", clean=False)


def _validate(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")


@lru_cache(maxsize=1)
def get_model():
    # Lazy load so importing this module (and running other tests) is fast
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def strip_footer(text: str) -> str:
    """Remove the IEEE Xplore 'Authorized licensed use...' footer."""
    return FOOTER_RE.sub(" ", text)


def clean_page_text(text: str) -> str:
    """Strip footer and flatten whitespace (used by semantic chunking)."""
    text = strip_footer(text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def fix_sentence_boundaries(sentences: list[str]) -> list[str]:
    out: list[str] = []
    carry = ""

    for s in sentences:
        if carry:
            s = f"{carry} {s}"
            carry = ""

        if out and LABEL_END_RE.search(out[-1]):
            out[-1] = f"{out[-1]} {s}"
        else:
            out.append(s)

        m = HEADING_TAIL_RE.match(out[-1])
        if m:
            out[-1], carry = m.group(1), m.group(2)

    if carry:
        out.append(carry)

    return out


def split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _segmenter.segment(text) if s.strip()]
    return fix_sentence_boundaries(sentences)


def _split_long(sentences: list[str], max_len: int) -> list[str]:
    """Break any single sentence longer than max_len on spaces."""
    out = []
    for s in sentences:
        while len(s) > max_len:
            cut = s.rfind(" ", 0, max_len)
            cut = cut if cut > 0 else max_len
            out.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def fixed_size_chunking(
    text,
    chunk_size=500,
    overlap=100,
):
    _validate(chunk_size, overlap)

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start += chunk_size - overlap

    return chunks


def recursive_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    _validate(chunk_size, overlap)

    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        strip_whitespace=True,
    )

    return splitter.split_text(text)


def semantic_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    similarity_threshold: float = 0.65,
    min_chunk_chars: int = 100,
) -> list[str]:
    """
    Sentence-similarity chunking with a hard chunk_size cap, sentence-level
    overlap on size-based splits, and a minimum chunk length so tiny
    fragments are not emitted.
    """
    _validate(chunk_size, overlap)

    text = clean_page_text(text)
    if not text:
        return []

    sentences = _split_long(split_sentences(text), chunk_size)

    if not sentences:
        return []

    if len(sentences) == 1:
        return sentences

    embeddings = get_model().encode(sentences, normalize_embeddings=True)

    chunks = []
    current = [sentences[0]]
    current_len = len(sentences[0])

    for i in range(1, len(sentences)):
        sentence = sentences[i]
        similarity = float(np.dot(embeddings[i - 1], embeddings[i]))

        too_big = current_len + 1 + len(sentence) > chunk_size
        topic_break = (
            similarity < similarity_threshold
            and current_len >= min_chunk_chars
        )

        if too_big or topic_break:
            chunks.append(" ".join(current))

            carry, carry_len = [], 0
            if too_big:  # overlap only on size-based splits
                for prev in reversed(current):
                    if carry_len + len(prev) > overlap:
                        break
                    carry.insert(0, prev)
                    carry_len += len(prev) + 1

                if carry_len + len(sentence) + 1 > chunk_size:
                    carry = []

            current = carry + [sentence]
            current_len = len(" ".join(current))
        else:
            current.append(sentence)
            current_len += 1 + len(sentence)

    chunks.append(" ".join(current))

    # Merge a tiny trailing chunk into the previous one if it fits
    if len(chunks) > 1 and len(chunks[-1]) < min_chunk_chars:
        if len(chunks[-2]) + 1 + len(chunks[-1]) <= chunk_size:
            chunks[-2] = chunks[-2] + " " + chunks[-1]
            chunks.pop()

    return chunks


# ---------------------------------------------------------------------------
# Page-level pipeline
# ---------------------------------------------------------------------------

def chunk_pages(
    pages: list[dict],
    paper_id: str,
    source_filename: str,
    strategy: str = "recursive",
    chunk_size: int = 500,
    overlap: int = 50,
    include_references: bool = False,
) -> list[dict]:

    chunkers = {
        "fixed": fixed_size_chunking,
        "recursive": recursive_chunking,
        "semantic": semantic_chunking,
    }

    if strategy not in chunkers:
        raise ValueError("Unsupported chunking strategy")

    all_chunks = []
    in_references = False

    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"]

        if not page_text or not page_text.strip():
            continue

        page_text = strip_footer(page_text)

        if not include_references:
            if in_references:
                continue

            match = REF_RE.search(page_text)
            if match:
                page_text = page_text[: match.start()]
                in_references = True

        if not page_text.strip():
            continue

        page_chunks = chunkers[strategy](
            page_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk in page_chunks:
            if not chunk.strip():
                continue

            all_chunks.append({
                "chunk_id": str(uuid4()),
                "paper_id": paper_id,
                "source_filename": source_filename,
                "page_number": page_number,
                "chunking_method": strategy,
                "text": chunk,
            })

    return all_chunks