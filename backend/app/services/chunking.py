from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter
def fixed_size_chunking(
    text,
    chunk_size=500,
    overlap=100
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)

        if end == len(text):
            break

        start += chunk_size - overlap

    return chunks




def recursive_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

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


import numpy as np

from sentence_transformers import SentenceTransformer


# Load the embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def semantic_chunking(
    text,
    similarity_threshold=0.65
):

    sentences = [
        sentence.strip()
        for sentence in text.replace("\n", " ").split(".")
        if sentence.strip()
    ]

    if not sentences:
        return []

    if len(sentences) == 1:
        return sentences

    embeddings = model.encode(
        sentences,
        normalize_embeddings=True
    )

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = np.dot(embeddings[i - 1], embeddings[i])

        if similarity >= similarity_threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentences[i]]

    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks

def chunk_pages(
    pages: list[dict],
    paper_id: str,
    source_filename: str,
    strategy: str = "recursive",
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:

    chunkers = {
        "fixed": fixed_size_chunking,
        "recursive": recursive_chunking,
        "semantic": semantic_chunking,
    }

    if strategy not in chunkers:
        raise ValueError("Unsupported chunking strategy")

    all_chunks = []

    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"]

        if not page_text or not page_text.strip():
            continue

        if strategy == "semantic":
            page_chunks = semantic_chunking(page_text)
        else:
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


