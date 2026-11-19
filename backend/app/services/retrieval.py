"""
Retrieval Service for ResearchLens AI.

Provides an in-memory vector index powered by SentenceTransformers (all-MiniLM-L6-v2).
Stores chunk text with page and paper metadata, and supports cosine similarity search
with optional paper filtering (by paper_id or source_filename) and cross-paper search.
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from typing import Any, Dict, List, Optional
import numpy as np

from backend.app.services.chunking import get_model


class InMemoryVectorIndex:
    """
    Beginner-friendly in-memory vector store using NumPy for cosine similarity.
    
    Each stored record contains:
      - chunk_id (str)
      - paper_id (str)
      - source_filename (str)
      - page_number (int)
      - chunking_method (str)
      - text (str)
      - embedding (np.ndarray, normalized)
    """

    def __init__(self, model=None):
        self._model = model if model is not None else get_model()
        self._chunks: List[Dict[str, Any]] = []
        self._embeddings: Optional[np.ndarray] = None  # 2D array of shape (N, dim)

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Embeds and stores a list of chunk dicts.
        
        Expected keys per chunk:
          - chunk_id, paper_id, page_number, chunking_method, text
          - optional: source_filename
        
        Returns the number of chunks added.
        """
        valid_chunks = [c for c in chunks if c.get("text", "").strip()]
        if not valid_chunks:
            return 0

        texts = [c["text"] for c in valid_chunks]
        # normalize_embeddings=True ensures embeddings have unit norm (L2 = 1.0)
        # Therefore, dot product directly equals cosine similarity!
        new_embeddings = self._model.encode(texts, normalize_embeddings=True)
        new_embeddings = np.asarray(new_embeddings, dtype=np.float32)

        if self._embeddings is None or len(self._chunks) == 0:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        # Store metadata copies (shallow copy is fine)
        for c in valid_chunks:
            self._chunks.append({
                "chunk_id": c.get("chunk_id"),
                "paper_id": c.get("paper_id"),
                "source_filename": c.get("source_filename", ""),
                "page_number": c.get("page_number"),
                "chunking_method": c.get("chunking_method"),
                "text": c.get("text"),
            })

        return len(valid_chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        paper_id: Optional[str] = None,
        source_filename: Optional[str] = None,
        paper_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Finds the top-k most similar chunks to a query.

        Parameters:
          query: User query or research question.
          top_k: Number of results to return.
          paper_id: Optional filter. If specified, only chunks from this paper_id are searched.
          paper_ids: Optional set of paper IDs. If specified, only chunks from selected papers are searched.
                     An empty list returns no results.
          source_filename: Optional filter. If specified, only chunks from this filename are searched.
                           If neither is specified, performs cross-paper search.

        Returns:
          List of result dicts, sorted by similarity score descending:
          [
            {
              "chunk_id": ...,
              "paper_id": ...,
              "source_filename": ...,
              "page_number": ...,
              "chunking_method": ...,
              "text": ...,
              "score": float (cosine similarity between -1.0 and 1.0)
            },
            ...
          ]
        """
        if not query.strip() or self._embeddings is None or len(self._chunks) == 0:
            return []
        if paper_ids is not None and len(paper_ids) == 0:
            return []

        # 1. Determine candidate indices based on filters
        candidate_indices: List[int] = []
        selected_paper_ids = set(paper_ids) if paper_ids is not None else None
        for idx, chunk in enumerate(self._chunks):
            if paper_id is not None and chunk.get("paper_id") != paper_id:
                continue
            if selected_paper_ids is not None and chunk.get("paper_id") not in selected_paper_ids:
                continue
            if source_filename is not None and chunk.get("source_filename") != source_filename:
                continue
            candidate_indices.append(idx)

        if not candidate_indices:
            return []

        # 2. Encode query vector
        query_embedding = self._model.encode([query], normalize_embeddings=True)
        query_vec = np.asarray(query_embedding[0], dtype=np.float32)

        # 3. Calculate cosine similarity for candidate chunks
        candidate_vectors = self._embeddings[candidate_indices]
        scores = np.dot(candidate_vectors, query_vec)

        # 4. Sort and select top_k
        top_k = max(1, min(top_k, len(candidate_indices)))
        # Argsort gives ascending order, so take [-top_k:] and reverse
        ranked_local_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for local_idx in ranked_local_indices:
            global_idx = candidate_indices[local_idx]
            chunk_data = dict(self._chunks[global_idx])
            chunk_data["score"] = float(round(scores[local_idx], 4))
            results.append(chunk_data)

        return results

    def get_paper_ids(self) -> List[str]:
        """Returns a list of unique paper_ids present in the index."""
        return sorted(list(set(c["paper_id"] for c in self._chunks if c.get("paper_id"))))

    def get_source_filenames(self) -> List[str]:
        """Returns a list of unique source_filenames present in the index."""
        return sorted(list(set(c["source_filename"] for c in self._chunks if c.get("source_filename"))))

    def clear(self) -> None:
        """Clears all indexed chunks and embeddings."""
        self._chunks.clear()
        self._embeddings = None

    def __len__(self) -> int:
        return len(self._chunks)


# Module-level default index for convenience
_default_index: Optional[InMemoryVectorIndex] = None


def get_default_vector_index() -> InMemoryVectorIndex:
    """Returns a shared InMemoryVectorIndex instance."""
    global _default_index
    if _default_index is None:
        _default_index = InMemoryVectorIndex()
    return _default_index
