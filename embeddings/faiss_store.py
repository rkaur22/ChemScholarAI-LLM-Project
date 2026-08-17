"""
persistent FAISS index for paper embeddings.

Cosine similarity is implemented as inner product over L2-normalized
vectors, wrapped in an IndexIDMap2 so each vector is addressable by its
PostgreSQL `papers.id` (no separate id-mapping file needed).
"""

from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np


class FaissStore:
    def __init__(self, dim: int, index_path: Path):
        self.dim = dim
        self.index_path = Path(index_path)
        self.index = self._load_or_create()

    def _load_or_create(self):
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))
        flat = faiss.IndexFlatIP(self.dim)
        return faiss.IndexIDMap2(flat)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        vectors = np.ascontiguousarray(vectors.astype("float32"))
        faiss.normalize_L2(vectors)
        return vectors

    def add(self, ids: List[int], vectors: np.ndarray) -> None:
        vectors = self._normalize(vectors)
        ids_arr = np.array(ids, dtype="int64")
        # Drop any existing vectors for these ids first, so re-running the
        # script on already-embedded papers replaces rather than duplicates.
        self.index.remove_ids(ids_arr)
        self.index.add_with_ids(vectors, ids_arr)

    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[List[int], List[float]]:
        query_vector = self._normalize(query_vector.reshape(1, -1))
        scores, ids = self.index.search(query_vector, k)
        return ids[0].tolist(), scores[0].tolist()

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))