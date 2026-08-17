"""
embeddings/specter2_embedder.py — SPECTER2 embeddings for papers AND queries.

Two SPECTER2 adapters are loaded onto the shared allenai/specter2_base model:
  - "allenai/specter2"            (proximity)   -> paper embeddings (title+abstract)
  - "allenai/specter2_adhoc_query" (adhoc query) -> query embeddings for search

Both adapters live on ONE base model instance to avoid loading BERT twice;
embed_papers()/embed_query() just switch the active adapter before the
forward pass. See https://huggingface.co/allenai/specter2_base
"""

from typing import List, Optional

import numpy as np
import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

BASE_MODEL = "allenai/specter2_base"
PAPER_ADAPTER = "allenai/specter2"
QUERY_ADAPTER = "allenai/specter2_adhoc_query"
MAX_LENGTH = 512
EMBEDDING_DIM = 768  # bert-base hidden size


class Specter2Embedder:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.model = AutoAdapterModel.from_pretrained(BASE_MODEL)
        self.paper_adapter = self.model.load_adapter(PAPER_ADAPTER, source="hf", set_active=False)
        self.query_adapter = self.model.load_adapter(QUERY_ADAPTER, source="hf", set_active=False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _embed(self, texts: List[str], adapter_name: str, batch_size: int) -> np.ndarray:
        self.model.set_active_adapters(adapter_name)
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)
            output = self.model(**inputs)
            # SPECTER2's embedding is the [CLS] token of the last hidden state
            cls_vecs = output.last_hidden_state[:, 0, :]
            all_vecs.append(cls_vecs.cpu().numpy())
        return np.vstack(all_vecs).astype("float32")

    def embed_papers(self, titles: List[str], abstracts: List[str], batch_size: int = 32) -> np.ndarray:
        """Paper embeddings, SPECTER2-style: title [SEP] abstract. Uses the
        proximity adapter — this is what gets stored in FAISS."""
        texts = [
            (t or "") + self.tokenizer.sep_token + (a or "")
            for t, a in zip(titles, abstracts)
        ]
        return self._embed(texts, self.paper_adapter, batch_size)

    def embed_query(self, query: str) -> np.ndarray:
        """Single free-text query embedding, using the adhoc-query adapter
        (tuned for query -> paper retrieval, distinct from paper -> paper)."""
        return self._embed([query], self.query_adapter, batch_size=1)[0]