"""Local retrieval for the assistant's document tool: embeddings, FAISS, BM25 and hybrid search.

This is the **demo-mode** retriever (Plan.md §11) and the reference for the Databricks version:
on Databricks the same chunks go into a Vector Search index (managed embeddings); after the
credits end, the backend serves this FAISS index instead.

* `FastEmbedder`: a small ONNX sentence-embedding model (BAAI/bge-small-en-v1.5, 384 dims) via
  `fastembed`, no PyTorch. Install the `rag` dependency group.
* `HashingEmbedder`: a dependency-free bag-of-words embedder for tests and for learning.
* `BM25`: classic keyword ranking; good at exact terms (`etco2`, `FSN-2026-11`, `SITE_B`).
* `Retriever`: vector, BM25 or **hybrid** search (reciprocal rank fusion), with version
  filtering (current documents, or what was in force on a date) and a similarity floor so the
  assistant can say "not in the documents" instead of answering from a weak match.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from heavden.agent.corpus import Chunk

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
RRF_K = 60  # standard constant for reciprocal rank fusion

_TOKEN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be by do does for from has have how i if in is it its of on or "
    "should that the this to was what when where which who why will with".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens; keeps codes like `fsn-2026-11`, `site_b` and `3.2.0` intact."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norms == 0, 1, norms)


class HashingEmbedder:
    """Bag-of-words counts hashed into `dim` buckets, then L2-normalised. Lexical only: it knows
    "oxygen" matches "oxygen" but not that it relates to "SpO2"."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for token, count in Counter(tokenize(text)).items():
            # a stable hash (Python's hash() changes between runs)
            v[int.from_bytes(token.encode(), "little") % 2_147_483_647 % self.dim] += count
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class FastEmbedder:
    """Sentence embeddings with `fastembed` (ONNX). Queries and passages are embedded
    differently, as BGE models expect."""

    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: str | Path | None = None):
        from fastembed import TextEmbedding

        self.name = model
        self._model = TextEmbedding(model, cache_dir=str(cache_dir) if cache_dir else None)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.stack(list(self._model.passage_embed(texts))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(next(iter(self._model.query_embed([text]))))


class BM25:
    """Okapi BM25 over pre-tokenised texts."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg_length = self.lengths.mean() if len(self.docs) else 0.0
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: np.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / max(self.avg_length, 1e-9))
        for term in set(tokenize(query)):
            if term not in self.idf:
                continue
            tf = np.array([d.get(term, 0) for d in self.docs], dtype=float)
            out += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return out


def in_force(chunks: list[Chunk], as_of: date | None = None) -> np.ndarray:
    """Mask of chunks to search: `status == current`, or, with `as_of`, the latest version of
    each document whose effective date is on or before `as_of`."""
    if as_of is None:
        return np.array([c.status == "current" for c in chunks])
    latest: dict[str, date] = {}
    for c in chunks:
        if c.effective_date <= as_of and c.effective_date > latest.get(c.doc_id, date.min):
            latest[c.doc_id] = c.effective_date
    return np.array([latest.get(c.doc_id) == c.effective_date for c in chunks])


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    rank: int
    score: float  # what the ranking used (cosine, BM25 or fused RRF score)
    similarity: float  # cosine similarity to the query (NaN for BM25-only search)


class Retriever:
    """Search over chunks with a FAISS inner-product index (cosine on normalised vectors)."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder, vectors: np.ndarray | None = None):
        import faiss

        self.chunks = list(chunks)
        self.embedder = embedder
        if vectors is None:
            vectors = embedder.embed_documents([c.embedding_text for c in self.chunks])
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)
        self.bm25 = BM25([c.embedding_text for c in self.chunks])

    def _vector_scores(self, query: str) -> np.ndarray:
        q = self.embedder.embed_query(query).reshape(1, -1).astype(np.float32)
        sims, ids = self.index.search(q, len(self.chunks))
        out = np.empty(len(self.chunks), dtype=float)
        out[ids[0]] = sims[0]
        return out

    def search(
        self, query: str, k: int = 4, mode: str = "hybrid", as_of: date | None = None
    ) -> list[SearchResult]:
        """Top `k` chunks for `query`. `mode` is "vector", "bm25" or "hybrid"."""
        allowed = in_force(self.chunks, as_of)
        sims = self._vector_scores(query) if mode in ("vector", "hybrid") else None
        if mode == "vector":
            score = sims
        elif mode == "bm25":
            score = self.bm25.scores(query)
        elif mode == "hybrid":
            score = reciprocal_rank_fusion([sims, self.bm25.scores(query)], allowed)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        score = np.where(allowed, score, -np.inf)
        order = [i for i in np.argsort(-score, kind="stable") if allowed[i]][:k]
        return [
            SearchResult(
                self.chunks[i],
                rank + 1,
                float(score[i]),
                float(sims[i]) if sims is not None else float("nan"),
            )
            for rank, i in enumerate(order)
        ]

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        with open(folder / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict()) + "\n")
        (folder / "meta.json").write_text(
            json.dumps({"embedder": self.embedder.name, "n_chunks": len(self.chunks)}, indent=1)
        )

    @classmethod
    def load(cls, folder: Path, embedder: Embedder) -> Retriever:
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text())
        if meta["embedder"] != embedder.name:
            raise ValueError(f"index built with {meta['embedder']}, not {embedder.name}")
        with open(folder / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk.from_dict(json.loads(line)) for line in f]
        return cls(chunks, embedder, np.load(folder / "vectors.npy"))


def reciprocal_rank_fusion(
    score_lists: list[np.ndarray], allowed: np.ndarray | None = None, k: int = RRF_K
) -> np.ndarray:
    """Sum of 1 / (k + rank) across rankings. Uses ranks only, so scores on different scales
    (cosine, BM25) can be combined without tuning weights.

    A ranking only votes for items it actually matched (score > 0). Otherwise, when BM25 finds
    no query word in most chunks, their arbitrary tie order would still earn them credit.
    """
    fused = np.zeros(len(score_lists[0]))
    for scores in score_lists:
        s = np.asarray(scores, dtype=float)
        s = np.where(allowed, s, -np.inf) if allowed is not None else s
        ranks = np.empty(len(s))
        ranks[np.argsort(-s, kind="stable")] = np.arange(1, len(s) + 1)
        fused += np.where(s > 0, 1 / (k + ranks), 0.0)
    return fused


def answerable(results: list[SearchResult], min_similarity: float) -> bool:
    """Whether the best match is close enough to answer from. Below the floor, the assistant
    should say the documents don't cover the question."""
    return bool(results) and max(r.similarity for r in results) >= min_similarity


def format_context(results: list[SearchResult]) -> str:
    """Numbered context passages, each with its citation."""
    return "\n\n".join(f"[{r.rank}] {r.chunk.citation}\n{r.chunk.text}" for r in results)


SYSTEM_PROMPT = """You answer questions for staff of the (fictional) HeavDen hospital network
using ONLY the numbered document passages provided.
- Cite every claim with its passage number, e.g. [2].
- If the passages don't contain the answer, say so. Don't guess or use outside knowledge.
- Don't give clinical advice beyond what the protocols say. All data and documents are synthetic.
"""


def build_prompt(question: str, results: list[SearchResult]) -> list[dict[str, str]]:
    """Chat messages for an LLM: the rules, then the passages and the question."""
    user = f"Passages:\n\n{format_context(results)}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# --- Retrieval evaluation --------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalQuestion:
    question: str
    doc_id: str | None  # None: the corpus doesn't answer it (should be refused)
    version: str | None = None
    section: str | None = None  # a substring of the expected chunk's section
    as_of: date | None = None


def load_questions(path: Path) -> list[RetrievalQuestion]:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [
        RetrievalQuestion(
            question=q["question"],
            doc_id=q.get("doc_id"),
            version=str(q["version"]) if q.get("version") else None,
            section=q.get("section"),
            as_of=q.get("as_of"),
        )
        for q in raw["questions"]
    ]


def is_match(chunk: Chunk, q: RetrievalQuestion) -> bool:
    return (
        chunk.doc_id == q.doc_id
        and (q.version is None or chunk.version == q.version)
        and (q.section is None or q.section.lower() in chunk.section.lower())
    )


def evaluate_retrieval(
    retriever: Retriever, questions: list[RetrievalQuestion], k: int = 4, mode: str = "hybrid"
) -> list[dict]:
    """Per answerable question: rank of the first correct chunk (None if not in the top k),
    and the best similarity. Summarise with `retrieval_summary`."""
    rows = []
    for q in questions:
        results = retriever.search(q.question, k=k, mode=mode, as_of=q.as_of)
        rank = next((r.rank for r in results if q.doc_id and is_match(r.chunk, q)), None)
        rows.append(
            {
                "question": q.question,
                "answerable": q.doc_id is not None,
                "rank": rank,
                "top_similarity": results[0].similarity if results else float("nan"),
                "top_citation": results[0].chunk.citation if results else None,
            }
        )
    return rows


def retrieval_summary(rows: list[dict], k: int = 4) -> dict[str, float]:
    """hit@1, hit@k and mean reciprocal rank over answerable questions."""
    # a missed question has rank None (or NaN once it has been through a DataFrame)
    ranks = [r["rank"] if pd.notna(r["rank"]) else None for r in rows if r["answerable"]]
    return {
        "questions": len(ranks),
        "hit@1": float(np.mean([r == 1 for r in ranks])),
        f"hit@{k}": float(np.mean([r is not None and r <= k for r in ranks])),
        "mrr": float(np.mean([1 / r if r is not None else 0.0 for r in ranks])),
    }
