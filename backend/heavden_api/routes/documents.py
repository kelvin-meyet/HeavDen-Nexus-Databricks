"""Document search: the RAG retrieval step on its own (the chat endpoint will build on it)."""

from __future__ import annotations

from datetime import date
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from heavden.agent import retrieval
from heavden_api.deps import AppState, state

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("/search")
def search(
    q: str = Query(..., min_length=2, max_length=500),
    k: int = Query(4, ge=1, le=10),
    mode: Literal["hybrid", "vector", "bm25"] = "hybrid",
    as_of: date | None = Query(None, description="Use the document versions in force that day"),
    s: AppState = Depends(state),
) -> dict:
    """Passages that best answer `q`, each with its citation. `covered` is false when even the
    best match is weak, i.e. the documents probably don't answer the question."""
    retriever = s.retriever()
    if retriever is None:
        raise HTTPException(503, "this snapshot has no document index")
    results = retriever.search(q, k=k, mode=mode, as_of=as_of)
    # the similarity floor needs cosine scores, which BM25-only search doesn't produce
    covered = retrieval.answerable(results, s.settings.min_similarity) if mode != "bm25" else None
    return {
        "query": q,
        "covered": covered,
        "results": [
            {
                "rank": r.rank,
                "citation": r.chunk.citation,
                "doc_id": r.chunk.doc_id,
                "title": r.chunk.title,
                "version": r.chunk.version,
                "section": r.chunk.section,
                "similarity": None if mode == "bm25" else round(r.similarity, 4),
                "text": r.chunk.text,
            }
            for r in results
        ],
    }
