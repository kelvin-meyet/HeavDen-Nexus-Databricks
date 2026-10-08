from datetime import date
from pathlib import Path

import numpy as np
import pytest

from heavden.agent import corpus, retrieval

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "docs" / "corpus"
EVALSET = ROOT / "databricks" / "src" / "heavden" / "agent" / "evalsets" / "retrieval.yaml"

DOC = """---
doc_id: T-1
title: Test Protocol
version: "2.0"
status: current
effective_date: 2026-11-09
---

# Test Protocol

> **Fictional document.** Not guidance.

## 1. Scores

Call the team when the score is 5 or more.

| Score | Action |
|---|---|
| 5 | call |

## 2. Contacts

### 2.1 Site A

Dial 2222.

```json
{"a": 1}
```
"""


def _doc(tmp_path, text=DOC, name="t.md"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return corpus.load_document(path)


def test_front_matter_is_parsed(tmp_path):
    doc = _doc(tmp_path)
    assert (doc.doc_id, doc.version, doc.status) == ("T-1", "2.0", "current")
    assert doc.effective_date == date(2026, 11, 9)


def test_missing_front_matter_fields_fail(tmp_path):
    with pytest.raises(ValueError, match="front matter"):
        _doc(tmp_path, "---\ntitle: x\n---\nbody\n")


def test_chunks_follow_headings_and_drop_the_disclaimer(tmp_path):
    chunks = corpus.chunk_markdown(_doc(tmp_path), min_words=0)
    assert [c.section for c in chunks] == ["1. Scores", "2. Contacts > 2.1 Site A"]
    assert not any("Fictional" in c.text for c in chunks)
    assert "| 5 | call |" in chunks[0].text  # the table stays with its section
    assert '{"a": 1}' in chunks[1].text
    assert chunks[1].citation == "Test Protocol v2.0 > 2. Contacts > 2.1 Site A"
    assert chunks[0].embedding_text.startswith("Test Protocol (v2.0) > 1. Scores")


def test_tables_are_never_split(tmp_path):
    rows = "\n".join(f"| row {i} | value {i} |" for i in range(80))
    text = DOC.replace("| 5 | call |", rows)
    chunks = corpus.chunk_markdown(_doc(tmp_path, text), max_words=50)
    table_chunks = [c for c in chunks if "| row 0 |" in c.text]
    assert len(table_chunks) == 1 and "| row 79 |" in table_chunks[0].text


def test_real_corpus_loads_with_unique_chunk_ids():
    docs = corpus.load_corpus(CORPUS)
    assert len(docs) >= 9
    chunks = corpus.chunk_corpus(docs)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert {d.status for d in docs} == {"current", "superseded"}


def test_chunk_round_trips_through_dict(tmp_path):
    chunk = corpus.chunk_markdown(_doc(tmp_path))[0]
    assert corpus.Chunk.from_dict(chunk.to_dict()) == chunk


def test_tokenizer_keeps_codes():
    assert retrieval.tokenize("See FSN-2026-11 at SITE_B (firmware 3.2.0)") == [
        "see",
        "fsn-2026-11",
        "site_b",
        "firmware",
        "3.2.0",
    ]


def test_bm25_ranks_the_matching_text_first():
    bm25 = retrieval.BM25(["oxygen targets for copd", "battery outages", "rapid response team"])
    assert int(np.argmax(bm25.scores("copd oxygen"))) == 0
    assert bm25.scores("nothing matches").sum() == 0


def test_rrf_only_rewards_matched_items():
    vector = np.array([0.9, 0.8, 0.7])
    keyword = np.array([0.0, 0.0, 5.0])  # only item 2 matched the keywords
    fused = retrieval.reciprocal_rank_fusion([vector, keyword])
    assert fused[2] > fused[1]  # keyword match lifts item 2 over item 1
    assert fused[1] == pytest.approx(1 / 62)  # item 1 gets no credit from the keyword ranking


def test_in_force_picks_versions_by_date():
    chunks = corpus.chunk_corpus(corpus.load_corpus(CORPUS))
    protocol = [c for c in chunks if c.doc_id == "HD-CLIN-001"]
    current = {
        c.version for c, ok in zip(protocol, retrieval.in_force(protocol), strict=True) if ok
    }
    before = retrieval.in_force(protocol, as_of=date(2026, 11, 5))
    assert current == {"2.0"}
    assert {c.version for c, ok in zip(protocol, before, strict=True) if ok} == {"1.0"}


@pytest.fixture(scope="module")
def retriever():
    pytest.importorskip("faiss")
    chunks = corpus.chunk_corpus(corpus.load_corpus(CORPUS))
    return retrieval.Retriever(chunks, retrieval.HashingEmbedder())


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_search_finds_the_right_document(retriever, mode):
    top = retriever.search("Who do I call on RRT extension 4444 at Valley?", k=3, mode=mode)
    assert any(r.chunk.doc_id == "HD-OPS-001" for r in top)
    assert all(r.chunk.status == "current" for r in top)
    assert [r.rank for r in top] == [1, 2, 3]


def test_search_as_of_returns_the_old_protocol(retriever):
    old = retriever.search("NEWS2 score escalation", k=5, as_of=date(2026, 11, 5))
    assert {r.chunk.version for r in old if r.chunk.doc_id == "HD-CLIN-001"} == {"1.0"}


def test_save_and_load_give_the_same_results(retriever, tmp_path):
    retriever.save(tmp_path / "idx")
    loaded = retrieval.Retriever.load(tmp_path / "idx", retrieval.HashingEmbedder())
    q = "battery outage"
    assert [r.chunk.chunk_id for r in loaded.search(q)] == [
        r.chunk.chunk_id for r in retriever.search(q)
    ]
    with pytest.raises(ValueError, match="built with"):
        retrieval.Retriever.load(tmp_path / "idx", retrieval.HashingEmbedder(dim=64))


def test_prompt_cites_numbered_passages(retriever):
    results = retriever.search("SpO2 target for COPD", k=2)
    messages = retrieval.build_prompt("What is the SpO2 target for COPD?", results)
    assert messages[0]["role"] == "system" and "ONLY" in messages[0]["content"]
    assert f"[1] {results[0].chunk.citation}" in messages[1]["content"]
    assert retrieval.answerable(results, min_similarity=0.0)
    assert not retrieval.answerable(results, min_similarity=1.01)


def test_eval_set_is_well_formed_and_scored(retriever):
    questions = retrieval.load_questions(EVALSET)
    doc_ids = {d.doc_id for d in corpus.load_corpus(CORPUS)}
    assert {q.doc_id for q in questions if q.doc_id} <= doc_ids
    assert any(q.doc_id is None for q in questions)
    rows = retrieval.evaluate_retrieval(retriever, questions, k=4, mode="bm25")
    summary = retrieval.retrieval_summary(rows, k=4)
    assert summary["questions"] == sum(q.doc_id is not None for q in questions)
    assert 0.5 < summary["hit@4"] <= 1  # even plain keyword search finds most answers
