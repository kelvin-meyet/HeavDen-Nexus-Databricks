"""The document corpus behind the assistant's RAG tool: loading and chunking (Plan.md §5.5, §11).

Documents are Markdown files in `docs/corpus/` with YAML front matter (`doc_id`, `title`,
`version`, `status`, `effective_date`, ...). Several versions of one document can coexist
(e.g. the escalation protocol v1.0 and v2.0), so every chunk carries its document's version and
dates, and retrieval can filter to what was in force at a given time.

Chunking follows the document's own structure: a chunk never crosses a heading, tables and code
blocks are never split, and each chunk knows its heading path (its "section"), which becomes
part of the citation. `chunk_fixed` is the naive baseline, kept for comparison.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

# Boilerplate banners at the top of every document; they'd match every query, so they're dropped.
DISCLAIMER_PREFIXES = ("> **Fictional", "> **Synthetic", "> **All data")


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    version: str
    status: str  # current | superseded
    effective_date: date
    body: str
    path: str
    meta: dict = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    version: str
    status: str
    effective_date: date
    section: str  # heading path below the title, e.g. "3. Escalation by NEWS2 score"
    text: str
    path: str

    @property
    def citation(self) -> str:
        section = f" > {self.section}" if self.section else ""
        return f"{self.title} v{self.version}{section}"

    @property
    def embedding_text(self) -> str:
        """What gets embedded: the text plus its context, so a chunk that just says
        "call within 30 minutes" still knows which protocol and section it belongs to."""
        return f"{self.title} (v{self.version}) > {self.section}\n\n{self.text}"

    def to_dict(self) -> dict:
        return {**self.__dict__, "effective_date": self.effective_date.isoformat()}

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(**{**d, "effective_date": date.fromisoformat(d["effective_date"])})


def parse_front_matter(text: str) -> tuple[dict, str]:
    match = re.match(r"^---\n(.*?)\n---\n", text, flags=re.DOTALL)
    if not match:
        return {}, text
    return yaml.safe_load(match.group(1)) or {}, text[match.end() :]


def load_document(path: Path) -> Document:
    meta, body = parse_front_matter(path.read_text(encoding="utf-8").replace("\r\n", "\n"))
    missing = {"doc_id", "title", "version", "status", "effective_date"} - meta.keys()
    if missing:
        raise ValueError(f"{path.name}: front matter lacks {sorted(missing)}")
    effective = meta["effective_date"]
    return Document(
        doc_id=str(meta["doc_id"]),
        title=str(meta["title"]),
        version=str(meta["version"]),
        status=str(meta["status"]),
        effective_date=effective if isinstance(effective, date) else date.fromisoformat(effective),
        body=body,
        path=path.as_posix(),
        meta=meta,
    )


def load_corpus(folder: Path) -> list[Document]:
    return [load_document(p) for p in sorted(Path(folder).glob("*.md"))]


def _blocks(body: str) -> list[tuple[str, str]]:
    """Split Markdown into ("heading", text) and ("block", text) items.

    A block is a paragraph, list, table or fenced code block; tables and code are kept whole.
    """
    items: list[tuple[str, str]] = []
    buf: list[str] = []
    in_code = False

    def flush():
        if buf:
            items.append(("block", "\n".join(buf).strip()))
            buf.clear()

    for line in body.split("\n"):
        if line.startswith("```"):
            buf.append(line)
            in_code = not in_code
            if not in_code:
                flush()
            continue
        if in_code:
            buf.append(line)
        elif re.match(r"^#{1,6} ", line):
            flush()
            items.append(("heading", line))
        elif not line.strip():
            flush()
        elif buf and buf[-1].startswith("|") != line.startswith("|"):
            flush()  # a table starts or ends without a blank line
            buf.append(line)
        else:
            buf.append(line)
    flush()
    return items


def _words(text: str) -> int:
    return len(text.split())


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "intro"


def _make_chunk(doc: Document, section: str, n: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}@{doc.version}#{_slug(section.split(' > ')[-1])}-{n}",
        doc_id=doc.doc_id,
        title=doc.title,
        version=doc.version,
        status=doc.status,
        effective_date=doc.effective_date,
        section=section,
        text=text,
        path=doc.path,
    )


def chunk_markdown(doc: Document, max_words: int = 220, min_words: int = 40) -> list[Chunk]:
    """Heading-aware chunks of at most `max_words`.

    A single oversized block (e.g. a big table) stays whole, and a chunk is never closed while
    shorter than `min_words`, so a one-line intro stays with the table it introduces.
    """
    chunks: list[Chunk] = []
    headings: list[tuple[int, str]] = []  # (level, text) stack below the H1 title
    blocks: list[str] = []
    seen_h2 = False

    def section() -> str:
        return " > ".join(text for _, text in headings)

    def flush():
        if not blocks:
            return
        current: list[str] = []
        for block in blocks:
            text = "\n\n".join(current)
            if _words(text) >= min_words and _words(f"{text}\n\n{block}") > max_words:
                chunks.append(_make_chunk(doc, section(), len(chunks), "\n\n".join(current)))
                current = []
            current.append(block)
        chunks.append(_make_chunk(doc, section(), len(chunks), "\n\n".join(current)))
        blocks.clear()

    for kind, text in _blocks(doc.body):
        if kind == "heading":
            flush()
            level = len(text) - len(text.lstrip("#"))
            if level == 1:
                continue  # the title is part of every citation already
            seen_h2 = True
            headings = [h for h in headings if h[0] < level] + [(level, text.lstrip("# ").strip())]
        elif seen_h2 or not text.startswith(DISCLAIMER_PREFIXES):
            blocks.append(text)
    flush()
    return chunks


def chunk_fixed(doc: Document, size_words: int = 120, overlap_words: int = 20) -> list[Chunk]:
    """Naive baseline: fixed windows of words, ignoring structure (cuts tables and sentences)."""
    words = doc.body.split()
    step = size_words - overlap_words
    return [
        _make_chunk(doc, "", i, " ".join(words[start : start + size_words]))
        for i, start in enumerate(range(0, max(len(words) - overlap_words, 1), step))
    ]


def chunk_corpus(docs: list[Document], max_words: int = 220, min_words: int = 40) -> list[Chunk]:
    chunks = [c for d in docs for c in chunk_markdown(d, max_words, min_words)]
    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate chunk ids")
    return chunks
