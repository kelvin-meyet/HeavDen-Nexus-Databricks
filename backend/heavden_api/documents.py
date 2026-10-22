"""The website's document index, built when the image is built (backend/Dockerfile).

    python -m heavden_api.documents <folder> [--cache-dir <embedding model cache>]

The documents are in git (docs/corpus, shipped in the package), so their index goes with the
code that cites them: a change to a document reaches the website with the next image. The
snapshot holds only data, exported from Databricks (heavden_api.export).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from heavden.agent.corpus import chunk_corpus, load_corpus
from heavden.agent.publish import corpus_dir
from heavden.agent.retrieval import Embedder, FastEmbedder, Retriever


def build_index(folder: Path, embedder: Embedder) -> int:
    """Every chunk of the documents, embedded and saved to `folder`; returns the chunk count."""
    chunks = chunk_corpus(load_corpus(corpus_dir()))
    Retriever(chunks, embedder).save(folder)
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    n = build_index(args.folder, FastEmbedder(cache_dir=args.cache_dir))
    print(f"document index: {n} chunks in {args.folder}")
