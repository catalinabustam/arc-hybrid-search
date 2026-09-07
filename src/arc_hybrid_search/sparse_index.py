"""BM25 sparse index management for one catalog's BM25 directory."""

from pathlib import Path

import bm25s
import Stemmer


def build_bm25_index(index_path: Path, documents: list[str]) -> None:
    """Build a BM25 index over `documents` and persist it to `index_path`."""
    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(documents, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(str(index_path), corpus=documents)


def load_bm25_index(index_path: Path) -> tuple[bm25s.BM25, Stemmer.Stemmer]:
    """Load a BM25 index previously built by `build_bm25_index`."""
    stemmer = Stemmer.Stemmer("english")
    retriever = bm25s.BM25.load(str(index_path), load_corpus=False)
    return retriever, stemmer


def bm25_retrieve(
    query: str,
    retriever,
    doc_ids: list[str],
    stemmer,
    top_k: int,
    allowed_doc_ids: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Retrieve top-k BM25 pairs, optionally keeping only allowed document IDs."""
    if not doc_ids:
        return []
    query_tokens = bm25s.tokenize(query, stemmer=stemmer, stopwords="en")
    k = len(doc_ids)
    lexical_indices, scores = retriever.retrieve(query_tokens, k=k)
    results = [
        (doc_ids[idx], float(score))
        for idx, score in zip(lexical_indices[0], scores[0])
        if allowed_doc_ids is None or doc_ids[idx] in allowed_doc_ids
    ]
    return results[:top_k]
