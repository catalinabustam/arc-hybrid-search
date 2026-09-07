"""Public retrieval API: `HybridSearchIndex`."""

from pathlib import Path

import pandas as pd

from .config import DEFAULT_DATA_DIR, EMBEDDING_MODEL, catalog_dir
from .documents import build_ids
from .fusion import joint_dense_retrieve, reciprocal_rank_fusion
from .sparse_index import bm25_retrieve, load_bm25_index
from .vector_store import METADATA_COLUMNS, load_collections


class HybridSearchIndex:
    """Loads an already-built catalog ("raw" or "expanded") and answers
    `retrieve()` calls. Building is a separate, explicit step — see
    `build_index()`. This class only ever reads what's already on disk, and
    loads each catalog's resources lazily, the first time it's requested.
    """

    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR, model_name: str = EMBEDDING_MODEL):
        self._data_dir = Path(data_dir)
        self._model_name = model_name
        self._loaded: dict[str, dict] = {}

    def retrieve(
        self,
        query: str,
        catalog: str = "expanded",
        top_k: int = 5,
        metadata_filter: dict[str, list[str]] | None = None,
    ) -> list[dict]:
        """Return up to `top_k` matching ARC rows for `query`.

        Parameters
        ----------
        query : free-text question to search for.
        catalog : "raw" for the untouched ARC catalog, "expanded" for the
            list/option-expanded version (matches individual list items and
            answer options directly).
        top_k : maximum number of results to return.
        metadata_filter : e.g. `{"section": ["Demographics"]}` — columns are
            AND-combined, values within a column are OR-combined. `None`
            (the default) searches the whole catalog.

        Returns
        -------
        A list of dicts sorted by descending relevance, each with
        `row_index`, `question`, `definition`, `section`, `form`,
        `variable`, and `score` (0-1, normalized RRF score).
        """
        resources = self._load(catalog)
        df = resources["df"]

        where = _chroma_where(metadata_filter)
        allowed_doc_ids = _allowed_doc_ids(df, resources["ids"], metadata_filter)

        search_size = len(allowed_doc_ids) if allowed_doc_ids is not None else len(resources["ids"])
        if search_size == 0:
            return []

        dense_results = joint_dense_retrieve(
            query,
            [resources["ques_def_coll"], resources["questions_coll"]],
            top_k=search_size,
            where=where,
        )
        sparse_results = bm25_retrieve(
            query,
            resources["bm25_retriever"],
            resources["ids"],
            resources["stemmer"],
            top_k=search_size,
            allowed_doc_ids=allowed_doc_ids,
        )
        fused = reciprocal_rank_fusion([dense_results, sparse_results])

        id_to_row = {doc_id: i for i, doc_id in enumerate(resources["ids"])}
        output = []
        for doc_id, scores in fused:
            row_index = id_to_row.get(doc_id)
            if row_index is None:
                continue
            if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
                continue
            row = df.iloc[row_index]
            output.append(
                {
                    "row_index": row_index,
                    "question": row.get("Question", ""),
                    "definition": row.get("Definition", ""),
                    "section": row.get("Section", ""),
                    "form": row.get("Form", ""),
                    "variable": row.get("Variable", ""),
                    "type": row.get("Type", ""),
                    "score": scores["normalized_score"],
                }
            )
            if len(output) >= top_k:
                break
        return output

    def _load(self, catalog: str) -> dict:
        if catalog in self._loaded:
            return self._loaded[catalog]

        cat_dir = catalog_dir(self._data_dir, catalog)
        csv_path = cat_dir / f"arc_{catalog}.csv"
        if not csv_path.exists():
            raise RuntimeError(f"No '{catalog}' index found at {cat_dir}. Run build_index() first.")

        df = pd.read_csv(csv_path, dtype=str).fillna("")
        documents = build_documents(df)
        ids = build_ids(df)
        questions_coll, ques_def_coll = load_collections(cat_dir / "chroma", self._model_name)
        bm25_retriever, stemmer = load_bm25_index(cat_dir / "bm25_index")

        resources = {
            "df": df,
            "ids": ids,
            "questions_coll": questions_coll,
            "ques_def_coll": ques_def_coll,
            "bm25_retriever": bm25_retriever,
            "stemmer": stemmer,
        }
        self._loaded[catalog] = resources
        return resources


def _chroma_where(metadata_filter: dict[str, list[str]] | None) -> dict | None:
    """Build a Chroma `where` clause, restricted to columns actually stored
    as metadata (see `vector_store.METADATA_COLUMNS`) — filtering on other
    columns still works via `_allowed_doc_ids` below, just less efficiently.
    """
    if not metadata_filter:
        return None
    clauses = [
        {col: {"$in": values}} for col, values in metadata_filter.items() if col in METADATA_COLUMNS
    ]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _allowed_doc_ids(
    df: pd.DataFrame, ids: list[str], metadata_filter: dict[str, list[str]] | None
) -> set[str] | None:
    if not metadata_filter:
        return None
    mask = pd.Series(True, index=df.index)
    for col, values in metadata_filter.items():
        if col in df.columns:
            mask &= df[col].astype(str).isin(values)
    matching_positions = set(df.index[mask])
    return {ids[i] for i in matching_positions}
