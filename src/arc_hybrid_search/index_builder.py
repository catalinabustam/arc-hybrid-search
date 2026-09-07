"""End-to-end index build: download the latest ARC catalog + ARC_Lists, then
build both the "raw" and "expanded" catalogs (Chroma + BM25 each).

This only ever runs when `build_index()` is called explicitly — nothing in
this package builds anything on import, so importing `arc_hybrid_search` and
calling `HybridSearchIndex(...)` against an already-built `data_dir` never
triggers a download or a rebuild.
"""

from pathlib import Path

import pandas as pd

from .config import DEFAULT_DATA_DIR, EMBEDDING_MODEL, catalog_dir, lists_dir
from .documents import build_documents, build_ids
from .download import download_arc_catalog, download_arc_lists
from .expand import create_expanded_arc_dataframe
from .sparse_index import build_bm25_index
from .vector_store import build_collections


def build_index(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    model_name: str = EMBEDDING_MODEL,
    github_token: str | None = None,
) -> None:
    """Download the latest ARC catalog + ARC_Lists and (re)build both catalogs.

    Safe to call again later to refresh with an updated ARC catalog — Chroma
    collections are upserted and the BM25 index/CSV are simply overwritten.

    Parameters
    ----------
    data_dir : where to store the downloaded lists and both catalogs'
        Chroma/BM25/CSV artifacts. Defaults to `./arc_data`.
    model_name : sentence-transformers embedding model used for the dense
        (Chroma) side of retrieval.
    github_token : optional GitHub token (or set the `GITHUB_TOKEN` env var)
        to avoid the 60 req/hour unauthenticated rate limit when listing
        `ARC_Lists/` — see `download.download_arc_lists`.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading ARC catalog...")
    arc_df = download_arc_catalog()

    print("Downloading ARC_Lists...")
    lists_path = lists_dir(data_dir)
    download_arc_lists(lists_path, token=github_token)

    print("Building 'raw' catalog index (no expansion)...")
    _build_catalog(data_dir, "raw", arc_df, model_name)

    print("Building 'expanded' catalog index (list/option expansion)...")
    expanded_df = create_expanded_arc_dataframe(arc_df, str(lists_path))
    _build_catalog(data_dir, "expanded", expanded_df, model_name)

    print(f"Done. Index ready at {data_dir}/")


def _build_catalog(data_dir: Path, catalog: str, df: pd.DataFrame, model_name: str) -> None:
    cat_dir = catalog_dir(data_dir, catalog)
    cat_dir.mkdir(parents=True, exist_ok=True)

    df = df.fillna("")
    documents = build_documents(df)
    ids = build_ids(df)

    build_collections(cat_dir / "chroma", df, documents, ids, model_name=model_name)
    build_bm25_index(cat_dir / "bm25_index", documents)
    df.to_csv(cat_dir / f"arc_{catalog}.csv", index=False)
