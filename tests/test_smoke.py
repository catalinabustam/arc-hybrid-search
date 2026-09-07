"""Smoke tests: package imports, and the pure-function pieces that don't
require network access or downloading the embedding model. Full build/
retrieve coverage needs `build_index()` to have run against a real `data_dir`
and is exercised manually/in CI with network access — see README.md.
"""

import pandas as pd
import pytest

from arc_hybrid_search import HybridSearchIndex, build_index
from arc_hybrid_search.config import CATALOGS, catalog_dir, lists_dir
from arc_hybrid_search.documents import build_documents, build_ids


def test_public_api_importable():
    assert callable(build_index)
    assert callable(HybridSearchIndex)


def test_catalog_dir_rejects_unknown_catalog():
    with pytest.raises(ValueError):
        catalog_dir("data", "not_a_real_catalog")


def test_catalog_dir_and_lists_dir(tmp_path):
    assert catalog_dir(tmp_path, "raw") == tmp_path / "raw"
    assert catalog_dir(tmp_path, "expanded") == tmp_path / "expanded"
    assert lists_dir(tmp_path) == tmp_path / "Lists"
    assert set(CATALOGS) == {"raw", "expanded"}


def test_build_documents_and_ids():
    df = pd.DataFrame(
        {
            "Question": ["What is your age?", "What is your sex?"],
            "Definition": ["Age in years", ""],
        }
    )
    documents = build_documents(df)
    ids = build_ids(df)

    assert documents == [
        "What is your age?.  Age in years. ",
        "What is your sex?.  . ",
    ]
    assert ids == ["row_0", "row_1"]


def test_retrieve_before_build_raises(tmp_path):
    index = HybridSearchIndex(data_dir=tmp_path)
    with pytest.raises(RuntimeError, match="Run build_index"):
        index.retrieve("anything")
