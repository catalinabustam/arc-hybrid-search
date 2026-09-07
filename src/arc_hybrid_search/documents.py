"""Build the embedded document text and shared doc ids for a catalog dataframe.

Pure functions of the catalog dataframe alone, so both the build step and the
retrieval step derive `documents`/`ids` the same way from the persisted CSV
and are guaranteed to agree without storing them separately.
"""

import pandas as pd

_TEXT_COLUMNS = ["Question", "Definition"]


def build_documents(df: pd.DataFrame) -> list[str]:
    """ "Question. Definition. " text embedded/indexed for each row."""
    columns = [col for col in _TEXT_COLUMNS if col in df.columns]
    return [" ".join(f"{value}. " for _, value in row[columns].items()) for _, row in df.iterrows()]


def build_ids(df: pd.DataFrame) -> list[str]:
    """Doc ids shared by the Chroma collections and the BM25 index."""
    return [f"row_{i}" for i in range(len(df))]
