"""ChromaDB collection management for one catalog's Chroma directory.

Each catalog ("raw" or "expanded") gets its own persistent Chroma client at
`<catalog_dir>/chroma`, holding two collections: one indexed on the question
text alone, one on question + definition (+ options). Retrieval queries both
and keeps the max score per document — see `fusion.joint_dense_retrieve`.
"""

from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions

from .config import EMBEDDING_MODEL

QUESTIONS_COLLECTION = "questions"
QUES_DEF_COLLECTION = "ques_def"
METADATA_COLUMNS = ("Form", "Section", "Question")


def _metadatas(df: pd.DataFrame) -> list[dict]:
    return [
        {col: row[col] for col in METADATA_COLUMNS if col in row.index} for _, row in df.iterrows()
    ]


def build_collections(
    chroma_path: Path,
    df: pd.DataFrame,
    documents: list[str],
    ids: list[str],
    model_name: str = EMBEDDING_MODEL,
) -> None:
    """(Re)build the two Chroma collections for one catalog."""
    client = chromadb.PersistentClient(path=str(chroma_path))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    metadatas = _metadatas(df)

    ques_def = client.get_or_create_collection(
        name=QUES_DEF_COLLECTION,
        embedding_function=embedding_fn,  # type: ignore[arg-type]  # chromadb stub generic mismatch
        metadata={"hnsw:space": "cosine"},
    )
    ques_def.upsert(ids=ids, documents=documents, metadatas=metadatas)  # type: ignore[arg-type]

    questions = client.get_or_create_collection(
        name=QUESTIONS_COLLECTION,
        embedding_function=embedding_fn,  # type: ignore[arg-type]  # chromadb stub generic mismatch
        metadata={"hnsw:space": "cosine"},
    )
    questions.upsert(
        ids=ids,
        documents=df["Question"].tolist(),
        metadatas=metadatas,  # type: ignore[arg-type]
    )


def load_collections(chroma_path: Path, model_name: str = EMBEDDING_MODEL):
    """Load the two Chroma collections previously built by `build_collections`."""
    client = chromadb.PersistentClient(path=str(chroma_path))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    try:
        questions = client.get_collection(
            name=QUESTIONS_COLLECTION,
            embedding_function=embedding_fn,  # type: ignore[arg-type]
        )
        ques_def = client.get_collection(
            name=QUES_DEF_COLLECTION,
            embedding_function=embedding_fn,  # type: ignore[arg-type]
        )
    except Exception as exc:
        raise RuntimeError(
            f"Chroma collections not found at {chroma_path}. Run build_index() first."
        ) from exc
    return questions, ques_def
