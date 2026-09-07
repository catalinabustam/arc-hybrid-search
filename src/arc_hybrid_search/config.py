"""Shared configuration and default paths for `arc_hybrid_search`."""
from pathlib import Path

ARC_REPO = "ISARICResearch/ARC"
ARC_BRANCH = "main"
ARC_URL = f"https://raw.githubusercontent.com/{ARC_REPO}/refs/heads/{ARC_BRANCH}/ARC.csv"

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_DATA_DIR = Path("arc_data")

# The two independent catalogs this package builds and searches.
CATALOGS = ("raw", "expanded")


def catalog_dir(data_dir: Path | str, catalog: str) -> Path:
    """Return the on-disk directory for one catalog ("raw" or "expanded")."""
    if catalog not in CATALOGS:
        raise ValueError(f"catalog must be one of {CATALOGS}, got {catalog!r}")
    return Path(data_dir) / catalog


def lists_dir(data_dir: Path | str) -> Path:
    """Return the directory where the downloaded ARC_Lists CSVs are stored."""
    return Path(data_dir) / "Lists"
