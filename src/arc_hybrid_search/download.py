"""Downloading the ARC catalog and its ARC_Lists folder from GitHub.

Both are fetched fresh every time `download_arc_catalog` / `download_arc_lists`
run — there's no caching here, since the whole point of `build_index()` is to
pick up the latest published ARC catalog on demand.
"""
import io
import os
from pathlib import Path

import pandas as pd
import requests

from .config import ARC_BRANCH, ARC_REPO, ARC_URL

# GitHub's git/trees API is rate-limited to 60 requests/hour per IP when
# unauthenticated, which can be exhausted just from other traffic sharing
# that IP (e.g. shared CI runners). Passing a token raises it to 5000/hour.
# `download_arc_catalog` doesn't need this — it hits raw.githubusercontent.com,
# which isn't subject to the same limit.
GITHUB_TOKEN_ENV_VAR = "GITHUB_TOKEN"


def _github_headers(token: str | None) -> dict[str, str]:
    token = token or os.environ.get(GITHUB_TOKEN_ENV_VAR)
    return {"Authorization": f"Bearer {token}"} if token else {}


def download_arc_catalog(url: str = ARC_URL) -> pd.DataFrame:
    """Download and parse the main `ARC.csv` catalog."""
    response = requests.get(url, timeout=(3.05, 10))
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def download_arc_lists(
    dest_dir: Path | str, repo: str = ARC_REPO, branch: str = ARC_BRANCH, token: str | None = None
) -> None:
    """Recursively download every CSV under `ARC_Lists/` in the ARC repo.

    Uses the Git Trees API (`recursive=1`) to enumerate the whole repo tree in
    one call, then pulls each `Lists/...csv` file via the raw content CDN.
    The local subfolder structure under `dest_dir` mirrors `Lists/` in the
    repo, which is what `expand.create_expanded_arc_dataframe` expects.

    Parameters
    ----------
    token : optional GitHub personal access token (or set the `GITHUB_TOKEN`
        env var) to avoid the 60 req/hour unauthenticated rate limit on the
        Trees API call above. Not required for the raw file downloads.
    """
    dest_dir = Path(dest_dir)
    headers = _github_headers(token)
    tree_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
    response = requests.get(tree_url, headers=headers, timeout=(3.05, 10))
    response.raise_for_status()
    tree = response.json().get("tree", [])

    csv_paths = [
        entry["path"]
        for entry in tree
        if entry.get("type") == "blob"
        and entry["path"].startswith("Lists/")
        and entry["path"].endswith(".csv")
    ]
    if not csv_paths:
        raise RuntimeError("No CSV files found under Lists/ in the ARC repo.")

    raw_base = f"https://raw.githubusercontent.com/{repo}/{branch}/"
    for path in csv_paths:
        file_response = requests.get(raw_base + path, timeout=(3.05, 10))
        file_response.raise_for_status()
        local_path = dest_dir / Path(path).relative_to("Lists")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(file_response.content)
