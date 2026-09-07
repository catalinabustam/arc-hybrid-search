# arc-hybrid-search

Hybrid (dense + BM25) search over the [ISARIC ARC](https://github.com/ISARICResearch/ARC)
reference catalog, packaged for reuse across projects.

Builds **two independent catalogs**, each with its own ChromaDB collections
and BM25 index:

- **`raw`** — the ARC catalog exactly as published (`ARC.csv`), untouched.
- **`expanded`** — `user_list`/`multilist` rows exploded into one row per
  list item, and `radio`/`checkbox` rows with their cleaned options appended
  to the question text (so retrieval can match individual list items or
  options directly).

Nothing is downloaded or built on import — only when you call `build_index()`.

## Installation

From this folder:

```bash
pip install -e .
```

Or build a wheel and install it elsewhere:

```bash
pip install build
python -m build
pip install dist/arc_hybrid_search-0.1.0-py3-none-any.whl
```

Requires Python 3.12+. Building the index needs network access to
`api.github.com` and `raw.githubusercontent.com`; retrieval afterwards is
fully local.

## Usage

### 1. Build the index (one-time, or whenever you want the latest ARC catalog)

```python
from arc_hybrid_search import build_index

build_index(data_dir="./arc_data")  # downloads ARC.csv + ARC_Lists, builds both catalogs
```

GitHub's Trees API (used to enumerate `ARC_Lists/`) is rate-limited to 60
unauthenticated requests/hour **per IP**, which can be exhausted just from
other traffic sharing that IP (shared CI runners, corporate NAT, etc.). If
you hit `403 API rate limit exceeded`, pass a
[GitHub personal access token](https://github.com/settings/tokens) (no
special scopes needed for public repos) via `github_token=...` or the
`GITHUB_TOKEN` env var — that raises the limit to 5000/hour:

```python
build_index(data_dir="./arc_data", github_token="ghp_...")
```

This downloads the latest `ARC.csv` and `ARC_Lists/`, then writes:

```
arc_data/
├── ARC_Lists/
├── raw/
│   ├── arc_raw.csv
│   ├── chroma/
│   └── bm25_index/
└── expanded/
    ├── arc_expanded.csv
    ├── chroma/
    └── bm25_index/
```

Safe to re-run later to refresh with an updated ARC catalog.

### 2. Retrieve

```python
from arc_hybrid_search import HybridSearchIndex

index = HybridSearchIndex(data_dir="./arc_data")

results = index.retrieve(
    query="What is the patient's age at admission?",
    catalog="expanded",  # or "raw"
    top_k=5,
    metadata_filter={"Form": ["Demographics"]},  # optional
)

for r in results:
    print(f"{r['score']:.0%}  {r['variable']:<25}  {r['question']}")
```

Each result is a dict:

```python
{
    "row_index": 12,
    "question": "What is the patient's age at admission?",
    "definition": "Age in completed years at time of admission",
    "section": "Demographics",
    "form": "Demographics",
    "variable": "demog_age",
    "score": 0.87,
}
```

`metadata_filter` columns are AND-combined; values within a column are
OR-combined (e.g. `{"Form": ["Demographics"], "Section": ["Vitals"]}`).

### Available filter columns

Any column present in the ARC catalog can be used as a `metadata_filter`
key, but they fall into two tiers:

**Indexed in Chroma (fast — filters the dense search itself):**

| Column     | Example values                          |
|------------|------------------------------------------|
| `Form`     | `"Demographics"`, `"Vitals"`, `"Outcome"` |
| `Section`  | `"Admission"`, `"Comorbidities"`          |
| `Question` | exact question text                       |

**Any other ARC column (works, applied after retrieval — less efficient
since the dense/BM25 search itself still runs unfiltered first):**

| Column                                    | Example values                              |
|--------------------------------------------|----------------------------------------------|
| `Variable`                                 | `"demog_age"`                                |
| `Type`                                     | `"radio"`, `"text"`, `"user_list"`           |
| `Body System`                              | `"Respiratory"`, `"Cardiovascular"`          |
| `Research Category`                        | `"Core"`, `"Optional"`                       |
| `Identifier`                                | `"y"`                                        |
| `preset_ARChetype Disease CRF_Covid`       | preset flag columns, one per ARC preset      |
| `preset_ARChetype Syndromic CRF_ARI`       | (see full list in `ARC.csv`)                 |
| `preset_Score_mSOFA`, `preset_Populations_Paediatric`, etc. | |

```python
# Fast: Form/Section are Chroma-indexed
results = index.retrieve("cough duration", metadata_filter={"Form": ["Signs and Symptoms"]})

# Works, but post-filtered: restrict to a specific ARC preset
results = index.retrieve(
    "fever onset",
    metadata_filter={"preset_ARChetype Disease CRF_Dengue": ["1"]},
)
```

If you find yourself filtering on a non-indexed column often, it's worth
adding it to `vector_store.METADATA_COLUMNS` so Chroma can filter on it
directly — just remember to run `build_index()` again afterwards so the
collections get rebuilt with that column in their metadata.

`HybridSearchIndex` loads each catalog's Chroma collections and BM25 index
lazily, the first time you request it — creating one instance and calling
`retrieve()` many times (with either `catalog` value) is the intended usage;
there's no need to build a new instance per call.

## Notes

- The embedding model defaults to `BAAI/bge-large-en-v1.5` (same as the
  original app); override via `build_index(model_name=...)` and
  `HybridSearchIndex(model_name=...)` — both sides must agree, since Chroma
  needs the same embedding function to query a collection it built.
- This package intentionally does **not** include translation, REDCap
  export, or UI concerns — it's scoped to hybrid retrieval only, so it can
  be reused by any project that needs "find the closest ARC variable(s) to
  this question."

## Development

```bash
pip install -e ".[dev]"  # if you add a [project.optional-dependencies] dev group
ruff check .
ruff format .
mypy src/
pytest tests/
```

`tests/test_smoke.py` only covers the pure-function pieces (no network, no
embedding model download). Exercising `build_index()` + `retrieve()`
end-to-end requires network access and is meant to be run manually or in a
CI job with network enabled.
