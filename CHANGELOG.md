# Changelog

All notable changes to this project are documented here.
This project follows [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-09-07

### Added
- `build_index()` — downloads the latest ARC catalog and `ARC_Lists`, builds
  both a `raw` and an `expanded` (list/option-expanded) catalog, each with
  its own ChromaDB collections and BM25 index.
- `HybridSearchIndex.retrieve()` — hybrid (dense + BM25, RRF-fused) search
  over either catalog, with optional metadata filtering.
- Optional `github_token` support for `build_index()` to avoid GitHub's
  unauthenticated API rate limit when downloading `ARC_Lists`.
