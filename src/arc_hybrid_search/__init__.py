"""arc_hybrid_search — hybrid (dense + BM25) search over the ISARIC ARC
reference catalog, with both a "raw" (untouched) and an "expanded"
(list/option-expanded) variant.

    from arc_hybrid_search import build_index, HybridSearchIndex

    build_index(data_dir="./arc_data")          # explicit, one-time (or refresh) step

    index = HybridSearchIndex(data_dir="./arc_data")
    results = index.retrieve("patient's age at admission", catalog="expanded", top_k=5)
"""

from .index_builder import build_index
from .search import HybridSearchIndex

__all__ = ["HybridSearchIndex", "build_index"]
