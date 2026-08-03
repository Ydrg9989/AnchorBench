"""Post-processing analyses that aggregate raw results into the
unified JSON files consumed by ``anchorbench.paper`` artifact generators.

Modules:
    unified     -- recompute ``unified_all_suites.json`` for a results tree
    gold_shift  -- gold-shift decomposition (Appendix Table 11)
    sampling    -- sampling-robustness aggregation + figures
    mitigation  -- mitigation headroom aggregator (Table 12 / Fig 6)
"""

from __future__ import annotations
