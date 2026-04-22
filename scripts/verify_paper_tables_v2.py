#!/usr/bin/env python3
"""DEPRECATED: thin alias for ``verify_paper_tables.py``.

The v1 and v2 verifiers were merged into a single, data-driven verifier
in ``scripts/verify_paper_tables.py``. This file is preserved so old
shell aliases / CI invocations keep working; please update callers to
the canonical path.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import verify_paper_tables  # noqa: E402

if __name__ == "__main__":
    print("[deprecation] verify_paper_tables_v2.py -> verify_paper_tables.py",
          file=sys.stderr)
    sys.exit(verify_paper_tables.main())
