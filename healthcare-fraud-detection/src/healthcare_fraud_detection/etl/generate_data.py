#!/usr/bin/env python3
"""
Generate synthetic patients, providers, and claims into ``data/raw/``.

Intended to be run from the project root, e.g.::

    python src/healthcare_fraud_detection/etl/generate_data.py

Path setup below allows imports of ``synthetic_healthcare`` (repo root) and
``healthcare_fraud_detection`` (``src/``) when the script is invoked by path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_paths() -> Path:
    here = Path(__file__).resolve()
    project_root = here.parents[3]
    src_root = here.parents[2]
    for p in (str(project_root), str(src_root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return project_root


def main() -> None:
    project_root = _bootstrap_paths()
    import generate_synthetic_healthcare_data as runner

    prev = Path.cwd()
    try:
        os.chdir(project_root)
        runner.main()
    finally:
        os.chdir(prev)


if __name__ == "__main__":
    main()
