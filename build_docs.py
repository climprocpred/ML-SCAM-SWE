#!/usr/bin/env python3
"""Build pdoc HTML documentation for ML-SCAM-SWE.

Adds the swe/ directory to sys.path so that the flat-module bare imports
used in the source files resolve correctly, then runs pdoc over every
Python module in swe/.

Usage::

    python build_docs.py

Output is written to docs/.
"""

import pathlib
import sys

# Add swe/ to sys.path so bare imports (e.g. ``from NNorm_GAM2 import ...``)
# resolve correctly without converting the directory into a package.
ROOT = pathlib.Path(__file__).parent
swe_dir = ROOT / "swe"
sys.path.insert(0, str(swe_dir))

import pdoc

modules = sorted(
    f.stem
    for f in swe_dir.glob("*.py")
    if not f.stem.startswith("_")
)

output_dir = ROOT / "docs"
output_dir.mkdir(exist_ok=True)

pdoc.pdoc(*modules, output_directory=output_dir)
print(f"Docs written to {output_dir}/")
