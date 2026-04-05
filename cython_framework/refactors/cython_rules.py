"""Cython augmented pure Python conversion rules.

Usage:
    python cython_framework/refactors/cython_rules.py <src_path> --apply

Rules applied (in order):
    1. ConvertTypeToCython - int -> cython.int, float -> cython.double
    2. ConvertFunctionAnnotations - adds @cython.ccall, @cython.returns()
    3. AddCythonMarkerDecorator - # cython: cfunc -> @cython.cfunc
    4. DeclareClassAttributes - auto cython.declare() for @cython.cclass
    5. AddCythonImport - inserts import cython when needed
"""

from refactor import run
from refactor.rules.cython_augmented import (
    AddCythonImport,
    AddCythonMarkerDecorator,
    ConvertFunctionAnnotations,
    ConvertTypeToCython,
    DeclareClassAttributes,
)

ALL_RULES = [
    ConvertTypeToCython,
    ConvertFunctionAnnotations,
    AddCythonMarkerDecorator,
    DeclareClassAttributes,
    AddCythonImport,
]

if __name__ == "__main__":
    raise SystemExit(run(ALL_RULES))
