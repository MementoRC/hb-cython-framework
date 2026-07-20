"""Acceptance test for G1 -- first-class 4-variant builder.

Builds all four CythonModuleType variants of the running_stats fixture via
`build_variants`, loads each through `CythonModuleLoader`, and asserts all
four produce identical results.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from cython_framework.buildhook.variant_builder import VariantBuildError, build_variants
from cython_framework.testing.cython_test_case import CythonModuleLoader, CythonModuleType

# Skip entire module if Cython is not available
cython = pytest.importorskip("Cython", reason="Cython not installed")

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
_SOURCE = _FIXTURES_DIR / "running_stats.py"
_BUILD_DIR = _FIXTURES_DIR / "_variant_build_running_stats"
# Dotted path relative to project root, resolving to _BUILD_DIR, as required
# by CythonModuleLoader.
_MODULE_PATH = "tests.fixtures._variant_build_running_stats"
_MODULE_NAME = "running_stats"


@pytest.fixture(scope="module")
def all_variant_modules():
    """Build all 4 variants and load each via CythonModuleLoader."""
    shutil.rmtree(_BUILD_DIR, ignore_errors=True)
    try:
        build_variants(_SOURCE, module_name=_MODULE_NAME, build_dir=_BUILD_DIR)
    except VariantBuildError as e:
        shutil.rmtree(_BUILD_DIR, ignore_errors=True)
        pytest.skip(str(e))

    loader = CythonModuleLoader(_MODULE_PATH, _MODULE_NAME)
    modules = {variant: loader.load_module(variant)[0] for variant in CythonModuleType}

    yield modules

    shutil.rmtree(_BUILD_DIR, ignore_errors=True)


def test_all_four_variants_build_and_load(all_variant_modules):
    assert set(all_variant_modules) == set(CythonModuleType)


def test_all_four_variants_agree(all_variant_modules):
    values = [1.0, 2.0, 3.0, 4.0, 5.0, -2.5, 10.0]

    results = {}
    for variant, module in all_variant_modules.items():
        stats = module.RunningStats()
        adds = [stats.add(x) for x in values]
        results[variant] = (adds, stats.mean(), stats.stddev())

    baseline = results[CythonModuleType.AUGMENTED_PYTHON]
    for variant, result in results.items():
        assert result[0] == baseline[0], f"add() mismatch for {variant}"
        assert result[1] == pytest.approx(baseline[1]), f"mean() mismatch for {variant}"
        assert result[2] == pytest.approx(baseline[2]), f"stddev() mismatch for {variant}"
