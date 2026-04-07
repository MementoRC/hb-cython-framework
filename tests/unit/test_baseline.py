"""Tests for cython_framework.validation.baseline - Performance Baseline Capture.

TDD: Tests written BEFORE implementation.
"""

from __future__ import annotations

from pathlib import Path

from cython_framework.manifest import (
    BenchmarkBaseline,
    Manifest,
    ModuleEntry,
    add_module,
    load,
    save,
)
from cython_framework.testing.cython_test_case import CythonModuleType
from cython_framework.validation.baseline import capture_baseline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODULE_PATH = "tests.fixtures.augmented_example"
MODULE_NAME = "example"
IMPLEMENTATIONS = [CythonModuleType.AUGMENTED_PYTHON]
FUNC_NAMES = ["add", "multiply", "fibonacci"]
FUNC_ARGS = {"add": (1, 2), "multiply": (2.0, 3.0), "fibonacci": (10,)}


def _make_manifest_with_module(
    tmp_path: Path, module_key: str, status: str = "tests_generated"
) -> Path:
    """Create a manifest JSON with a single module entry."""
    manifest = Manifest(project="test_project")
    entry = ModuleEntry(
        tier=1,
        source_type="pyx",
        source_path="tests/fixtures/augmented_example/example.pyx",
        status=status,
    )
    manifest = add_module(manifest, module_key, entry)
    manifest_path = tmp_path / "manifest.json"
    save(manifest, manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# Tests for capture_baseline
# ---------------------------------------------------------------------------


class TestCaptureBaselineReturn:
    """capture_baseline returns a BenchmarkBaseline with correct structure."""

    def test_returns_benchmark_baseline_instance(self):
        result = capture_baseline(
            module_path=MODULE_PATH,
            module_name=MODULE_NAME,
            implementations=IMPLEMENTATIONS,
            func_names=FUNC_NAMES,
            func_args=FUNC_ARGS,
            iterations=10,
            warmup=2,
        )
        assert isinstance(result, BenchmarkBaseline)

    def test_baseline_has_correct_iterations(self):
        result = capture_baseline(
            module_path=MODULE_PATH,
            module_name=MODULE_NAME,
            implementations=IMPLEMENTATIONS,
            func_names=FUNC_NAMES,
            func_args=FUNC_ARGS,
            iterations=10,
            warmup=2,
        )
        assert result.iterations == 10

    def test_baseline_has_correct_warmup(self):
        result = capture_baseline(
            module_path=MODULE_PATH,
            module_name=MODULE_NAME,
            implementations=IMPLEMENTATIONS,
            func_names=FUNC_NAMES,
            func_args=FUNC_ARGS,
            iterations=10,
            warmup=2,
        )
        assert result.warmup == 2

    def test_baseline_has_captured_at_timestamp(self):
        result = capture_baseline(
            module_path=MODULE_PATH,
            module_name=MODULE_NAME,
            implementations=IMPLEMENTATIONS,
            func_names=FUNC_NAMES,
            func_args=FUNC_ARGS,
            iterations=10,
            warmup=2,
        )
        assert isinstance(result.captured_at, str)
        assert len(result.captured_at) > 0

    def test_baseline_results_contains_implementations(self):
        result = capture_baseline(
            module_path=MODULE_PATH,
            module_name=MODULE_NAME,
            implementations=IMPLEMENTATIONS,
            func_names=FUNC_NAMES,
            func_args=FUNC_ARGS,
            iterations=10,
            warmup=2,
        )
        # Results keys should include implementation name(s) and function names
        assert len(result.results) > 0

    def test_baseline_results_has_numeric_stats(self):
        from cython_framework.manifest import BenchmarkResult as ManifestBenchmarkResult

        result = capture_baseline(
            module_path=MODULE_PATH,
            module_name=MODULE_NAME,
            implementations=IMPLEMENTATIONS,
            func_names=FUNC_NAMES,
            func_args=FUNC_ARGS,
            iterations=10,
            warmup=2,
        )
        for _key, bench_result in result.results.items():
            assert isinstance(bench_result, ManifestBenchmarkResult)
            assert bench_result.mean_ns > 0
            assert bench_result.min_ns > 0
            assert bench_result.max_ns >= bench_result.min_ns

    def test_baseline_result_keys_include_func_names(self):
        result = capture_baseline(
            module_path=MODULE_PATH,
            module_name=MODULE_NAME,
            implementations=IMPLEMENTATIONS,
            func_names=["add", "multiply"],
            func_args={"add": (1, 2), "multiply": (2.0, 3.0)},
            iterations=10,
            warmup=2,
        )
        keys = list(result.results.keys())
        # At least one key should contain "add" and one "multiply"
        assert any("add" in k for k in keys)
        assert any("multiply" in k for k in keys)


class TestCaptureBaselineManifestIntegration:
    """capture_baseline can update a manifest file."""

    def test_stores_baseline_in_manifest(self, tmp_path):
        module_key = "example"
        manifest_path = _make_manifest_with_module(tmp_path, module_key)

        capture_baseline(
            module_path=MODULE_PATH,
            module_name=MODULE_NAME,
            implementations=IMPLEMENTATIONS,
            func_names=FUNC_NAMES,
            func_args=FUNC_ARGS,
            iterations=10,
            warmup=2,
            manifest_path=manifest_path,
            module_key=module_key,
        )

        reloaded = load(manifest_path)
        entry = reloaded.modules[module_key]
        assert entry.benchmark_baseline is not None
        assert isinstance(entry.benchmark_baseline, BenchmarkBaseline)

    def test_transitions_status_to_baselined(self, tmp_path):
        module_key = "example"
        manifest_path = _make_manifest_with_module(tmp_path, module_key, status="tests_generated")

        capture_baseline(
            module_path=MODULE_PATH,
            module_name=MODULE_NAME,
            implementations=IMPLEMENTATIONS,
            func_names=FUNC_NAMES,
            func_args=FUNC_ARGS,
            iterations=10,
            warmup=2,
            manifest_path=manifest_path,
            module_key=module_key,
        )

        reloaded = load(manifest_path)
        entry = reloaded.modules[module_key]
        assert entry.status == "baselined"

    def test_no_manifest_update_when_no_path(self, tmp_path):
        """Without manifest_path, just returns BenchmarkBaseline without side effects."""
        result = capture_baseline(
            module_path=MODULE_PATH,
            module_name=MODULE_NAME,
            implementations=IMPLEMENTATIONS,
            func_names=FUNC_NAMES,
            func_args=FUNC_ARGS,
            iterations=10,
            warmup=2,
            manifest_path=None,
            module_key=None,
        )
        assert isinstance(result, BenchmarkBaseline)
