"""Tests for cython_framework.validation.regression - Performance Regression Gate.

TDD: Tests written BEFORE implementation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cython_framework.manifest import (
    BenchmarkBaseline,
    BenchmarkResult,
    Manifest,
    ModuleEntry,
    Thresholds,
    add_module,
    save,
    set_thresholds,
    update_baseline,
)
from cython_framework.validation.regression import (
    ModuleRegression,
    RegressionReport,
    check_regression,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(mean_ns: int) -> BenchmarkResult:
    return BenchmarkResult(mean_ns=mean_ns, stddev_ns=10, min_ns=mean_ns - 10, max_ns=mean_ns + 10)


def _make_baseline(results: dict[str, int]) -> BenchmarkBaseline:
    return BenchmarkBaseline(
        captured_at="2024-01-01T00:00:00+00:00",
        iterations=100,
        warmup=10,
        results={k: _make_result(v) for k, v in results.items()},
    )


def _make_thresholds(max_factor: float = 1.2) -> Thresholds:
    return Thresholds(
        max_regression_factor=max_factor,
        set_by="test",
        set_at="2024-01-01T00:00:00+00:00",
    )


def _build_manifest(tmp_path: Path, modules: dict) -> Path:
    """Build and save a manifest with given module configs.

    modules = {
        "module_key": {
            "baseline": {"impl/func": mean_ns},
            "threshold": 1.2,
            "status": "baselined",  # optional
        }
    }
    """
    manifest = Manifest(project="test_project")
    for key, config in modules.items():
        status = config.get("status", "baselined")
        entry = ModuleEntry(
            tier=1,
            source_type="pyx",
            source_path=f"{key}.pyx",
            status=status,
        )
        manifest = add_module(manifest, key, entry)

        if "baseline" in config:
            baseline = _make_baseline(config["baseline"])
            manifest = update_baseline(manifest, key, baseline)

        if "threshold" in config:
            thresholds = _make_thresholds(config["threshold"])
            manifest = set_thresholds(manifest, key, thresholds)

        if status != "baselined":
            # Need to force status since manifest transitions enforce order
            from cython_framework.manifest import update_status as _update_status

            manifest = _update_status(manifest, key, status, force=True)

    manifest_path = tmp_path / "manifest.json"
    save(manifest, manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# Tests for ModuleRegression dataclass
# ---------------------------------------------------------------------------


class TestModuleRegression:
    def test_construction(self):
        mr = ModuleRegression(
            module_key="core.engine",
            passed=True,
            baseline_ns=1000,
            current_ns=1050,
            threshold_factor=1.2,
            actual_factor=1.05,
        )
        assert mr.module_key == "core.engine"
        assert mr.passed is True
        assert mr.baseline_ns == 1000
        assert mr.current_ns == 1050
        assert mr.threshold_factor == 1.2
        assert mr.actual_factor == pytest.approx(1.05)

    def test_passes_when_within_threshold(self):
        mr = ModuleRegression(
            module_key="mod",
            passed=True,
            baseline_ns=1000,
            current_ns=1100,
            threshold_factor=1.2,
            actual_factor=1.1,
        )
        assert mr.passed is True

    def test_fails_when_exceeds_threshold(self):
        mr = ModuleRegression(
            module_key="mod",
            passed=False,
            baseline_ns=1000,
            current_ns=1500,
            threshold_factor=1.2,
            actual_factor=1.5,
        )
        assert mr.passed is False


# ---------------------------------------------------------------------------
# Tests for RegressionReport
# ---------------------------------------------------------------------------


class TestRegressionReport:
    def test_all_passed_when_all_modules_pass(self):
        mr = ModuleRegression(
            module_key="mod",
            passed=True,
            baseline_ns=1000,
            current_ns=1000,
            threshold_factor=1.2,
            actual_factor=1.0,
        )
        report = RegressionReport(modules={"mod": mr}, skipped=[])
        assert report.all_passed is True

    def test_all_passed_false_when_any_module_fails(self):
        mr = ModuleRegression(
            module_key="mod",
            passed=False,
            baseline_ns=1000,
            current_ns=2000,
            threshold_factor=1.2,
            actual_factor=2.0,
        )
        report = RegressionReport(modules={"mod": mr}, skipped=[])
        assert report.all_passed is False

    def test_all_passed_true_with_empty_modules(self):
        report = RegressionReport(modules={}, skipped=["no_baseline"])
        assert report.all_passed is True

    def test_skipped_list(self):
        report = RegressionReport(modules={}, skipped=["mod_a", "mod_b"])
        assert "mod_a" in report.skipped
        assert "mod_b" in report.skipped


# ---------------------------------------------------------------------------
# Tests for check_regression
# ---------------------------------------------------------------------------


class TestCheckRegressionWithinThreshold:
    def test_returns_regression_report(self, tmp_path):
        manifest_path = _build_manifest(
            tmp_path,
            {"core.engine": {"baseline": {"Augmented Python/add": 1000}, "threshold": 1.2}},
        )
        current = {"core.engine": {"Augmented Python/add": 1050}}
        report = check_regression(manifest_path, current_results=current)
        assert isinstance(report, RegressionReport)

    def test_passes_when_within_threshold(self, tmp_path):
        manifest_path = _build_manifest(
            tmp_path,
            {"core.engine": {"baseline": {"Augmented Python/add": 1000}, "threshold": 1.2}},
        )
        current = {"core.engine": {"Augmented Python/add": 1100}}  # 1.1x < 1.2x threshold
        report = check_regression(manifest_path, current_results=current)
        assert report.all_passed is True
        assert report.modules["core.engine"].passed is True

    def test_fails_when_exceeds_threshold(self, tmp_path):
        manifest_path = _build_manifest(
            tmp_path,
            {"core.engine": {"baseline": {"Augmented Python/add": 1000}, "threshold": 1.2}},
        )
        current = {"core.engine": {"Augmented Python/add": 1500}}  # 1.5x > 1.2x threshold
        report = check_regression(manifest_path, current_results=current)
        assert report.all_passed is False
        assert report.modules["core.engine"].passed is False

    def test_actual_factor_computed_correctly(self, tmp_path):
        manifest_path = _build_manifest(
            tmp_path,
            {"mod": {"baseline": {"impl/func": 1000}, "threshold": 2.0}},
        )
        current = {"mod": {"impl/func": 1500}}
        report = check_regression(manifest_path, current_results=current)
        assert report.modules["mod"].actual_factor == pytest.approx(1.5)

    def test_baseline_ns_is_max_of_baseline_means(self, tmp_path):
        """baseline_ns uses the maximum mean across all matched impl/func keys."""
        manifest_path = _build_manifest(
            tmp_path,
            {
                "mod": {
                    "baseline": {"impl/add": 1000, "impl/multiply": 2000},
                    "threshold": 1.2,
                }
            },
        )
        current = {"mod": {"impl/add": 1050, "impl/multiply": 2100}}
        report = check_regression(manifest_path, current_results=current)
        assert "mod" in report.modules


class TestCheckRegressionSkipping:
    def test_skips_module_without_baseline(self, tmp_path):
        manifest_path = _build_manifest(
            tmp_path,
            {"no_baseline_mod": {"threshold": 1.2, "status": "tests_generated"}},
        )
        report = check_regression(manifest_path, current_results={})
        assert "no_baseline_mod" in report.skipped
        assert "no_baseline_mod" not in report.modules

    def test_skips_module_without_thresholds(self, tmp_path):
        manifest_path = _build_manifest(
            tmp_path,
            {"no_threshold_mod": {"baseline": {"impl/add": 1000}}},
        )
        report = check_regression(manifest_path, current_results={})
        assert "no_threshold_mod" in report.skipped

    def test_skips_module_not_in_current_results(self, tmp_path):
        manifest_path = _build_manifest(
            tmp_path,
            {"mod": {"baseline": {"impl/add": 1000}, "threshold": 1.2}},
        )
        report = check_regression(manifest_path, current_results={})
        assert "mod" in report.skipped

    def test_multiple_modules_mixed_pass_fail(self, tmp_path):
        manifest_path = _build_manifest(
            tmp_path,
            {
                "mod_a": {"baseline": {"impl/add": 1000}, "threshold": 1.2},
                "mod_b": {"baseline": {"impl/add": 1000}, "threshold": 1.2},
            },
        )
        current = {
            "mod_a": {"impl/add": 1050},  # passes
            "mod_b": {"impl/add": 2000},  # fails
        }
        report = check_regression(manifest_path, current_results=current)
        assert report.all_passed is False
        assert report.modules["mod_a"].passed is True
        assert report.modules["mod_b"].passed is False
