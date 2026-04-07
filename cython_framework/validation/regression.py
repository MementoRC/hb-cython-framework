"""Performance regression gate for Cython migration modules.

Compares current benchmark results against stored baselines in the manifest
and reports pass/fail per module according to configured thresholds.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from cython_framework.manifest import load


@dataclasses.dataclass
class ModuleRegression:
    """Regression result for a single module."""

    module_key: str
    passed: bool
    baseline_ns: int
    current_ns: int
    threshold_factor: float
    actual_factor: float


@dataclasses.dataclass
class RegressionReport:
    """Aggregated regression report across all checked modules."""

    modules: dict[str, ModuleRegression]
    skipped: list[str]

    @property
    def all_passed(self) -> bool:
        """True if every checked module passed (skipped modules are ignored)."""
        return all(m.passed for m in self.modules.values())


def check_regression(
    manifest_path: str | Path,
    current_results: dict[str, dict[str, int]] | None = None,
) -> RegressionReport:
    """Check performance regression for all modules in the manifest.

    For each module that has both a ``benchmark_baseline`` and ``thresholds``
    set, compares the *current_results* against the baseline.  The comparison
    takes the **maximum** mean_ns across all matching ``impl/func`` keys
    (baseline side) versus the maximum mean_ns in *current_results* for that
    module, then checks whether ``current / baseline <= max_regression_factor``.

    Modules are **skipped** (added to :attr:`RegressionReport.skipped`) when:
    - No ``benchmark_baseline`` is set on the module entry.
    - No ``thresholds`` are set on the module entry.
    - The module key is absent from *current_results*.

    Args:
        manifest_path: Path to the manifest JSON file.
        current_results: Mapping of ``{module_key: {impl_name: mean_ns}}``.
            If *None*, all modules with baselines and thresholds are skipped.

    Returns:
        A :class:`RegressionReport` with per-module results and a list of
        skipped module keys.
    """
    if current_results is None:
        current_results = {}

    manifest = load(manifest_path)
    modules: dict[str, ModuleRegression] = {}
    skipped: list[str] = []

    for key, entry in manifest.modules.items():
        # Skip modules without baseline or thresholds
        if entry.benchmark_baseline is None or entry.thresholds is None:
            skipped.append(key)
            continue

        # Skip if not in current results
        if key not in current_results:
            skipped.append(key)
            continue

        baseline = entry.benchmark_baseline
        thresholds = entry.thresholds
        current_impl = current_results[key]

        # Compute representative baseline_ns: max mean across all stored results
        if baseline.results:
            baseline_ns = max(r.mean_ns for r in baseline.results.values())
        else:
            skipped.append(key)
            continue

        # Compute current_ns: max mean across current results for this module
        if current_impl:
            current_ns = max(current_impl.values())
        else:
            skipped.append(key)
            continue

        actual_factor = current_ns / baseline_ns if baseline_ns > 0 else float("inf")
        passed = actual_factor <= thresholds.max_regression_factor

        modules[key] = ModuleRegression(
            module_key=key,
            passed=passed,
            baseline_ns=baseline_ns,
            current_ns=current_ns,
            threshold_factor=thresholds.max_regression_factor,
            actual_factor=actual_factor,
        )

    return RegressionReport(modules=modules, skipped=skipped)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main() -> None:  # pragma: no cover
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Check performance regression against manifest baselines."
    )
    parser.add_argument("manifest_path", help="Path to the manifest JSON file")
    parser.add_argument(
        "--current-results",
        type=json.loads,
        default=None,
        help=(
            "JSON mapping of {module_key: {impl/func: mean_ns}}, "
            'e.g. \'{"core.engine": {"Augmented Python/add": 1050}}\''
        ),
    )

    args = parser.parse_args()

    report = check_regression(
        manifest_path=args.manifest_path,
        current_results=args.current_results,
    )

    print(f"Regression report: {'PASSED' if report.all_passed else 'FAILED'}")
    for key, result in report.modules.items():
        status = "PASS" if result.passed else "FAIL"
        print(
            f"  [{status}] {key}: "
            f"baseline={result.baseline_ns}ns current={result.current_ns}ns "
            f"factor={result.actual_factor:.3f} (threshold={result.threshold_factor:.3f})"
        )
    if report.skipped:
        print(f"  Skipped: {', '.join(report.skipped)}")

    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    _main()
