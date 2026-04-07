"""Performance baseline capture for Cython migration modules.

Benchmarks multiple implementations of a module using nanosecond precision
timers, and optionally stores results in a migration manifest.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from cython_framework.manifest import (
    BenchmarkBaseline,
    BenchmarkResult,
    load,
    save,
    update_baseline,
    update_status,
)
from cython_framework.testing.cython_test_case import CythonModuleLoader, CythonModuleType


def capture_baseline(
    module_path: str,
    module_name: str,
    implementations: Sequence[CythonModuleType],
    func_names: Sequence[str],
    func_args: dict[str, tuple],
    iterations: int = 1000,
    warmup: int = 100,
    manifest_path: str | Path | None = None,
    module_key: str | None = None,
) -> BenchmarkBaseline:
    """Capture a performance baseline for a module across multiple implementations.

    For each implementation x function pair, runs *warmup* calls (not timed),
    then *iterations* timed calls using ``time.perf_counter_ns()``.
    Computes mean, stddev, min, and max in nanoseconds.

    If *manifest_path* and *module_key* are both provided, the baseline is
    stored in the manifest and the module status is transitioned to
    ``"baselined"``.

    Args:
        module_path: Dotted import path to the module package
            (e.g. ``"tests.fixtures.augmented_example"``).
        module_name: Module file name without extension (e.g. ``"example"``).
        implementations: Sequence of :class:`CythonModuleType` variants to benchmark.
        func_names: Names of functions to benchmark.
        func_args: Mapping of function name -> positional args tuple.
        iterations: Number of timed iterations per function.
        warmup: Number of warmup (un-timed) calls before timing.
        manifest_path: Optional path to the manifest JSON file.
        module_key: Optional module key in the manifest to update.

    Returns:
        A :class:`BenchmarkBaseline` with results keyed as ``"<impl_name>/<func_name>"``.
    """
    loader = CythonModuleLoader(module_path, module_name)

    results: dict[str, BenchmarkResult] = {}
    captured_at = datetime.now(tz=timezone.utc).isoformat()

    for impl_type in implementations:
        try:
            module, impl_name = loader.load_module(impl_type)
        except (FileNotFoundError, ImportError):
            continue

        for func_name in func_names:
            func = getattr(module, func_name, None)
            if func is None:
                continue

            args = func_args.get(func_name, ())

            # Warmup
            for _ in range(warmup):
                func(*args)

            # Timed runs
            timings: list[int] = []
            for _ in range(iterations):
                t0 = time.perf_counter_ns()
                func(*args)
                timings.append(time.perf_counter_ns() - t0)

            mean_ns = int(sum(timings) / len(timings))
            min_ns = min(timings)
            max_ns = max(timings)

            variance = sum((t - mean_ns) ** 2 for t in timings) / len(timings)
            stddev_ns = int(math.isqrt(int(variance)))

            key = f"{impl_name}/{func_name}"
            results[key] = BenchmarkResult(
                mean_ns=mean_ns,
                stddev_ns=stddev_ns,
                min_ns=min_ns,
                max_ns=max_ns,
            )

    baseline = BenchmarkBaseline(
        captured_at=captured_at,
        iterations=iterations,
        warmup=warmup,
        results=results,
    )

    if manifest_path is not None and module_key is not None:
        manifest = load(manifest_path)
        manifest = update_baseline(manifest, module_key, baseline)
        manifest = update_status(manifest, module_key, "baselined")
        save(manifest, manifest_path)

    return baseline


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main() -> None:  # pragma: no cover
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Capture a performance baseline for a Cython module."
    )
    parser.add_argument("module_path", help="Dotted import path to the module package")
    parser.add_argument("module_name", help="Module file name without extension")
    parser.add_argument(
        "--implementations",
        nargs="+",
        default=["AUGMENTED_PYTHON"],
        choices=[t.name for t in CythonModuleType],
        help="Implementation types to benchmark",
    )
    parser.add_argument(
        "--func-names", nargs="+", required=True, help="Function names to benchmark"
    )
    parser.add_argument(
        "--func-args",
        type=json.loads,
        default={},
        help="JSON mapping of func_name -> args list, e.g. '{\"add\": [1, 2]}'",
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--module-key", default=None)

    args = parser.parse_args()

    impl_types = [CythonModuleType[name] for name in args.implementations]
    func_args = {k: tuple(v) for k, v in args.func_args.items()}

    baseline = capture_baseline(
        module_path=args.module_path,
        module_name=args.module_name,
        implementations=impl_types,
        func_names=args.func_names,
        func_args=func_args,
        iterations=args.iterations,
        warmup=args.warmup,
        manifest_path=args.manifest_path,
        module_key=args.module_key,
    )

    print(f"Captured baseline at {baseline.captured_at}")
    print(f"  Iterations: {baseline.iterations}, Warmup: {baseline.warmup}")
    for key, result in baseline.results.items():
        print(f"  {key}: mean={result.mean_ns}ns stddev={result.stddev_ns}ns")


if __name__ == "__main__":
    _main()
