"""CLI: benchmark all four CythonModuleType variants of an augmented module.

Builds PURE_PYTHON, AUGMENTED_PYTHON, COMPILED_AUGMENTED_PYTHON and
PURE_CYTHON variants of a single augmented ``.py`` source via
:func:`cython_framework.buildhook.variant_builder.build_variants`, loads all
four through :class:`~cython_framework.testing.cython_test_case.CythonModuleLoader`,
and times a module-level callable in each via
:meth:`~cython_framework.testing.cython_test_case.BenchmarkTestCase.benchmark_all`.

Usage:
    python -m cython_framework.benchmark_cli tests/fixtures/running_stats.py
    python -m cython_framework.benchmark_cli tests/fixtures/running_stats.py \
        --func bench_running_stats --runs 500
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from cython_framework.buildhook.variant_builder import VariantBuildError, build_variants
from cython_framework.testing.cython_test_case import (
    BenchmarkTestCase,
    CythonModuleLoader,
    CythonModuleType,
)

DEFAULT_FUNC = "bench_running_stats"
DEFAULT_RUNS = 200

_VARIANT_ORDER = (
    CythonModuleType.PURE_PYTHON,
    CythonModuleType.AUGMENTED_PYTHON,
    CythonModuleType.COMPILED_AUGMENTED_PYTHON,
    CythonModuleType.PURE_CYTHON,
)


def _resolve_source(module_path: str) -> Path:
    """Resolve a filesystem path or dotted module name to a .py source path."""
    candidate = Path(module_path)
    if candidate.suffix == ".py":
        return candidate
    return Path(*module_path.split(".")).with_suffix(".py")


def _project_root() -> Path:
    """Repo root: this file lives at <root>/cython_framework/benchmark_cli.py."""
    return Path(__file__).resolve().parent.parent


def _load_all_variants(module_path: str, module_name: str) -> dict[CythonModuleType, tuple]:
    loader = CythonModuleLoader(module_path, module_name)
    return {variant: loader.load_module(variant) for variant in _VARIANT_ORDER}


def _print_table(results: dict[CythonModuleType, object]) -> None:
    header = f"{'Variant':<28}{'Runs':>10}{'Total (s)':>12}{'Avg (ms)':>12}"
    print(header)
    print("-" * len(header))
    for variant in _VARIANT_ORDER:
        result = results[variant]
        print(
            f"{result.implementation:<28}{result.runs:>10}"
            f"{result.total_time:>12.4f}{result.avg_time * 1000:>12.4f}"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark all 4 CythonModuleType variants of an augmented Python module."
    )
    parser.add_argument("module_path", help="Path (or dotted name) to an augmented .py module")
    parser.add_argument(
        "--func",
        default=DEFAULT_FUNC,
        help=f"Module-level callable to benchmark (default: {DEFAULT_FUNC})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Iterations per variant (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--build-dir",
        default=None,
        help="Build variants into this directory instead of a scratch one under the repo root",
    )
    parser.add_argument(
        "--keep", action="store_true", help="Keep build artifacts on disk instead of deleting them"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    source = _resolve_source(args.module_path)
    if not source.exists():
        print(f"Module source not found: {source}", file=sys.stderr)
        return 1

    module_name = source.stem
    project_root = _project_root()
    build_dir = (
        Path(args.build_dir).resolve()
        if args.build_dir is not None
        else project_root / "_benchmark_build" / module_name
    )
    try:
        dotted_module_path = ".".join(build_dir.relative_to(project_root).parts)
    except ValueError:
        print(f"--build-dir must be inside the project root ({project_root})", file=sys.stderr)
        return 1

    try:
        artifacts = build_variants(source, module_name=module_name, build_dir=build_dir)
    except VariantBuildError as exc:
        print(
            f"Could not build all 4 variants (missing C compiler / Cython?): {exc}", file=sys.stderr
        )
        return 1
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        modules = _load_all_variants(dotted_module_path, module_name)
        benchmarker = BenchmarkTestCase.__new__(BenchmarkTestCase)
        benchmarker.modules = modules
        results = benchmarker.benchmark_all(args.func, num_runs=args.runs)
        _print_table(results)
    finally:
        if not args.keep:
            shutil.rmtree(artifacts.build_root, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
