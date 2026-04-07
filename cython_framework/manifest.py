"""Migration manifest data model for Cython-to-Pure-Python migration.

All dataclasses are frozen (immutable). All CRUD operations return new objects.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any  # noqa: UP035

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ManifestVersionError(Exception):
    """Raised when the manifest schema version is incompatible."""


class InvalidStatusTransition(Exception):  # noqa: N818
    """Raised when an invalid module status transition is attempted."""


# ---------------------------------------------------------------------------
# Status machine
# ---------------------------------------------------------------------------

# Valid transitions: status -> set of allowed next statuses
_TRANSITIONS: dict[str, set[str]] = {
    "unanalyzed": {"analyzed", "flagged_for_rust"},
    "analyzed": {"thresholds_set"},
    "thresholds_set": {"tests_generated"},
    "tests_generated": {"baselined"},
    "baselined": {"migrating"},
    "migrating": {"validated"},
    # Terminal states - no outgoing transitions
    "flagged_for_rust": set(),
    "validated": set(),
}

_CURRENT_SCHEMA_VERSION = "1.0"
_SUPPORTED_MAJOR_VERSION = 1


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Hotspot:
    """A performance hotspot identified in a Cython annotate report."""

    function: str
    line: int
    interaction_type: str
    score: float


@dataclasses.dataclass(frozen=True)
class AnnotateScore:
    """Summary of a Cython annotate pass for a module."""

    total_lines: int
    yellow_lines: int
    score: float
    hotspots: tuple[Hotspot, ...] = dataclasses.field(default_factory=tuple)


@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
    """Timing result for a single benchmark function (all values in nanoseconds)."""

    mean_ns: int
    stddev_ns: int
    min_ns: int
    max_ns: int


@dataclasses.dataclass(frozen=True)
class BenchmarkBaseline:
    """Captured benchmark baseline for a module."""

    captured_at: str
    iterations: int
    warmup: int
    results: dict[str, BenchmarkResult] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Thresholds:
    """Performance regression thresholds for a module."""

    max_regression_factor: float
    set_by: str
    set_at: str


@dataclasses.dataclass(frozen=True)
class ModuleEntry:
    """Entry describing a single module in the migration manifest."""

    tier: int
    source_type: str
    source_path: str
    annotate_score: AnnotateScore | None = None
    benchmark_baseline: BenchmarkBaseline | None = None
    thresholds: Thresholds | None = None
    status: str = "unanalyzed"
    github_issue: str | None = None
    rust_candidate: bool = False
    rust_rationale: str | None = None


@dataclasses.dataclass(frozen=True)
class Manifest:
    """Top-level migration manifest."""

    project: str
    schema_version: str = _CURRENT_SCHEMA_VERSION
    modules: dict[str, ModuleEntry] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def path_to_module_key(path: str | Path, project_root: str | Path) -> str:
    """Convert a filesystem path to a dotted module key relative to project_root.

    Example: ``/project/core/engine.pyx`` -> ``core.engine``
    """
    path = Path(path)
    project_root = Path(project_root)
    relative = path.relative_to(project_root)
    parts = list(relative.parts)
    # Strip file extension from last component
    parts[-1] = relative.stem
    return ".".join(parts)


# ---------------------------------------------------------------------------
# Immutable CRUD helpers
# ---------------------------------------------------------------------------


def _replace_modules(manifest: Manifest, modules: dict[str, ModuleEntry]) -> Manifest:
    """Return a new Manifest with an updated modules dict."""
    return dataclasses.replace(manifest, modules=modules)


def replace_module(manifest: Manifest, key: str, entry: ModuleEntry) -> Manifest:
    """Return a new Manifest with the entry at *key* replaced.

    Raises KeyError if *key* does not exist.
    """
    if key not in manifest.modules:
        raise KeyError(key)
    new_modules = {**manifest.modules, key: entry}
    return _replace_modules(manifest, new_modules)


def add_module(manifest: Manifest, key: str, entry: ModuleEntry) -> Manifest:
    """Return a new Manifest with *entry* added (or replacing) at *key*."""
    new_modules = {**manifest.modules, key: entry}
    return _replace_modules(manifest, new_modules)


def update_baseline(manifest: Manifest, key: str, baseline: BenchmarkBaseline) -> Manifest:
    """Return a new Manifest with the benchmark baseline set on module *key*.

    Raises KeyError if *key* does not exist.
    """
    if key not in manifest.modules:
        raise KeyError(key)
    updated_entry = dataclasses.replace(manifest.modules[key], benchmark_baseline=baseline)
    return replace_module(manifest, key, updated_entry)


def set_thresholds(manifest: Manifest, key: str, thresholds: Thresholds) -> Manifest:
    """Return a new Manifest with the thresholds set on module *key*.

    Raises KeyError if *key* does not exist.
    """
    if key not in manifest.modules:
        raise KeyError(key)
    updated_entry = dataclasses.replace(manifest.modules[key], thresholds=thresholds)
    return replace_module(manifest, key, updated_entry)


def update_status(manifest: Manifest, key: str, new_status: str, force: bool = False) -> Manifest:
    """Return a new Manifest with the status updated on module *key*.

    Validates the transition against the status machine unless *force* is True.

    Raises:
        KeyError: if *key* does not exist.
        InvalidStatusTransition: if the transition is not permitted.
    """
    if key not in manifest.modules:
        raise KeyError(key)
    entry = manifest.modules[key]
    if not force:
        current = entry.status
        allowed = _TRANSITIONS.get(current, set())
        if new_status not in allowed:
            raise InvalidStatusTransition(
                f"Cannot transition '{current}' -> '{new_status}'. "
                f"Allowed: {sorted(allowed) or '(terminal state)'}"
            )
    updated_entry = dataclasses.replace(entry, status=new_status)
    return replace_module(manifest, key, updated_entry)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _hotspot_to_dict(h: Hotspot) -> dict[str, Any]:
    return {
        "function": h.function,
        "line": h.line,
        "interaction_type": h.interaction_type,
        "score": h.score,
    }


def _hotspot_from_dict(d: dict[str, Any]) -> Hotspot:
    return Hotspot(
        function=d["function"],
        line=d["line"],
        interaction_type=d["interaction_type"],
        score=d["score"],
    )


def _annotate_score_to_dict(a: AnnotateScore) -> dict[str, Any]:
    return {
        "total_lines": a.total_lines,
        "yellow_lines": a.yellow_lines,
        "score": a.score,
        "hotspots": [_hotspot_to_dict(h) for h in a.hotspots],
    }


def _annotate_score_from_dict(d: dict[str, Any]) -> AnnotateScore:
    return AnnotateScore(
        total_lines=d["total_lines"],
        yellow_lines=d["yellow_lines"],
        score=d["score"],
        hotspots=tuple(_hotspot_from_dict(h) for h in d.get("hotspots", [])),
    )


def _benchmark_result_to_dict(r: BenchmarkResult) -> dict[str, Any]:
    return {
        "mean_ns": r.mean_ns,
        "stddev_ns": r.stddev_ns,
        "min_ns": r.min_ns,
        "max_ns": r.max_ns,
    }


def _benchmark_result_from_dict(d: dict[str, Any]) -> BenchmarkResult:
    return BenchmarkResult(
        mean_ns=int(d["mean_ns"]),
        stddev_ns=int(d["stddev_ns"]),
        min_ns=int(d["min_ns"]),
        max_ns=int(d["max_ns"]),
    )


def _baseline_to_dict(b: BenchmarkBaseline) -> dict[str, Any]:
    return {
        "captured_at": b.captured_at,
        "iterations": b.iterations,
        "warmup": b.warmup,
        "results": {k: _benchmark_result_to_dict(v) for k, v in b.results.items()},
    }


def _baseline_from_dict(d: dict[str, Any]) -> BenchmarkBaseline:
    return BenchmarkBaseline(
        captured_at=d["captured_at"],
        iterations=d["iterations"],
        warmup=d["warmup"],
        results={k: _benchmark_result_from_dict(v) for k, v in d.get("results", {}).items()},
    )


def _thresholds_to_dict(t: Thresholds) -> dict[str, Any]:
    return {
        "max_regression_factor": t.max_regression_factor,
        "set_by": t.set_by,
        "set_at": t.set_at,
    }


def _thresholds_from_dict(d: dict[str, Any]) -> Thresholds:
    return Thresholds(
        max_regression_factor=d["max_regression_factor"],
        set_by=d["set_by"],
        set_at=d["set_at"],
    )


def _entry_to_dict(e: ModuleEntry) -> dict[str, Any]:
    return {
        "tier": e.tier,
        "source_type": e.source_type,
        "source_path": e.source_path,
        "annotate_score": (
            _annotate_score_to_dict(e.annotate_score) if e.annotate_score is not None else None
        ),
        "benchmark_baseline": (
            _baseline_to_dict(e.benchmark_baseline) if e.benchmark_baseline is not None else None
        ),
        "thresholds": (_thresholds_to_dict(e.thresholds) if e.thresholds is not None else None),
        "status": e.status,
        "github_issue": e.github_issue,
        "rust_candidate": e.rust_candidate,
        "rust_rationale": e.rust_rationale,
    }


def _entry_from_dict(d: dict[str, Any]) -> ModuleEntry:
    annotate_raw = d.get("annotate_score")
    baseline_raw = d.get("benchmark_baseline")
    thresholds_raw = d.get("thresholds")
    return ModuleEntry(
        tier=d["tier"],
        source_type=d["source_type"],
        source_path=d["source_path"],
        annotate_score=(
            _annotate_score_from_dict(annotate_raw) if annotate_raw is not None else None
        ),
        benchmark_baseline=(
            _baseline_from_dict(baseline_raw) if baseline_raw is not None else None
        ),
        thresholds=(_thresholds_from_dict(thresholds_raw) if thresholds_raw is not None else None),
        status=d.get("status", "unanalyzed"),
        github_issue=d.get("github_issue"),
        rust_candidate=d.get("rust_candidate", False),
        rust_rationale=d.get("rust_rationale"),
    )


def _manifest_to_dict(manifest: Manifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "project": manifest.project,
        "modules": {k: _entry_to_dict(v) for k, v in manifest.modules.items()},
    }


def _manifest_from_dict(d: dict[str, Any]) -> Manifest:
    version_str = d["schema_version"]
    try:
        major = int(version_str.split(".")[0])
    except (ValueError, IndexError) as exc:
        raise ManifestVersionError(f"Cannot parse schema_version: {version_str!r}") from exc
    if major != _SUPPORTED_MAJOR_VERSION:
        raise ManifestVersionError(
            f"Unsupported schema version {version_str!r}. "
            f"Expected major version {_SUPPORTED_MAJOR_VERSION}."
        )
    return Manifest(
        schema_version=version_str,
        project=d["project"],
        modules={k: _entry_from_dict(v) for k, v in d.get("modules", {}).items()},
    )


# ---------------------------------------------------------------------------
# Public I/O
# ---------------------------------------------------------------------------


def save(manifest: Manifest, path: str | Path) -> None:
    """Serialise *manifest* to JSON at *path*."""
    path = Path(path)
    path.write_text(
        json.dumps(_manifest_to_dict(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load(path: str | Path) -> Manifest:
    """Load a Manifest from *path*.

    Returns an empty Manifest (project="") if the file does not exist.

    Raises:
        ManifestVersionError: if the schema version is incompatible.
    """
    path = Path(path)
    if not path.exists():
        return Manifest(project="")
    data = json.loads(path.read_text(encoding="utf-8"))
    return _manifest_from_dict(data)
