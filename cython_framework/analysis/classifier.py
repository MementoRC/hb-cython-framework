"""Tier classifier for Cython/C++ module migration analysis.

Detects the migration tier of a source file:

- Tier 1: Python-compatible (augmented pure Python, plain ``.py`` / ``.pyx``
  without heavy Cython constructs).
- Tier 2: Heavy Cython – ``nogil``, memoryviews, ``fused`` types, ``cimport``,
  or ``cython.view``.
- Tier 3: Native C/C++ files (``.c``, ``.cc``, ``.cpp``, ``.h``, ``.hpp``)
  *or* ``.pyx`` files that contain a ``cdef extern from`` declaration.

CLI usage::

    python -m cython_framework.analysis.classifier --module <path>

Prints a JSON representation of the :class:`ClassificationResult`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ClassificationResult:
    """Immutable classification result for a single source file."""

    tier: int
    source_type: str
    evidence: list[str]
    rust_candidate: bool


# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------

# Tier-3 indicators inside .pyx files
_TIER3_PYX_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("cdef_extern", re.compile(r"\bcdef\s+extern\s+from\b")),
]

# Tier-2 indicators inside .pyx files
_TIER2_PYX_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("nogil", re.compile(r"\bnogil\b")),
    ("memoryview", re.compile(r"\b\w+\[[\w\s,:\]]*\]\s")),  # e.g. double[:]
    ("memoryview_arg", re.compile(r"\b\w+\[[\w\s:,]+\]")),  # memoryview in args
    ("cimport", re.compile(r"\bcimport\b")),
    ("fused", re.compile(r"\bfused\b")),
    ("cython.view", re.compile(r"\bcython\.view\b")),
]

# Tier 3 file extensions (always tier 3, regardless of content)
_TIER3_EXTENSIONS: frozenset[str] = frozenset({"c", "cc", "cpp", "h", "hpp"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _scan_patterns(
    content: str,
    patterns: list[tuple[str, re.Pattern[str]]],
) -> list[str]:
    """Return names of patterns that match anywhere in *content*."""
    return [name for name, pat in patterns if pat.search(content)]


def _classify_pyx(content: str) -> tuple[int, list[str]]:
    """Classify a ``.pyx`` file by scanning for tier-3 then tier-2 patterns."""
    tier3_evidence = _scan_patterns(content, _TIER3_PYX_PATTERNS)
    if tier3_evidence:
        return 3, tier3_evidence

    tier2_evidence = _scan_patterns(content, _TIER2_PYX_PATTERNS)
    if tier2_evidence:
        return 2, tier2_evidence

    return 1, []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify(path: str | Path) -> ClassificationResult:
    """Classify the migration tier of *path*.

    Parameters
    ----------
    path:
        Filesystem path to the source file (``str`` or :class:`pathlib.Path`).

    Returns
    -------
    ClassificationResult
        Immutable result with ``tier``, ``source_type``, ``evidence``, and
        ``rust_candidate`` fields.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    suffix = path.suffix.lstrip(".")  # e.g. "py", "pyx", "cpp"
    source_type = suffix

    # ------------------------------------------------------------------
    # Tier 3: native C/C++ by extension
    # ------------------------------------------------------------------
    if suffix in _TIER3_EXTENSIONS:
        evidence = [f"native_extension_{suffix}"]
        return ClassificationResult(
            tier=3,
            source_type=source_type,
            evidence=evidence,
            rust_candidate=True,
        )

    # ------------------------------------------------------------------
    # .pyx files: scan content for tier 3 / tier 2 patterns
    # ------------------------------------------------------------------
    if suffix == "pyx":
        content = path.read_text(encoding="utf-8", errors="replace")
        tier, evidence = _classify_pyx(content)
        rust_candidate = tier == 3
        return ClassificationResult(
            tier=tier,
            source_type=source_type,
            evidence=evidence,
            rust_candidate=rust_candidate,
        )

    # ------------------------------------------------------------------
    # Everything else (.py, etc.) → Tier 1
    # ------------------------------------------------------------------
    return ClassificationResult(
        tier=1,
        source_type=source_type,
        evidence=[],
        rust_candidate=False,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify the migration tier of a Cython/C++ source file.",
    )
    parser.add_argument(
        "--module",
        required=True,
        metavar="PATH",
        help="Path to the source file to classify.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI main – returns exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        result = classify(args.module)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    payload = {
        "tier": result.tier,
        "source_type": result.source_type,
        "evidence": result.evidence,
        "rust_candidate": result.rust_candidate,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
