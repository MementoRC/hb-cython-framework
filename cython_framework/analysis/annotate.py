"""Cython --annotate HTML parser.

Parses the HTML output produced by ``cython --annotate`` and extracts
yellow-line statistics that indicate Python interaction overhead.

Usage::

    from cython_framework.analysis.annotate import parse_annotate_html

    result = parse_annotate_html("my_module.html")
    print(result.score)   # fraction of yellow lines

CLI::

    python -m cython_framework.analysis.annotate --module my_module.html
"""

from __future__ import annotations

import dataclasses
import json
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AnnotateResult:
    """Result of parsing a Cython ``--annotate`` HTML report.

    Attributes:
        total_lines: Total number of source lines found in the HTML.
        yellow_lines: Number of lines with a yellowish background colour
            (indicating Python-level interaction overhead).
        score: Fraction of yellow lines (``yellow_lines / total_lines``).
            Returns ``0.0`` when there are no lines.
        hotspots: Reserved for future enhancement; always an empty list.
    """

    total_lines: int
    yellow_lines: int
    score: float
    hotspots: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# HTML parser implementation
# ---------------------------------------------------------------------------


def _is_yellowish(color_hex: str) -> bool:
    """Return True when *color_hex* represents a yellowish colour.

    A colour is considered yellowish when:
    - R > 200
    - G > 200
    - B < R   (strictly less – this excludes whites and grays)

    The value is a 6-character hex string (e.g. ``"FFFF00"`` or ``"FFFFaa"``).
    """
    if len(color_hex) != 6:
        return False
    try:
        r = int(color_hex[0:2], 16)
        g = int(color_hex[2:4], 16)
        b = int(color_hex[4:6], 16)
    except ValueError:
        return False
    return r > 200 and g > 200 and b < r


class _AnnotateHTMLParser(HTMLParser):
    """Minimal SAX-style parser for Cython annotate HTML output.

    Cython emits lines like::

        <span class="line" style="background-color: #FFFF00;">...</span>
        <span class="line" style="">...</span>

    We count every ``<span class="line">`` as a total line and check
    whether its ``style`` attribute contains a yellowish background colour.
    """

    def __init__(self) -> None:
        super().__init__()
        self.total_lines = 0
        self.yellow_lines = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "span":
            return

        attr_dict = dict(attrs)
        classes = (attr_dict.get("class") or "").split()
        if "line" not in classes:
            return

        self.total_lines += 1

        style = attr_dict.get("style") or ""
        color_hex = _extract_bg_color_hex(style)
        if color_hex and _is_yellowish(color_hex):
            self.yellow_lines += 1


def _extract_bg_color_hex(style: str) -> str | None:
    """Extract the 6-char hex colour from a CSS background-color value.

    Handles ``background-color: #RRGGBB`` (case-insensitive, tolerant of
    surrounding whitespace).  Returns ``None`` when no such value is found.
    """
    lower = style.lower()
    marker = "background-color:"
    idx = lower.find(marker)
    if idx == -1:
        return None
    rest = style[idx + len(marker) :].strip()
    if not rest.startswith("#"):
        return None
    hex_part = rest[1:7]  # take exactly 6 chars after '#'
    return hex_part if len(hex_part) == 6 else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_annotate_html(path: str | Path) -> AnnotateResult:
    """Parse a Cython ``--annotate`` HTML file and return statistics.

    Args:
        path: Path to the ``.html`` file produced by ``cython --annotate``.

    Returns:
        An :class:`AnnotateResult` with counts and a score.

    Raises:
        FileNotFoundError: When *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Annotate HTML file not found: {path}")

    html_content = path.read_text(encoding="utf-8")

    parser = _AnnotateHTMLParser()
    parser.feed(html_content)

    total = parser.total_lines
    yellow = parser.yellow_lines
    score = yellow / total if total > 0 else 0.0

    return AnnotateResult(
        total_lines=total,
        yellow_lines=yellow,
        score=score,
        hotspots=[],
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main() -> None:  # pragma: no cover
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Parse a Cython --annotate HTML file and print statistics as JSON."
    )
    parser.add_argument(
        "--module",
        required=True,
        metavar="PATH",
        help="Path to the .html file produced by cython --annotate",
    )
    args = parser.parse_args()

    try:
        result = parse_annotate_html(args.module)
    except FileNotFoundError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)

    print(json.dumps(dataclasses.asdict(result), indent=2))


if __name__ == "__main__":  # pragma: no cover
    _main()
