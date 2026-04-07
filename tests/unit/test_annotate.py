"""Tests for cython_framework.analysis.annotate - Cython --annotate HTML parser.

TDD: Tests written BEFORE implementation.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from cython_framework.analysis.annotate import AnnotateResult, parse_annotate_html

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).parent.parent / "fixtures"
_TIER1_HTML = _FIXTURES / "annotate_output" / "tier1_example.html"


# ---------------------------------------------------------------------------
# AnnotateResult dataclass
# ---------------------------------------------------------------------------


class TestAnnotateResultDataclass:
    def test_construction(self):
        result = AnnotateResult(
            total_lines=10,
            yellow_lines=3,
            score=0.3,
            hotspots=[],
        )
        assert result.total_lines == 10
        assert result.yellow_lines == 3
        assert result.score == pytest.approx(0.3)
        assert result.hotspots == []

    def test_frozen(self):
        result = AnnotateResult(total_lines=10, yellow_lines=3, score=0.3, hotspots=[])
        with pytest.raises((AttributeError, TypeError)):
            result.total_lines = 5  # type: ignore[misc]

    def test_equality(self):
        r1 = AnnotateResult(total_lines=10, yellow_lines=3, score=0.3, hotspots=[])
        r2 = AnnotateResult(total_lines=10, yellow_lines=3, score=0.3, hotspots=[])
        assert r1 == r2

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(AnnotateResult)

    def test_json_serializable(self):
        result = AnnotateResult(total_lines=10, yellow_lines=3, score=0.3, hotspots=[])
        d = dataclasses.asdict(result)
        json_str = json.dumps(d)
        assert '"total_lines": 10' in json_str
        assert '"yellow_lines": 3' in json_str


# ---------------------------------------------------------------------------
# parse_annotate_html - fixture file
# ---------------------------------------------------------------------------


class TestParseAnnotateHtmlFixture:
    def test_fixture_exists(self):
        assert _TIER1_HTML.exists(), f"Fixture not found: {_TIER1_HTML}"

    def test_parse_tier1_fixture(self):
        result = parse_annotate_html(_TIER1_HTML)
        assert isinstance(result, AnnotateResult)

    def test_fixture_total_lines(self):
        """The fixture has 8 <span class="line"> elements."""
        result = parse_annotate_html(_TIER1_HTML)
        assert result.total_lines == 8

    def test_fixture_yellow_lines(self):
        """Lines 1, 3, 4, 7 have yellowish background colors in the fixture."""
        result = parse_annotate_html(_TIER1_HTML)
        assert result.yellow_lines == 4

    def test_fixture_score(self):
        """4 yellow / 8 total = 0.5"""
        result = parse_annotate_html(_TIER1_HTML)
        assert result.score == pytest.approx(4 / 8)

    def test_fixture_hotspots_empty_list(self):
        """Hotspots are reserved for future use."""
        result = parse_annotate_html(_TIER1_HTML)
        assert result.hotspots == []


# ---------------------------------------------------------------------------
# parse_annotate_html - synthetic HTML
# ---------------------------------------------------------------------------


_SYNTHETIC_ALL_YELLOW = """\
<!DOCTYPE html>
<html><head><title>Cython Annotate</title></head><body>
<div class="cython"><pre>
<span class="line" style="background-color: #FFFF00;">  1: <span>x = 1</span></span>
<span class="line" style="background-color: #FFFF00;">  2: <span>y = 2</span></span>
</pre></div>
</body></html>
"""

_SYNTHETIC_NO_YELLOW = """\
<!DOCTYPE html>
<html><head><title>Cython Annotate</title></head><body>
<div class="cython"><pre>
<span class="line" style="">  1: <span>x = 1</span></span>
<span class="line" style="">  2: <span>y = 2</span></span>
<span class="line" style="">  3: <span>z = 3</span></span>
</pre></div>
</body></html>
"""

_SYNTHETIC_MIXED = """\
<!DOCTYPE html>
<html><head><title>Cython Annotate</title></head><body>
<div class="cython"><pre>
<span class="line" style="background-color: #FFFF00;">  1: <span>import cython</span></span>
<span class="line" style="">  2: </span>
<span class="line" style="background-color: #FFFFaa;">  3: <span>@cython.ccall</span></span>
<span class="line" style="">  4: <span>    return a + b</span></span>
</pre></div>
</body></html>
"""

_SYNTHETIC_BLUE_NOT_YELLOW = """\
<!DOCTYPE html>
<html><head><title>Cython Annotate</title></head><body>
<div class="cython"><pre>
<span class="line" style="background-color: #0000FF;">  1: <span>x = 1</span></span>
<span class="line" style="background-color: #00FF00;">  2: <span>y = 2</span></span>
</pre></div>
</body></html>
"""


class TestParseAnnotateHtmlSynthetic:
    def test_all_yellow(self, tmp_path):
        f = tmp_path / "all_yellow.html"
        f.write_text(_SYNTHETIC_ALL_YELLOW)
        result = parse_annotate_html(f)
        assert result.total_lines == 2
        assert result.yellow_lines == 2
        assert result.score == pytest.approx(1.0)

    def test_no_yellow(self, tmp_path):
        f = tmp_path / "no_yellow.html"
        f.write_text(_SYNTHETIC_NO_YELLOW)
        result = parse_annotate_html(f)
        assert result.total_lines == 3
        assert result.yellow_lines == 0
        assert result.score == pytest.approx(0.0)

    def test_mixed(self, tmp_path):
        f = tmp_path / "mixed.html"
        f.write_text(_SYNTHETIC_MIXED)
        result = parse_annotate_html(f)
        assert result.total_lines == 4
        assert result.yellow_lines == 2
        assert result.score == pytest.approx(0.5)

    def test_blue_and_green_not_yellow(self, tmp_path):
        """Blue and green colors should not be counted as yellow."""
        f = tmp_path / "blue_green.html"
        f.write_text(_SYNTHETIC_BLUE_NOT_YELLOW)
        result = parse_annotate_html(f)
        assert result.total_lines == 2
        assert result.yellow_lines == 0
        assert result.score == pytest.approx(0.0)

    def test_returns_annotate_result_instance(self, tmp_path):
        f = tmp_path / "test.html"
        f.write_text(_SYNTHETIC_NO_YELLOW)
        result = parse_annotate_html(f)
        assert isinstance(result, AnnotateResult)

    def test_hotspots_always_empty_list(self, tmp_path):
        f = tmp_path / "test.html"
        f.write_text(_SYNTHETIC_MIXED)
        result = parse_annotate_html(f)
        assert result.hotspots == []


# ---------------------------------------------------------------------------
# parse_annotate_html - edge cases
# ---------------------------------------------------------------------------


class TestParseAnnotateHtmlEdgeCases:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_annotate_html(Path("/nonexistent/path/file.html"))

    def test_empty_html(self, tmp_path):
        """Empty HTML with no line spans should return zeros."""
        f = tmp_path / "empty.html"
        f.write_text("<html><body></body></html>")
        result = parse_annotate_html(f)
        assert result.total_lines == 0
        assert result.yellow_lines == 0
        assert result.score == pytest.approx(0.0)

    def test_accepts_path_object(self, tmp_path):
        f = tmp_path / "test.html"
        f.write_text(_SYNTHETIC_NO_YELLOW)
        result = parse_annotate_html(Path(f))
        assert isinstance(result, AnnotateResult)

    def test_accepts_string_path(self, tmp_path):
        f = tmp_path / "test.html"
        f.write_text(_SYNTHETIC_NO_YELLOW)
        result = parse_annotate_html(str(f))
        assert isinstance(result, AnnotateResult)

    def test_score_zero_when_no_lines(self, tmp_path):
        """Score must be 0.0 (not NaN or ZeroDivisionError) when total_lines == 0."""
        f = tmp_path / "empty.html"
        f.write_text("<html><body></body></html>")
        result = parse_annotate_html(f)
        assert result.score == pytest.approx(0.0)

    def test_yellow_detection_various_shades(self, tmp_path):
        """Various yellowish shades: high R, high G, low B (B < R)."""
        html = """\
<!DOCTYPE html>
<html><body><div class="cython"><pre>
<span class="line" style="background-color: #FFFF00;">  1: yellow</span>
<span class="line" style="background-color: #FFFFaa;">  2: light yellow</span>
<span class="line" style="background-color: #FFFF33;">  3: golden yellow</span>
<span class="line" style="background-color: #C8C800;">  4: dark yellow (R=200, G=200, B=0)</span>
<span class="line" style="background-color: #C8C801;">  5: borderline B<R</span>
<span class="line" style="">  6: no color</span>
</pre></div></body></html>
"""
        f = tmp_path / "shades.html"
        f.write_text(html)
        result = parse_annotate_html(f)
        assert result.total_lines == 6
        assert result.yellow_lines == 5

    def test_not_yellow_when_b_equals_r(self, tmp_path):
        """When B == R, not considered yellow (need B < R strictly)."""
        html = """\
<!DOCTYPE html>
<html><body><div class="cython"><pre>
<span class="line" style="background-color: #C8C8C8;">  1: gray (R=200,G=200,B=200)</span>
<span class="line" style="">  2: plain</span>
</pre></div></body></html>
"""
        f = tmp_path / "gray.html"
        f.write_text(html)
        result = parse_annotate_html(f)
        assert result.total_lines == 2
        assert result.yellow_lines == 0
