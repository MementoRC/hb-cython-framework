"""Tests for cython_framework.analysis.classifier - Tier classification logic.

TDD: Tests written BEFORE implementation.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from cython_framework.analysis.classifier import ClassificationResult, classify

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).parent.parent / "fixtures"
_TIER1_PY = _FIXTURES / "augmented_example" / "example.py"
_TIER2_PYX = _FIXTURES / "tier2_example" / "example.pyx"
_TIER3_CPP = _FIXTURES / "tier3_example" / "example.cpp"


# ---------------------------------------------------------------------------
# ClassificationResult dataclass
# ---------------------------------------------------------------------------


class TestClassificationResultDataclass:
    def test_construction(self):
        result = ClassificationResult(
            tier=1,
            source_type="py",
            evidence=["cython_annotations"],
            rust_candidate=False,
        )
        assert result.tier == 1
        assert result.source_type == "py"
        assert result.evidence == ["cython_annotations"]
        assert result.rust_candidate is False

    def test_frozen(self):
        result = ClassificationResult(tier=1, source_type="py", evidence=[], rust_candidate=False)
        with pytest.raises((AttributeError, TypeError)):
            result.tier = 2  # type: ignore[misc]

    def test_equality(self):
        r1 = ClassificationResult(
            tier=2, source_type="pyx", evidence=["nogil"], rust_candidate=False
        )
        r2 = ClassificationResult(
            tier=2, source_type="pyx", evidence=["nogil"], rust_candidate=False
        )
        assert r1 == r2

    def test_is_dataclass(self):
        result = ClassificationResult(tier=3, source_type="cpp", evidence=[], rust_candidate=True)
        assert dataclasses.is_dataclass(result)

    def test_tier_field_is_int(self):
        result = ClassificationResult(tier=1, source_type="py", evidence=[], rust_candidate=False)
        assert isinstance(result.tier, int)

    def test_source_type_field_is_str(self):
        result = ClassificationResult(tier=1, source_type="py", evidence=[], rust_candidate=False)
        assert isinstance(result.source_type, str)

    def test_evidence_field_is_list(self):
        result = ClassificationResult(
            tier=1, source_type="py", evidence=["x"], rust_candidate=False
        )
        assert isinstance(result.evidence, list)

    def test_rust_candidate_field_is_bool(self):
        result = ClassificationResult(tier=3, source_type="cpp", evidence=[], rust_candidate=True)
        assert isinstance(result.rust_candidate, bool)


# ---------------------------------------------------------------------------
# classify() - Tier 1 (augmented pure Python)
# ---------------------------------------------------------------------------


class TestClassifyTier1:
    def test_tier1_py_fixture(self):
        result = classify(_TIER1_PY)
        assert result.tier == 1

    def test_tier1_source_type(self):
        result = classify(_TIER1_PY)
        assert result.source_type == "py"

    def test_tier1_not_rust_candidate(self):
        result = classify(_TIER1_PY)
        assert result.rust_candidate is False

    def test_tier1_returns_classification_result(self):
        result = classify(_TIER1_PY)
        assert isinstance(result, ClassificationResult)

    def test_tier1_evidence_is_list(self):
        result = classify(_TIER1_PY)
        assert isinstance(result.evidence, list)

    def test_tier1_plain_py_file(self, tmp_path):
        """Any plain .py file with no special Cython patterns is Tier 1."""
        src = tmp_path / "plain.py"
        src.write_text("def hello():\n    return 42\n")
        result = classify(src)
        assert result.tier == 1
        assert result.source_type == "py"

    def test_tier1_plain_pyx_without_tier2_patterns(self, tmp_path):
        """A .pyx file without nogil/memoryview/cimport is Tier 1."""
        src = tmp_path / "simple.pyx"
        src.write_text("def greet(name):\n    return f'hello {name}'\n")
        result = classify(src)
        assert result.tier == 1
        assert result.source_type == "pyx"

    def test_tier1_accepts_string_path(self):
        result = classify(str(_TIER1_PY))
        assert result.tier == 1


# ---------------------------------------------------------------------------
# classify() - Tier 2 (heavy Cython: nogil, memoryviews, cimport, fused)
# ---------------------------------------------------------------------------


class TestClassifyTier2:
    def test_tier2_pyx_fixture(self):
        result = classify(_TIER2_PYX)
        assert result.tier == 2

    def test_tier2_source_type(self):
        result = classify(_TIER2_PYX)
        assert result.source_type == "pyx"

    def test_tier2_not_rust_candidate(self):
        result = classify(_TIER2_PYX)
        assert result.rust_candidate is False

    def test_tier2_has_evidence(self):
        result = classify(_TIER2_PYX)
        assert len(result.evidence) > 0

    def test_tier2_nogil_detected(self, tmp_path):
        src = tmp_path / "nogil_mod.pyx"
        src.write_text("def fast() nogil:\n    pass\n")
        result = classify(src)
        assert result.tier == 2
        assert any("nogil" in e for e in result.evidence)

    def test_tier2_memoryview_detected(self, tmp_path):
        src = tmp_path / "memview.pyx"
        src.write_text("def norm(double[:] arr):\n    pass\n")
        result = classify(src)
        assert result.tier == 2
        assert any("memoryview" in e for e in result.evidence)

    def test_tier2_cimport_detected(self, tmp_path):
        src = tmp_path / "cimport_mod.pyx"
        src.write_text("cimport numpy as np\n")
        result = classify(src)
        assert result.tier == 2
        assert any("cimport" in e for e in result.evidence)

    def test_tier2_fused_type_detected(self, tmp_path):
        src = tmp_path / "fused.pyx"
        src.write_text("ctypedef fused numeric:\n    int\n    double\n")
        result = classify(src)
        assert result.tier == 2
        assert any("fused" in e for e in result.evidence)

    def test_tier2_cython_view_detected(self, tmp_path):
        src = tmp_path / "cview.pyx"
        src.write_text("import cython.view\n")
        result = classify(src)
        assert result.tier == 2
        assert any("cython.view" in e for e in result.evidence)


# ---------------------------------------------------------------------------
# classify() - Tier 3 (C/C++ native or pyx with cdef extern)
# ---------------------------------------------------------------------------


class TestClassifyTier3:
    def test_tier3_cpp_fixture(self):
        result = classify(_TIER3_CPP)
        assert result.tier == 3

    def test_tier3_source_type_cpp(self):
        result = classify(_TIER3_CPP)
        assert result.source_type == "cpp"

    def test_tier3_rust_candidate(self):
        result = classify(_TIER3_CPP)
        assert result.rust_candidate is True

    def test_tier3_has_evidence(self):
        result = classify(_TIER3_CPP)
        assert len(result.evidence) > 0

    def test_tier3_c_extension(self, tmp_path):
        src = tmp_path / "native.c"
        src.write_text("int add(int a, int b) { return a + b; }\n")
        result = classify(src)
        assert result.tier == 3
        assert result.source_type == "c"
        assert result.rust_candidate is True

    def test_tier3_h_extension(self, tmp_path):
        src = tmp_path / "header.h"
        src.write_text("#pragma once\nint compute(int x);\n")
        result = classify(src)
        assert result.tier == 3
        assert result.source_type == "h"

    def test_tier3_hpp_extension(self, tmp_path):
        src = tmp_path / "header.hpp"
        src.write_text("#pragma once\nclass Foo {};\n")
        result = classify(src)
        assert result.tier == 3
        assert result.source_type == "hpp"

    def test_tier3_cc_extension(self, tmp_path):
        src = tmp_path / "impl.cc"
        src.write_text("int main() { return 0; }\n")
        result = classify(src)
        assert result.tier == 3
        assert result.source_type == "cc"

    def test_tier3_pyx_with_cdef_extern(self, tmp_path):
        src = tmp_path / "wrapper.pyx"
        src.write_text('cdef extern from "mylib.h":\n    double compute(double x)\n')
        result = classify(src)
        assert result.tier == 3
        assert result.source_type == "pyx"
        assert result.rust_candidate is True

    def test_tier3_pyx_cdef_extern_evidence(self, tmp_path):
        src = tmp_path / "wrapper.pyx"
        src.write_text('cdef extern from "mylib.h":\n    double compute(double x)\n')
        result = classify(src)
        assert any("cdef_extern" in e for e in result.evidence)

    def test_tier3_pyx_distutils_language_cpp(self, tmp_path):
        """A .pyx file with '# distutils: language=c++' is Tier 3."""
        src = tmp_path / "pubsub.pyx"
        src.write_text(
            "# distutils: language=c++\n"
            "# cython: language_level=3str\n"
            "\n"
            "def emit(topic, data):\n"
            "    pass\n"
        )
        result = classify(src)
        assert result.tier == 3
        assert result.source_type == "pyx"
        assert result.rust_candidate is True
        assert any("cpp_language" in e for e in result.evidence)

    def test_tier3_pyx_distutils_sources(self, tmp_path):
        """A .pyx file with '# distutils: sources=' is Tier 3."""
        src = tmp_path / "wrapped.pyx"
        src.write_text("# distutils: sources=mylib.cpp\n\ndef run():\n    pass\n")
        result = classify(src)
        assert result.tier == 3
        assert any("cpp_sources" in e for e in result.evidence)

    def test_tier3_pyx_cpp_stl_container(self, tmp_path):
        """A .pyx file using C++ STL containers is Tier 3."""
        src = tmp_path / "stl_mod.pyx"
        src.write_text(
            "from libcpp.unordered_map cimport unordered_map\n"
            "\n"
            "def make_map():\n"
            "    cdef unordered_map[int, int] m\n"
            "    return m\n"
        )
        result = classify(src)
        assert result.tier == 3
        assert any("cpp_stl_map" in e for e in result.evidence)

    def test_tier3_pyx_cpp_directive_via_pxd_companion(self, tmp_path):
        """C++ directive in a .pxd companion file escalates .pyx to Tier 3."""
        pyx = tmp_path / "mymod.pyx"
        pyx.write_text(
            "# cython: language_level=3str\n\ndef greet(name):\n    return f'hello {name}'\n"
        )
        pxd = tmp_path / "mymod.pxd"
        pxd.write_text(
            "# distutils: language=c++\n"
            "\n"
            "cdef extern from 'mymod_types.h':\n"
            "    ctypedef int MyInt\n"
        )
        result = classify(pyx)
        assert result.tier == 3
        assert result.rust_candidate is True
        assert "pxd_companion_analyzed" in result.evidence

    def test_tier2_pyx_with_pxd_companion_no_cpp(self, tmp_path):
        """A .pxd companion with only Tier-2 patterns keeps .pyx at Tier 2."""
        pyx = tmp_path / "heavymod.pyx"
        pyx.write_text(
            "# cython: language_level=3str\n\ndef fast(double[:] arr) nogil:\n    pass\n"
        )
        pxd = tmp_path / "heavymod.pxd"
        pxd.write_text("cimport numpy as np\n")
        result = classify(pyx)
        assert result.tier == 2
        assert result.rust_candidate is False
        assert "pxd_companion_analyzed" in result.evidence

    def test_tier2_fixture_still_tier2_no_cpp_directives(self):
        """Confirm tier2_example fixture has nogil/memoryview but no C++ directives → Tier 2."""
        result = classify(_TIER2_PYX)
        assert result.tier == 2
        assert result.rust_candidate is False
        # Must have at least one tier-2 evidence item
        assert len(result.evidence) > 0
        # Must NOT contain any tier-3 evidence keywords
        tier3_keywords = {
            "cdef_extern",
            "cpp_language",
            "cpp_sources",
            "cpp_stl_map",
            "cpp_stl_set",
            "cpp_stl_vector",
            "cpp_stl_pair",
        }
        assert not any(e in tier3_keywords for e in result.evidence)


# ---------------------------------------------------------------------------
# classify() - Error handling
# ---------------------------------------------------------------------------


class TestClassifyErrors:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "does_not_exist.py"
        with pytest.raises(FileNotFoundError):
            classify(missing)

    def test_missing_file_message_contains_path(self, tmp_path):
        missing = tmp_path / "ghost.pyx"
        with pytest.raises(FileNotFoundError, match="ghost.pyx"):
            classify(missing)

    def test_missing_file_string_path_raises(self, tmp_path):
        missing = str(tmp_path / "nope.cpp")
        with pytest.raises(FileNotFoundError):
            classify(missing)


# ---------------------------------------------------------------------------
# classify() - Result field completeness
# ---------------------------------------------------------------------------


class TestClassifyResultFields:
    def test_all_fields_present_tier1(self):
        result = classify(_TIER1_PY)
        assert hasattr(result, "tier")
        assert hasattr(result, "source_type")
        assert hasattr(result, "evidence")
        assert hasattr(result, "rust_candidate")

    def test_all_fields_present_tier2(self):
        result = classify(_TIER2_PYX)
        assert hasattr(result, "tier")
        assert hasattr(result, "source_type")
        assert hasattr(result, "evidence")
        assert hasattr(result, "rust_candidate")

    def test_all_fields_present_tier3(self):
        result = classify(_TIER3_CPP)
        assert hasattr(result, "tier")
        assert hasattr(result, "source_type")
        assert hasattr(result, "evidence")
        assert hasattr(result, "rust_candidate")

    def test_tier_is_int_for_all_tiers(self):
        for path in (_TIER1_PY, _TIER2_PYX, _TIER3_CPP):
            result = classify(path)
            assert isinstance(result.tier, int), f"tier not int for {path}"

    def test_source_type_matches_extension(self):
        assert classify(_TIER1_PY).source_type == "py"
        assert classify(_TIER2_PYX).source_type == "pyx"
        assert classify(_TIER3_CPP).source_type == "cpp"

    def test_evidence_contains_strings(self):
        for path in (_TIER1_PY, _TIER2_PYX, _TIER3_CPP):
            result = classify(path)
            for item in result.evidence:
                assert isinstance(item, str), f"evidence item not str for {path}"


# ---------------------------------------------------------------------------
# CLI (__main__) integration
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_outputs_json(self, tmp_path):
        """Running as __main__ --module <path> prints valid JSON."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cython_framework.analysis.classifier",
                "--module",
                str(_TIER1_PY),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert "tier" in data
        assert "source_type" in data
        assert "evidence" in data
        assert "rust_candidate" in data

    def test_cli_tier2(self, tmp_path):
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cython_framework.analysis.classifier",
                "--module",
                str(_TIER2_PYX),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["tier"] == 2

    def test_cli_tier3(self, tmp_path):
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cython_framework.analysis.classifier",
                "--module",
                str(_TIER3_CPP),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["tier"] == 3
        assert data["rust_candidate"] is True

    def test_cli_missing_file_nonzero_exit(self, tmp_path):
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cython_framework.analysis.classifier",
                "--module",
                str(tmp_path / "missing.py"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
