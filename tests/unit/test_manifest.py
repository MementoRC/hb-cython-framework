"""Tests for cython_framework.manifest - Migration manifest data model.

TDD: Tests written BEFORE implementation.
"""

from __future__ import annotations

import json

import pytest

from cython_framework.manifest import (
    AnnotateScore,
    BenchmarkBaseline,
    BenchmarkResult,
    Hotspot,
    InvalidStatusTransition,
    Manifest,
    ManifestVersionError,
    ModuleEntry,
    Thresholds,
    add_module,
    load,
    path_to_module_key,
    replace_module,
    save,
    set_thresholds,
    update_baseline,
    update_status,
)

# ---------------------------------------------------------------------------
# Dataclass construction & frozen behaviour
# ---------------------------------------------------------------------------


class TestHotspot:
    def test_construction(self):
        h = Hotspot(function="my_func", line=42, interaction_type="numpy", score=0.8)
        assert h.function == "my_func"
        assert h.line == 42
        assert h.interaction_type == "numpy"
        assert h.score == 0.8

    def test_frozen(self):
        h = Hotspot(function="f", line=1, interaction_type="t", score=0.0)
        with pytest.raises((AttributeError, TypeError)):
            h.score = 1.0  # type: ignore[misc]

    def test_equality(self):
        h1 = Hotspot(function="f", line=1, interaction_type="t", score=0.5)
        h2 = Hotspot(function="f", line=1, interaction_type="t", score=0.5)
        assert h1 == h2


class TestAnnotateScore:
    def test_construction_no_hotspots(self):
        a = AnnotateScore(total_lines=100, yellow_lines=10, score=0.1)
        assert a.total_lines == 100
        assert a.yellow_lines == 10
        assert a.score == 0.1
        assert a.hotspots == ()

    def test_construction_with_hotspots(self):
        h = Hotspot(function="f", line=1, interaction_type="t", score=0.5)
        a = AnnotateScore(total_lines=100, yellow_lines=10, score=0.1, hotspots=(h,))
        assert len(a.hotspots) == 1
        assert a.hotspots[0] == h

    def test_frozen(self):
        a = AnnotateScore(total_lines=10, yellow_lines=1, score=0.1)
        with pytest.raises((AttributeError, TypeError)):
            a.score = 0.9  # type: ignore[misc]


class TestBenchmarkResult:
    def test_construction(self):
        r = BenchmarkResult(mean_ns=1000, stddev_ns=50, min_ns=900, max_ns=1100)
        assert r.mean_ns == 1000
        assert r.stddev_ns == 50
        assert r.min_ns == 900
        assert r.max_ns == 1100

    def test_all_ints(self):
        r = BenchmarkResult(mean_ns=1, stddev_ns=2, min_ns=3, max_ns=4)
        assert isinstance(r.mean_ns, int)
        assert isinstance(r.stddev_ns, int)
        assert isinstance(r.min_ns, int)
        assert isinstance(r.max_ns, int)

    def test_frozen(self):
        r = BenchmarkResult(mean_ns=1, stddev_ns=2, min_ns=3, max_ns=4)
        with pytest.raises((AttributeError, TypeError)):
            r.mean_ns = 999  # type: ignore[misc]


class TestBenchmarkBaseline:
    def test_construction_empty_results(self):
        b = BenchmarkBaseline(
            captured_at="2024-01-01T00:00:00",
            iterations=1000,
            warmup=100,
        )
        assert b.captured_at == "2024-01-01T00:00:00"
        assert b.iterations == 1000
        assert b.warmup == 100
        assert b.results == {}

    def test_construction_with_results(self):
        r = BenchmarkResult(mean_ns=500, stddev_ns=10, min_ns=480, max_ns=520)
        b = BenchmarkBaseline(
            captured_at="2024-01-01T00:00:00",
            iterations=100,
            warmup=10,
            results={"test_func": r},
        )
        assert "test_func" in b.results
        assert b.results["test_func"] == r

    def test_frozen(self):
        b = BenchmarkBaseline(captured_at="t", iterations=1, warmup=0)
        with pytest.raises((AttributeError, TypeError)):
            b.iterations = 99  # type: ignore[misc]


class TestThresholds:
    def test_construction(self):
        t = Thresholds(max_regression_factor=1.2, set_by="user", set_at="2024-01-01")
        assert t.max_regression_factor == 1.2
        assert t.set_by == "user"
        assert t.set_at == "2024-01-01"

    def test_frozen(self):
        t = Thresholds(max_regression_factor=1.0, set_by="u", set_at="now")
        with pytest.raises((AttributeError, TypeError)):
            t.max_regression_factor = 2.0  # type: ignore[misc]


class TestModuleEntry:
    def test_defaults(self):
        e = ModuleEntry(tier=1, source_type="pyx", source_path="a/b.pyx")
        assert e.tier == 1
        assert e.source_type == "pyx"
        assert e.source_path == "a/b.pyx"
        assert e.annotate_score is None
        assert e.benchmark_baseline is None
        assert e.thresholds is None
        assert e.status == "unanalyzed"
        assert e.github_issue is None
        assert e.rust_candidate is False
        assert e.rust_rationale is None

    def test_custom_status(self):
        e = ModuleEntry(tier=2, source_type="py", source_path="c.py", status="analyzed")
        assert e.status == "analyzed"

    def test_frozen(self):
        e = ModuleEntry(tier=1, source_type="pyx", source_path="x.pyx")
        with pytest.raises((AttributeError, TypeError)):
            e.tier = 2  # type: ignore[misc]


class TestManifest:
    def test_defaults(self):
        m = Manifest(project="my_project")
        assert m.schema_version == "1.0"
        assert m.project == "my_project"
        assert m.modules == {}

    def test_custom_version(self):
        m = Manifest(project="p", schema_version="1.5")
        assert m.schema_version == "1.5"

    def test_frozen(self):
        m = Manifest(project="p")
        with pytest.raises((AttributeError, TypeError)):
            m.project = "q"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# path_to_module_key
# ---------------------------------------------------------------------------


class TestPathToModuleKey:
    def test_simple(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        src = root / "module" / "sub.py"
        src.parent.mkdir(parents=True)
        src.touch()
        key = path_to_module_key(src, root)
        assert key == "module.sub"

    def test_pyx_extension(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        src = root / "core" / "engine.pyx"
        src.parent.mkdir(parents=True)
        src.touch()
        key = path_to_module_key(src, root)
        assert key == "core.engine"

    def test_top_level_file(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        src = root / "toplevel.py"
        src.touch()
        key = path_to_module_key(src, root)
        assert key == "toplevel"

    def test_deep_nesting(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        src = root / "a" / "b" / "c" / "deep.pyx"
        src.parent.mkdir(parents=True)
        src.touch()
        key = path_to_module_key(src, root)
        assert key == "a.b.c.deep"

    def test_string_paths_accepted(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        src = root / "mod.py"
        src.touch()
        key = path_to_module_key(str(src), str(root))
        assert key == "mod"


# ---------------------------------------------------------------------------
# CRUD operations (immutable)
# ---------------------------------------------------------------------------


def _make_manifest() -> Manifest:
    return Manifest(project="test_project")


def _make_entry(tier: int = 1, status: str = "unanalyzed") -> ModuleEntry:
    return ModuleEntry(tier=tier, source_type="pyx", source_path="a.pyx", status=status)


class TestAddModule:
    def test_adds_to_empty(self):
        m = _make_manifest()
        e = _make_entry()
        m2 = add_module(m, "a.b", e)
        assert "a.b" in m2.modules
        assert m2.modules["a.b"] == e

    def test_returns_new_manifest(self):
        m = _make_manifest()
        m2 = add_module(m, "key", _make_entry())
        assert m is not m2

    def test_original_unchanged(self):
        m = _make_manifest()
        add_module(m, "key", _make_entry())
        assert "key" not in m.modules

    def test_replaces_existing(self):
        m = _make_manifest()
        e1 = _make_entry(tier=1)
        e2 = _make_entry(tier=2)
        m2 = add_module(m, "key", e1)
        m3 = add_module(m2, "key", e2)
        assert m3.modules["key"].tier == 2

    def test_preserves_other_modules(self):
        m = _make_manifest()
        m = add_module(m, "a", _make_entry(tier=1))
        m = add_module(m, "b", _make_entry(tier=2))
        assert "a" in m.modules
        assert "b" in m.modules


class TestReplaceModule:
    def test_replaces(self):
        m = _make_manifest()
        e1 = _make_entry(tier=1)
        e2 = _make_entry(tier=3)
        m = add_module(m, "key", e1)
        m2 = replace_module(m, "key", e2)
        assert m2.modules["key"].tier == 3

    def test_returns_new_manifest(self):
        m = add_module(_make_manifest(), "k", _make_entry())
        m2 = replace_module(m, "k", _make_entry(tier=2))
        assert m is not m2

    def test_missing_key_raises(self):
        m = _make_manifest()
        with pytest.raises(KeyError):
            replace_module(m, "nonexistent", _make_entry())


class TestUpdateBaseline:
    def test_sets_baseline(self):
        m = add_module(_make_manifest(), "mod", _make_entry())
        baseline = BenchmarkBaseline(captured_at="2024-01-01", iterations=1000, warmup=100)
        m2 = update_baseline(m, "mod", baseline)
        assert m2.modules["mod"].benchmark_baseline == baseline

    def test_returns_new_manifest(self):
        m = add_module(_make_manifest(), "mod", _make_entry())
        baseline = BenchmarkBaseline(captured_at="t", iterations=1, warmup=0)
        m2 = update_baseline(m, "mod", baseline)
        assert m is not m2

    def test_original_entry_unchanged(self):
        m = add_module(_make_manifest(), "mod", _make_entry())
        baseline = BenchmarkBaseline(captured_at="t", iterations=1, warmup=0)
        update_baseline(m, "mod", baseline)
        assert m.modules["mod"].benchmark_baseline is None

    def test_missing_key_raises(self):
        m = _make_manifest()
        baseline = BenchmarkBaseline(captured_at="t", iterations=1, warmup=0)
        with pytest.raises(KeyError):
            update_baseline(m, "nonexistent", baseline)


class TestSetThresholds:
    def test_sets_thresholds(self):
        m = add_module(_make_manifest(), "mod", _make_entry())
        t = Thresholds(max_regression_factor=1.2, set_by="ci", set_at="2024-01-01")
        m2 = set_thresholds(m, "mod", t)
        assert m2.modules["mod"].thresholds == t

    def test_returns_new_manifest(self):
        m = add_module(_make_manifest(), "mod", _make_entry())
        t = Thresholds(max_regression_factor=1.1, set_by="u", set_at="now")
        m2 = set_thresholds(m, "mod", t)
        assert m is not m2

    def test_missing_key_raises(self):
        m = _make_manifest()
        t = Thresholds(max_regression_factor=1.0, set_by="u", set_at="t")
        with pytest.raises(KeyError):
            set_thresholds(m, "nonexistent", t)


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------


class TestUpdateStatus:
    def test_valid_transition_unanalyzed_to_analyzed(self):
        m = add_module(_make_manifest(), "mod", _make_entry(status="unanalyzed"))
        m2 = update_status(m, "mod", "analyzed")
        assert m2.modules["mod"].status == "analyzed"

    def test_valid_transition_unanalyzed_to_flagged(self):
        m = add_module(_make_manifest(), "mod", _make_entry(status="unanalyzed"))
        m2 = update_status(m, "mod", "flagged_for_rust")
        assert m2.modules["mod"].status == "flagged_for_rust"

    def test_valid_transition_analyzed_to_thresholds_set(self):
        m = add_module(_make_manifest(), "mod", _make_entry(status="analyzed"))
        m2 = update_status(m, "mod", "thresholds_set")
        assert m2.modules["mod"].status == "thresholds_set"

    def test_valid_transition_thresholds_set_to_tests_generated(self):
        m = add_module(_make_manifest(), "mod", _make_entry(status="thresholds_set"))
        m2 = update_status(m, "mod", "tests_generated")
        assert m2.modules["mod"].status == "tests_generated"

    def test_valid_transition_tests_generated_to_baselined(self):
        m = add_module(_make_manifest(), "mod", _make_entry(status="tests_generated"))
        m2 = update_status(m, "mod", "baselined")
        assert m2.modules["mod"].status == "baselined"

    def test_valid_transition_baselined_to_migrating(self):
        m = add_module(_make_manifest(), "mod", _make_entry(status="baselined"))
        m2 = update_status(m, "mod", "migrating")
        assert m2.modules["mod"].status == "migrating"

    def test_valid_transition_migrating_to_validated(self):
        m = add_module(_make_manifest(), "mod", _make_entry(status="migrating"))
        m2 = update_status(m, "mod", "validated")
        assert m2.modules["mod"].status == "validated"

    def test_invalid_skip_raises(self):
        m = add_module(_make_manifest(), "mod", _make_entry(status="unanalyzed"))
        with pytest.raises(InvalidStatusTransition):
            update_status(m, "mod", "thresholds_set")

    def test_terminal_flagged_for_rust_raises(self):
        m = add_module(_make_manifest(), "mod", _make_entry(status="flagged_for_rust"))
        with pytest.raises(InvalidStatusTransition):
            update_status(m, "mod", "analyzed")

    def test_terminal_validated_raises(self):
        m = add_module(_make_manifest(), "mod", _make_entry(status="validated"))
        with pytest.raises(InvalidStatusTransition):
            update_status(m, "mod", "migrating")

    def test_backward_transition_raises(self):
        m = add_module(_make_manifest(), "mod", _make_entry(status="analyzed"))
        with pytest.raises(InvalidStatusTransition):
            update_status(m, "mod", "unanalyzed")

    def test_force_bypasses_validation(self):
        m = add_module(_make_manifest(), "mod", _make_entry(status="validated"))
        m2 = update_status(m, "mod", "unanalyzed", force=True)
        assert m2.modules["mod"].status == "unanalyzed"

    def test_returns_new_manifest(self):
        m = add_module(_make_manifest(), "mod", _make_entry(status="unanalyzed"))
        m2 = update_status(m, "mod", "analyzed")
        assert m is not m2

    def test_missing_key_raises(self):
        m = _make_manifest()
        with pytest.raises(KeyError):
            update_status(m, "nonexistent", "analyzed")


# ---------------------------------------------------------------------------
# Save / Load roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def _full_manifest(self) -> Manifest:
        hotspot = Hotspot(function="my_fn", line=10, interaction_type="numpy", score=0.7)
        score = AnnotateScore(total_lines=200, yellow_lines=20, score=0.1, hotspots=(hotspot,))
        result = BenchmarkResult(mean_ns=1000, stddev_ns=50, min_ns=900, max_ns=1100)
        baseline = BenchmarkBaseline(
            captured_at="2024-01-01T00:00:00",
            iterations=1000,
            warmup=100,
            results={"my_fn": result},
        )
        thresholds = Thresholds(max_regression_factor=1.2, set_by="ci", set_at="2024-01-01")
        entry = ModuleEntry(
            tier=1,
            source_type="pyx",
            source_path="hummingbot/core/engine.pyx",
            annotate_score=score,
            benchmark_baseline=baseline,
            thresholds=thresholds,
            status="baselined",
            github_issue=42,
            rust_candidate=True,
            rust_rationale="performance critical",
        )
        return add_module(Manifest(project="hummingbot"), "hummingbot.core.engine", entry)

    def test_roundtrip_full(self, tmp_path):
        m = self._full_manifest()
        path = tmp_path / "manifest.json"
        save(m, path)
        m2 = load(path)
        assert m2.project == m.project
        assert m2.schema_version == m.schema_version
        assert set(m2.modules.keys()) == set(m.modules.keys())

    def test_roundtrip_entry_fields(self, tmp_path):
        m = self._full_manifest()
        path = tmp_path / "manifest.json"
        save(m, path)
        m2 = load(path)
        e = m2.modules["hummingbot.core.engine"]
        assert e.tier == 1
        assert e.source_type == "pyx"
        assert e.status == "baselined"
        assert e.github_issue == 42
        assert e.rust_candidate is True
        assert e.rust_rationale == "performance critical"

    def test_roundtrip_benchmark_baseline(self, tmp_path):
        m = self._full_manifest()
        path = tmp_path / "manifest.json"
        save(m, path)
        m2 = load(path)
        e = m2.modules["hummingbot.core.engine"]
        assert e.benchmark_baseline is not None
        assert e.benchmark_baseline.iterations == 1000
        assert e.benchmark_baseline.warmup == 100
        assert "my_fn" in e.benchmark_baseline.results
        r = e.benchmark_baseline.results["my_fn"]
        assert r.mean_ns == 1000
        assert r.stddev_ns == 50

    def test_roundtrip_annotate_score(self, tmp_path):
        m = self._full_manifest()
        path = tmp_path / "manifest.json"
        save(m, path)
        m2 = load(path)
        e = m2.modules["hummingbot.core.engine"]
        assert e.annotate_score is not None
        assert e.annotate_score.total_lines == 200
        assert len(e.annotate_score.hotspots) == 1
        assert e.annotate_score.hotspots[0].function == "my_fn"

    def test_roundtrip_thresholds(self, tmp_path):
        m = self._full_manifest()
        path = tmp_path / "manifest.json"
        save(m, path)
        m2 = load(path)
        e = m2.modules["hummingbot.core.engine"]
        assert e.thresholds is not None
        assert e.thresholds.max_regression_factor == 1.2

    def test_save_creates_json_file(self, tmp_path):
        m = _make_manifest()
        path = tmp_path / "out.json"
        save(m, path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "schema_version" in data
        assert "project" in data
        assert "modules" in data

    def test_load_missing_file_returns_empty_manifest(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        m = load(path)
        assert isinstance(m, Manifest)
        assert m.modules == {}

    def test_load_string_path(self, tmp_path):
        m = _make_manifest()
        path = tmp_path / "manifest.json"
        save(m, path)
        m2 = load(str(path))
        assert m2.project == m.project

    def test_save_string_path(self, tmp_path):
        m = _make_manifest()
        path = tmp_path / "manifest.json"
        save(m, str(path))
        assert path.exists()


# ---------------------------------------------------------------------------
# Schema version validation
# ---------------------------------------------------------------------------


class TestSchemaVersioning:
    def test_same_major_version_accepted(self, tmp_path):
        m = Manifest(project="p", schema_version="1.3")
        path = tmp_path / "m.json"
        save(m, path)
        # Should not raise
        m2 = load(path)
        assert m2.schema_version == "1.3"

    def test_different_major_version_raises(self, tmp_path):
        data = {"schema_version": "2.0", "project": "p", "modules": {}}
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data))
        with pytest.raises(ManifestVersionError):
            load(path)

    def test_version_error_message_informative(self, tmp_path):
        data = {"schema_version": "3.1", "project": "p", "modules": {}}
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data))
        with pytest.raises(ManifestVersionError, match="3"):
            load(path)

    def test_missing_version_field_raises(self, tmp_path):
        data = {"project": "p", "modules": {}}
        path = tmp_path / "m.json"
        path.write_text(json.dumps(data))
        with pytest.raises((ManifestVersionError, KeyError)):
            load(path)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_manifest_with_many_modules(self):
        m = _make_manifest()
        for i in range(50):
            m = add_module(m, f"module.{i}", _make_entry(tier=i % 3 + 1))
        assert len(m.modules) == 50

    def test_module_entry_with_all_optional_fields(self):
        h = Hotspot(function="f", line=1, interaction_type="t", score=0.5)
        score = AnnotateScore(total_lines=10, yellow_lines=1, score=0.1, hotspots=(h,))
        result = BenchmarkResult(mean_ns=100, stddev_ns=5, min_ns=90, max_ns=110)
        baseline = BenchmarkBaseline(
            captured_at="now", iterations=10, warmup=1, results={"f": result}
        )
        threshold = Thresholds(max_regression_factor=1.5, set_by="user", set_at="now")
        e = ModuleEntry(
            tier=1,
            source_type="pyx",
            source_path="x.pyx",
            annotate_score=score,
            benchmark_baseline=baseline,
            thresholds=threshold,
            status="baselined",
            github_issue=123,
            rust_candidate=True,
            rust_rationale="it's fast",
        )
        assert e.annotate_score == score
        assert e.benchmark_baseline == baseline
        assert e.thresholds == threshold

    def test_invalid_status_transition_error_type(self):
        m = add_module(_make_manifest(), "mod", _make_entry(status="unanalyzed"))
        exc = None
        try:
            update_status(m, "mod", "validated")
        except InvalidStatusTransition as e:
            exc = e
        assert exc is not None
        assert isinstance(exc, InvalidStatusTransition)

    def test_benchmark_baseline_immutable_results(self):
        r = BenchmarkResult(mean_ns=100, stddev_ns=5, min_ns=90, max_ns=110)
        b = BenchmarkBaseline(captured_at="t", iterations=1, warmup=0, results={"f": r})
        # The results dict on a frozen dataclass should not be directly mutable
        # via reassignment (though the dict itself isn't frozen - that's expected)
        with pytest.raises((AttributeError, TypeError)):
            b.results = {}  # type: ignore[misc]
