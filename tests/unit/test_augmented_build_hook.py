"""Tests for the env-gated, pragma-driven augmented Cython build hook."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from cython_framework.buildhook.hook import (
    AugmentedCythonBuildHook,
    _default_scan_paths,
    _module_name,
    compilation_enabled,
    discover_augmented,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
AUGMENTED = FIXTURES / "augmented_example"


class TestCompilationEnabled:
    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "Yes", "on", " on "])
    def test_truthy(self, val):
        assert compilation_enabled({"HB_COMPILE_AUGMENTED": val}) is True

    @pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "maybe"])
    def test_falsy(self, val):
        assert compilation_enabled({"HB_COMPILE_AUGMENTED": val}) is False

    def test_default_off_when_unset(self):
        assert compilation_enabled({}) is False


class TestDiscoverAugmented:
    def test_finds_pragma_file(self):
        found = discover_augmented(AUGMENTED)
        assert (AUGMENTED / "example.py") in found

    def test_skips_pure_python_variant(self):
        found = discover_augmented(AUGMENTED)
        assert all("__pure_python__" not in p.parts for p in found)

    def test_missing_base_returns_empty(self, tmp_path):
        assert discover_augmented(tmp_path / "nope") == []

    def test_non_pragma_file_excluded(self, tmp_path):
        (tmp_path / "plain.py").write_text("def f():\n    return 1\n")
        assert discover_augmented(tmp_path) == []

    def test_skips_dot_dirs_like_worktrees(self, tmp_path):
        # Regression for #31: a nested git worktree must not be discovered.
        for rel in ("pkg", ".worktrees/x/pkg", ".mypy_cache/pkg"):
            (tmp_path / rel).mkdir(parents=True)
            shutil.copy(AUGMENTED / "example.py", tmp_path / rel / "mod.py")
        assert discover_augmented(tmp_path) == [tmp_path / "pkg" / "mod.py"]

    def test_dot_dir_above_base_is_not_skipped(self, tmp_path):
        # Only components relative to the scan base count, so a project that
        # itself lives under a dot-dir (e.g. ~/.local/...) still works.
        base = tmp_path / ".local" / "proj"
        (base / "pkg").mkdir(parents=True)
        shutil.copy(AUGMENTED / "example.py", base / "pkg" / "mod.py")
        assert discover_augmented(base) == [base / "pkg" / "mod.py"]

    def test_skips_non_identifier_dir(self, tmp_path):
        (tmp_path / "my-dir").mkdir()
        shutil.copy(AUGMENTED / "example.py", tmp_path / "my-dir" / "mod.py")
        assert discover_augmented(tmp_path) == []


class TestModuleName:
    def test_dotted_name(self, tmp_path):
        assert _module_name(tmp_path / "pkg" / "mod.py", tmp_path) == "pkg.mod"

    def test_non_identifier_raises_naming_file(self, tmp_path):
        py = tmp_path / "my-src" / "mod.py"
        with pytest.raises(ValueError, match="my-src"):
            _module_name(py, tmp_path)


class TestDefaultScanPaths:
    class _Cfg:
        def __init__(self, packages=(), only_include=None):
            self.packages = list(packages)
            self.only_include = only_include or {}

    def test_no_build_config_falls_back_to_root(self):
        assert _default_scan_paths(None) == ["."]

    def test_unconfigured_target_falls_back_to_root(self):
        assert _default_scan_paths(self._Cfg()) == ["."]

    def test_prefers_packages(self):
        cfg = self._Cfg(packages=["candles_feed"], only_include={"/x/other": "other"})
        assert _default_scan_paths(cfg) == ["candles_feed"]

    def test_uses_only_include_sources(self):
        assert _default_scan_paths(self._Cfg(only_include={"/x/pkg": "pkg"})) == ["/x/pkg"]


class _FakeApp:
    def display_warning(self, message):  # pragma: no cover - trivial
        pass


def _make_hook(root, config=None):
    # BuildHookInterface.__init__(root, config, build_config, metadata,
    #                             directory, target_name, app=None)
    return AugmentedCythonBuildHook(
        str(root), config or {}, None, None, str(root), "wheel", _FakeApp()
    )


class TestOffIsNoOp:
    def test_off_leaves_build_data_untouched(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HB_COMPILE_AUGMENTED", raising=False)
        hook = _make_hook(tmp_path, {"paths": ["cython_framework"]})
        build_data = {"pure_python": True, "force_include": {}}
        hook.initialize("standard", build_data)
        # Strict no-op: nothing compiled, nothing added, no compiler required.
        assert build_data == {"pure_python": True, "force_include": {}}

    def test_off_with_falsy_value(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HB_COMPILE_AUGMENTED", "0")
        hook = _make_hook(tmp_path)
        build_data = {"pure_python": True}
        hook.initialize("standard", build_data)
        assert build_data == {"pure_python": True}


def _has_c_compiler() -> bool:
    return bool(shutil.which("cc") or shutil.which("gcc") or shutil.which("clang"))


class TestOnCompiles:
    def test_on_produces_so_for_augmented_module(self, monkeypatch, tmp_path):
        if not _has_c_compiler():
            pytest.skip("no C compiler available")
        # Isolated package with one augmented module copied from the fixture.
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        shutil.copy(AUGMENTED / "example.py", pkg / "example.py")

        monkeypatch.setenv("HB_COMPILE_AUGMENTED", "1")
        monkeypatch.chdir(tmp_path)  # keep setuptools build scratch inside tmp
        hook = _make_hook(tmp_path, {"paths": ["pkg"]})
        build_data: dict = {}
        hook.initialize("standard", build_data)

        assert build_data.get("pure_python") is False
        assert build_data.get("force_include")
        produced = list(pkg.glob("example.*.so")) + list(pkg.glob("example.*.pyd"))
        assert produced, "expected a compiled artifact next to the augmented module"

    def test_on_with_no_augmented_targets_is_pure_python(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HB_COMPILE_AUGMENTED", "1")
        empty = tmp_path / "empty"
        empty.mkdir()
        (empty / "plain.py").write_text("def f():\n    return 1\n")
        hook = _make_hook(tmp_path, {"paths": ["empty"]})
        build_data: dict = {}
        hook.initialize("standard", build_data)
        # No pragma files -> warning + pure-python (build_data untouched).
        assert build_data == {}
