"""Tests for G5 -- scaffold command generating the augmented triple-layout.

Exercises cython_framework.scaffold_cli.scaffold_module() against a throwaway
module path under tmp_path, verifying the file set CythonModuleLoader expects,
idempotency, and --force / pragma-preservation semantics.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cython_framework.hooks.link_augmented_pyx import PRAGMA
from cython_framework.scaffold_cli import main, scaffold_module


@pytest.fixture
def module_path(tmp_path: Path) -> Path:
    return tmp_path / "pkg" / "widget.py"


def _expected_layout(module_path: Path) -> set[Path]:
    module_dir = module_path.parent
    return {
        module_path,
        module_dir / "__init__.py",
        module_dir / "__pure_python__" / "__init__.py",
        module_dir / "__pure_python__" / "widget.py",
        module_dir / "__pure_cython__" / "__init__.py",
        module_dir / "__pure_cython__" / "widget.pyx",
        module_dir / "widget.pyx",
    }


def test_scaffold_new_module_creates_full_layout(module_path):
    result = scaffold_module(module_path)

    module_dir = module_path.parent
    assert module_path.exists()
    assert (module_dir / "__init__.py").exists()
    assert (module_dir / "widget.pyx").is_symlink()
    assert (module_dir / "widget.pyx").resolve() == module_path.resolve()
    assert (module_dir / "__pure_python__" / "__init__.py").exists()
    assert (module_dir / "__pure_python__" / "widget.py").exists()
    assert (module_dir / "__pure_cython__" / "__init__.py").exists()
    assert (module_dir / "__pure_cython__" / "widget.pyx").exists()

    assert set(result.created) == _expected_layout(module_path)
    assert result.skipped == []


def test_scaffold_new_module_pragma_present(module_path):
    scaffold_module(module_path)
    assert PRAGMA in module_path.read_text()


def test_scaffold_new_module_matching_example_signatures(module_path):
    scaffold_module(module_path)
    module_dir = module_path.parent
    assert "def example(a: cython.int, b: cython.int)" in module_path.read_text()
    assert (
        "def example(a: int, b: int)" in (module_dir / "__pure_python__" / "widget.py").read_text()
    )
    assert (
        "cpdef int example(int a, int b)"
        in (module_dir / "__pure_cython__" / "widget.pyx").read_text()
    )


def test_scaffold_is_idempotent(module_path):
    scaffold_module(module_path)
    second = scaffold_module(module_path)

    assert second.created == []
    assert set(second.skipped) == _expected_layout(module_path)


def test_scaffold_never_overwrites_existing_source(module_path):
    scaffold_module(module_path)
    custom = module_path.read_text() + "\n\ndef real_function():\n    return 42\n"
    module_path.write_text(custom)

    scaffold_module(module_path, force=True)

    assert module_path.read_text() == custom


def test_scaffold_force_regenerates_stubs_as_todo_for_existing_source(module_path):
    scaffold_module(module_path)
    module_dir = module_path.parent
    pure_python_file = module_dir / "__pure_python__" / "widget.py"

    result = scaffold_module(module_path, force=True)

    assert pure_python_file in result.created
    assert "TODO" in pure_python_file.read_text()


def test_scaffold_adds_pragma_to_existing_unmarked_source(tmp_path):
    module_path = tmp_path / "pkg" / "legacy.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("def legacy():\n    return 1\n")

    result = scaffold_module(module_path)

    assert PRAGMA in module_path.read_text()
    assert "def legacy():" in module_path.read_text()
    assert module_path in result.created


def test_scaffold_skips_pragma_when_already_augmented(tmp_path):
    module_path = tmp_path / "pkg" / "already.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(f"# cython: language_level=3str\n{PRAGMA}\n\ndef f():\n    return 1\n")
    original = module_path.read_text()

    result = scaffold_module(module_path)

    assert module_path.read_text() == original
    assert module_path in result.skipped


def test_scaffold_does_not_clobber_real_pyx_file(tmp_path):
    module_path = tmp_path / "pkg" / "guarded.py"
    module_path.parent.mkdir(parents=True)
    pyx_path = module_path.with_suffix(".pyx")
    pyx_path.write_text("# a real, hand-written .pyx file\n")

    scaffold_module(module_path)

    assert pyx_path.read_text() == "# a real, hand-written .pyx file\n"
    assert not pyx_path.is_symlink()


def test_cli_main_smoke(tmp_path, capsys):
    module_path = tmp_path / "pkg" / "cli_widget.py"
    exit_code = main([str(module_path)])

    assert exit_code == 0
    assert module_path.exists()
    captured = capsys.readouterr()
    assert "Created:" in captured.out
