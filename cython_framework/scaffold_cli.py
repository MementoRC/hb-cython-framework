"""CLI: scaffold the __pure_python__ / __pure_cython__ triple-layout for an augmented module.

The authoring-side counterpart to
:func:`cython_framework.buildhook.variant_builder.build_variants` (which builds the
on-disk variants at test time). Given the path to an augmented module, this generates
the on-disk skeleton :class:`~cython_framework.testing.cython_test_case.CythonModuleLoader`
reads:

    <module_dir>/
        __init__.py
        <name>.py             # augmented Python source (created if missing)
        <name>.pyx            # symlink -> <name>.py
        __pure_python__/
            __init__.py
            <name>.py         # hand-maintained plain-Python reference
        __pure_cython__/
            __init__.py
            <name>.pyx        # hand-maintained pure-Cython implementation

If the augmented source already exists, it is never overwritten -- only its pragma
header is added when missing. In that case the pure_python/pure_cython stubs (when
generated) are conservative TODO placeholders rather than a working example, since the
real module's public API is not introspected.

Usage:
    python -m cython_framework.scaffold_cli hummingbot/data_feed/my_module/utils.py
    python -m cython_framework.scaffold_cli hummingbot.data_feed.my_module.utils --force
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from cython_framework.hooks.link_augmented_pyx import PRAGMA, is_augmented

_LANGUAGE_LEVEL_PRAGMA = "# cython: language_level=3str"

_AUGMENTED_TEMPLATE = f'''\
{_LANGUAGE_LEVEL_PRAGMA}
{PRAGMA}

import cython


@cython.ccall
def example(a: cython.int, b: cython.int) -> cython.int:
    """Placeholder scaffolded function -- replace with real logic."""
    return a + b
'''

_PURE_PYTHON_TEMPLATE = '''\
def example(a: int, b: int) -> int:
    """Placeholder scaffolded function -- replace with real logic."""
    return a + b
'''

_PURE_CYTHON_TEMPLATE = f'''\
{_LANGUAGE_LEVEL_PRAGMA}

cpdef int example(int a, int b):
    """Placeholder scaffolded function -- replace with a hand-tuned Cython implementation."""
    return a + b
'''

_PURE_PYTHON_STUB = '''\
"""Pure Python reference implementation for {name}.

TODO: mirror the public API of ../{name}.py without cython decorators/types.
"""
'''

_PURE_CYTHON_STUB = '''\
{language_level_pragma}
"""Pure Cython implementation for {name}.

TODO: hand-write a performant Cython implementation of ../{name}.py's public API.
"""
'''


@dataclass
class ScaffoldResult:
    """Every path a :func:`scaffold_module` call touched, split by outcome."""

    module_path: Path
    created: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)


def _resolve_source(module_path: str) -> Path:
    """Resolve a filesystem path or dotted module name to a .py source path."""
    candidate = Path(module_path)
    if candidate.suffix == ".py":
        return candidate
    return Path(*module_path.split(".")).with_suffix(".py")


def _write_if_absent(path: Path, content: str, *, force: bool, result: ScaffoldResult) -> None:
    if path.exists() and not force:
        result.skipped.append(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    result.created.append(path)


def _ensure_pragma(source: Path, result: ScaffoldResult) -> None:
    if is_augmented(source):
        result.skipped.append(source)
        return
    original = source.read_text()
    source.write_text(f"{_LANGUAGE_LEVEL_PRAGMA}\n{PRAGMA}\n\n{original}")
    result.created.append(source)


def _ensure_pyx_symlink(source: Path, *, force: bool, result: ScaffoldResult) -> None:
    pyx_path = source.with_suffix(".pyx")
    if pyx_path.is_symlink():
        if pyx_path.resolve().name == source.name:
            result.skipped.append(pyx_path)
            return
        if not force:
            result.skipped.append(pyx_path)
            return
        pyx_path.unlink()
    elif pyx_path.exists():
        # A real file already occupies this path -- never clobber it.
        result.skipped.append(pyx_path)
        return
    pyx_path.symlink_to(source.name)
    result.created.append(pyx_path)


def scaffold_module(module_path: str | Path, *, force: bool = False) -> ScaffoldResult:
    """Generate the __pure_python__ / __pure_cython__ triple-layout for an augmented module.

    :param module_path: filesystem path or dotted module name for the augmented .py source
    :param force: overwrite existing pure_python/pure_cython stub files and refresh the
        .pyx symlink. The augmented source itself is never overwritten if it already
        exists -- only its pragma header is added when missing.
    :return: a :class:`ScaffoldResult` listing every path created or left untouched
    """
    source = _resolve_source(str(module_path))
    name = source.stem
    module_dir = source.parent
    result = ScaffoldResult(module_path=source)

    if source.exists():
        _ensure_pragma(source, result)
        pure_python_content = _PURE_PYTHON_STUB.format(name=name)
        pure_cython_content = _PURE_CYTHON_STUB.format(
            name=name, language_level_pragma=_LANGUAGE_LEVEL_PRAGMA
        )
    else:
        _write_if_absent(source, _AUGMENTED_TEMPLATE, force=False, result=result)
        pure_python_content = _PURE_PYTHON_TEMPLATE
        pure_cython_content = _PURE_CYTHON_TEMPLATE

    _write_if_absent(module_dir / "__init__.py", "", force=False, result=result)

    pure_python_dir = module_dir / "__pure_python__"
    _write_if_absent(pure_python_dir / "__init__.py", "", force=False, result=result)
    _write_if_absent(
        pure_python_dir / f"{name}.py", pure_python_content, force=force, result=result
    )

    pure_cython_dir = module_dir / "__pure_cython__"
    _write_if_absent(pure_cython_dir / "__init__.py", "", force=False, result=result)
    _write_if_absent(
        pure_cython_dir / f"{name}.pyx", pure_cython_content, force=force, result=result
    )

    _ensure_pyx_symlink(source, force=force, result=result)

    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaffold the __pure_python__/__pure_cython__ triple-layout for an "
        "augmented Python module."
    )
    parser.add_argument("module_path", help="Path (or dotted name) to the augmented module")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing pure_python/pure_cython stubs and refresh the .pyx symlink "
        "(the augmented source itself is never overwritten)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = scaffold_module(args.module_path, force=args.force)

    if result.created:
        print("Created:")
        for path in result.created:
            print(f"  {path}")
    if result.skipped:
        print("Already present (skipped):")
        for path in result.skipped:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
