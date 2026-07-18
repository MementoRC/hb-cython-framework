"""Env-gated, pragma-driven Cython build hook for augmented pure-Python modules.

Eligibility is decided per file by the augmented pragma
(``# cython: augmented_pure_python=True``) -- there is no central manifest.

Compilation is activated only when ``HB_COMPILE_AUGMENTED`` is truthy
(default OFF). When OFF the hook is a strict no-op: it never scans, never imports
Cython, and never invokes a C compiler, so bleeding-edge and ordinary CI can
build a pure-Python wheel with no toolchain present.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.plugin import hookimpl

# Single source of truth for pragma detection (reused from the pre-commit hook).
from cython_framework.hooks.link_augmented_pyx import is_augmented

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_ENV_VAR = "HB_COMPILE_AUGMENTED"
_SKIP_DIRS = frozenset(
    {
        "__pure_python__",
        "__pure_cython__",
        ".git",
        ".pixi",
        "build",
        "dist",
        ".tox",
        ".venv",
        "__pycache__",
    }
)


def compilation_enabled(env=None) -> bool:
    """Return True iff ``HB_COMPILE_AUGMENTED`` is set to a truthy value."""
    env = os.environ if env is None else env
    return env.get(_ENV_VAR, "").strip().lower() in _TRUTHY


def discover_augmented(base) -> list[Path]:
    """Return every ``.py`` under *base* carrying the augmented pragma.

    Eligibility is per file (the pragma) -- never a central manifest. Variant
    fallback directories (``__pure_python__``/``__pure_cython__``) and build
    scratch dirs are skipped.
    """
    base = Path(base)
    if not base.exists():
        return []
    out: list[Path] = []
    for py in sorted(base.rglob("*.py")):
        if _SKIP_DIRS & set(py.parts):
            continue
        if is_augmented(py):
            out.append(py)
    return out


def _module_name(py: Path, root: Path) -> str:
    return ".".join(py.relative_to(root).with_suffix("").parts)


def _find_artifact(py: Path):
    # An in-place build drops ``<stem>.<abi>.so`` (or ``.pyd``) next to the source.
    for pattern in (f"{py.stem}.*.so", f"{py.stem}.*.pyd"):
        matches = sorted(py.parent.glob(pattern))
        if matches:
            return matches[-1]
    return None


class AugmentedCythonBuildHook(BuildHookInterface):
    """Compile pragma-eligible augmented modules only when explicitly enabled."""

    PLUGIN_NAME = "augmented-cython"

    def initialize(self, version, build_data):
        if not compilation_enabled():
            # No-op: pure-Python build. Cython/setuptools are never imported and
            # no C compiler is required.
            return

        root = Path(self.root)
        scan_paths = self.config.get("paths") or ["."]
        targets: list[Path] = []
        for rel in scan_paths:
            targets.extend(discover_augmented(root / rel))
        seen: set[Path] = set()
        targets = [t for t in targets if not (t in seen or seen.add(t))]
        if not targets:
            self.app.display_warning(
                f"{_ENV_VAR} is set but no augmented modules (pragma) were found "
                f"under {scan_paths}; producing a pure-Python wheel."
            )
            return

        artifacts = self._compile(targets, root)
        if not artifacts:
            return

        build_data["pure_python"] = False
        build_data["infer_tag"] = True
        force_include = build_data.setdefault("force_include", {})
        for so_path, rel_target in artifacts:
            force_include[str(so_path)] = rel_target

    def _compile(self, targets, root):
        # Deferred imports: required only on the ON path, so the OFF path never
        # needs Cython/setuptools/a C compiler to be importable.
        import numpy
        from Cython.Build import cythonize
        from setuptools import Distribution, Extension
        from setuptools.command.build_ext import build_ext as _build_ext

        directives = {
            "boundscheck": False,
            "nonecheck": False,
            "language_level": "3str",
            "binding": True,
        }
        macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
        include_dirs = [numpy.get_include()]
        extra_compile_args = ["-O2"] if sys.platform in ("linux", "darwin") else []

        extensions = [
            Extension(
                _module_name(py, root),
                [str(py)],
                define_macros=macros,
                include_dirs=include_dirs,
                extra_compile_args=extra_compile_args,
            )
            for py in targets
        ]
        ext_modules = cythonize(extensions, compiler_directives=directives, force=True)

        dist = Distribution({"ext_modules": ext_modules})
        cmd = _build_ext(dist)
        cmd.inplace = True
        cmd.ensure_finalized()
        cmd.run()

        artifacts = []
        for py in targets:
            so = _find_artifact(py)
            if so is None:
                raise RuntimeError(
                    f"{_ENV_VAR} is set but no compiled artifact was produced for "
                    f"{py.relative_to(root)}"
                )
            artifacts.append((so, str(so.relative_to(root))))
        return artifacts


@hookimpl
def hatch_register_build_hook():
    return AugmentedCythonBuildHook
