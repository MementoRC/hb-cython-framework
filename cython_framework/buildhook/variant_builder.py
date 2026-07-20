"""Build all four CythonModuleType variants from a single augmented .py source.

Generalizes the PATH-B build recipe that originally lived inline in
``tests/unit/test_cython_compilation.py::_cythonize_file`` into reusable
library code. Output paths match exactly what
:class:`cython_framework.testing.cython_test_case.CythonModuleLoader` reads
for each variant:

- ``AUGMENTED_PYTHON``            -> ``<build_dir>/<module_name>.py``
- ``PURE_PYTHON``                 -> ``<build_dir>/__pure_python__/<module_name>.py``
- ``COMPILED_AUGMENTED_PYTHON``   -> ``<build_dir>/<module_name>.cpython-*.so``
- ``PURE_CYTHON``                 -> ``<build_dir>/__pure_cython__/<module_name>.cpython-*.so``
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

from cython_framework.testing.cython_test_case import CythonModuleType

_ALL_VARIANTS: tuple[CythonModuleType, ...] = (
    CythonModuleType.PURE_PYTHON,
    CythonModuleType.AUGMENTED_PYTHON,
    CythonModuleType.COMPILED_AUGMENTED_PYTHON,
    CythonModuleType.PURE_CYTHON,
)


class VariantBuildError(RuntimeError):
    """Raised when cythonizing or compiling a variant fails."""


class VariantArtifacts(dict[CythonModuleType, Path]):
    """Mapping of variant -> artifact path, with a discoverable build root.

    Behaves exactly like the plain ``dict`` this function used to return.
    ``build_root`` additionally exposes the directory all artifacts were
    written under -- most useful when ``build_dir=None`` was passed to
    :func:`build_variants`, so the caller can find (and later remove) the
    temp directory that was created on its behalf.
    """

    build_root: Path


def _has_c_compiler() -> bool:
    """Check if a C compiler is available."""
    if platform.system() == "Windows":
        return shutil.which("cl") is not None
    return shutil.which("gcc") is not None or shutil.which("cc") is not None


def _so_glob_pattern(module_name: str) -> str:
    """Glob pattern for a compiled extension.

    Matches ``CythonModuleLoader._find_so_file`` exactly so that "builder
    reports success" implies "loader can load it".
    """
    return f"{module_name}.cpython-*.so"


def _numpy_include_dir() -> str | None:
    """Return numpy's include dir if numpy is importable in this env, else None.

    Keeps numpy an optional build dependency: a module with no numpy
    dependency must still compile when numpy is absent.
    """
    try:
        import numpy
    except ImportError:
        return None
    return numpy.get_include()


def _compile_to_so(source_py: Path, dest_dir: Path, module_name: str) -> Path:
    """Cythonize + compile ``source_py`` into ``dest_dir``.

    :return: path to the resulting compiled extension
    :raises VariantBuildError: if cythonizing or compiling fails
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_py = dest_dir / f"{module_name}.py"
    shutil.copy2(source_py, dest_py)

    pyx_path = dest_py.with_suffix(".pyx")
    if pyx_path.exists():
        pyx_path.unlink()
    try:
        pyx_path.symlink_to(dest_py.name)
    except OSError:
        # e.g. Windows without symlink privilege: degrade to a plain copy.
        shutil.copy2(dest_py, pyx_path)

    result = subprocess.run(
        [sys.executable, "-m", "cython", "--3str", str(pyx_path)],
        capture_output=True,
        text=True,
        cwd=str(dest_dir),
    )
    if result.returncode != 0:
        raise VariantBuildError(f"Cython compilation failed for {module_name}: {result.stderr}")

    c_path = pyx_path.with_suffix(".c")
    if not c_path.exists():
        raise VariantBuildError(f"Cython did not produce a .c file for {module_name}")

    numpy_include = _numpy_include_dir()
    ext_kwargs = ""
    if numpy_include is not None:
        ext_kwargs = (
            f"\n    include_dirs=[{numpy_include!r}],"
            '\n    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],'
        )

    setup_py = dest_dir / "_setup_compile.py"
    setup_py.write_text(f"""\
from setuptools import Extension, setup

ext = Extension(
    "{module_name}",
    sources=["{c_path.name}"],{ext_kwargs}
)
setup(
    name="{module_name}",
    ext_modules=[ext],
    packages=[],
    py_modules=[],
    script_args=["build_ext", "--inplace"],
)
""")

    result = subprocess.run(
        [sys.executable, str(setup_py)],
        capture_output=True,
        text=True,
        cwd=str(dest_dir),
    )
    if result.returncode != 0:
        raise VariantBuildError(f"C compilation failed for {module_name}: {result.stderr}")

    so_files = list(dest_dir.glob(_so_glob_pattern(module_name)))
    if not so_files:
        raise VariantBuildError(f"No compiled extension found for {module_name} after build")
    return so_files[0]


def _build_augmented_python(source_py: Path, build_dir: Path, module_name: str) -> Path:
    dest = build_dir / f"{module_name}.py"
    shutil.copy2(source_py, dest)
    return dest


def _build_pure_python(source_py: Path, build_dir: Path, module_name: str) -> Path:
    # Verbatim copy is intentional. In the augmented-pure-python pattern the
    # same source runs either interpreted or compiled: `cython.cclass`,
    # `cython.ccall`, etc. degrade to no-op identity decorators when the
    # module is not compiled, so `import cython` here is safe. `cython` is a
    # hard dependency of this framework, and the AUGMENTED_PYTHON variant
    # (identical content, loaded uncompiled from a different path) already
    # exercises this exact code path as the cross-variant baseline.
    pure_dir = build_dir / "__pure_python__"
    pure_dir.mkdir(parents=True, exist_ok=True)
    dest = pure_dir / f"{module_name}.py"
    shutil.copy2(source_py, dest)
    return dest


def _build_compiled_augmented_python(source_py: Path, build_dir: Path, module_name: str) -> Path:
    if not _has_c_compiler():
        raise VariantBuildError("No C compiler available for COMPILED_AUGMENTED_PYTHON")
    return _compile_to_so(source_py, build_dir, module_name)


def _build_pure_cython(
    source_py: Path,
    build_dir: Path,
    module_name: str,
    reuse_so: Path | None = None,
) -> Path:
    if not _has_c_compiler():
        raise VariantBuildError("No C compiler available for PURE_CYTHON")
    dest_dir = build_dir / "__pure_cython__"
    if reuse_so is not None:
        # COMPILED_AUGMENTED_PYTHON already compiled this identical
        # augmented source in this same build_variants() call -- copy its
        # .so instead of cythonizing + compiling a second time. The
        # filename is preserved, so it still matches the loader's expected
        # `{module_name}.cpython-*.so` pattern.
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / reuse_so.name
        shutil.copy2(reuse_so, dest)
        return dest
    return _compile_to_so(source_py, dest_dir, module_name)


_BUILDERS = {
    CythonModuleType.AUGMENTED_PYTHON: _build_augmented_python,
    CythonModuleType.PURE_PYTHON: _build_pure_python,
}


def _remove_new_entries(directory: Path, pre_existing: set[Path]) -> None:
    """Remove entries in ``directory`` that were not present before this build."""
    for entry in directory.iterdir():
        if entry in pre_existing:
            continue
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)


def build_variants(
    module_path: str | Path,
    module_name: str | None = None,
    variants: Iterable[CythonModuleType] | None = None,
    build_dir: str | Path | None = None,
) -> VariantArtifacts:
    """Build the requested CythonModuleType variants from one augmented .py source.

    All-or-nothing: if any variant fails to build, every file this call
    wrote (across all variants) is removed before the exception
    propagates -- no partial builds are left on disk.

    :param module_path: path to the augmented .py source file
    :param module_name: module name, defaults to ``module_path.stem``
    :param variants: which CythonModuleType variants to build, defaults to all four
    :param build_dir: directory to build into. If ``None``, a fresh temp
        directory is created (``tempfile.mkdtemp``) and used as the build
        root. On success that temp directory is left on disk -- its
        artifacts are what the caller's ``CythonModuleLoader`` needs to
        read -- and this function does NOT clean it up. **The caller owns
        that temp directory and is responsible for removing it** (e.g.
        ``shutil.rmtree(result.build_root)``) once done. On failure it IS
        removed, along with everything else this call wrote.
    :return: a dict-like :class:`VariantArtifacts` mapping CythonModuleType
        to the on-disk Path a CythonModuleLoader rooted at ``build_dir``
        will read for that variant; ``.build_root`` exposes the directory
        everything was written under.
    :raises FileNotFoundError: if ``module_path`` does not exist
    :raises VariantBuildError: if cythonizing or compiling a variant fails
    """
    source_py = Path(module_path)
    if not source_py.exists():
        raise FileNotFoundError(f"Augmented Python source not found: {source_py}")

    name = module_name or source_py.stem
    wanted = tuple(variants) if variants is not None else _ALL_VARIANTS

    if build_dir is None:
        build_root = Path(tempfile.mkdtemp(prefix="variant_build_"))
        full_cleanup_on_failure = True
        pre_existing: set[Path] = set()
    else:
        build_root = Path(build_dir)
        root_pre_existed = build_root.exists()
        build_root.mkdir(parents=True, exist_ok=True)
        full_cleanup_on_failure = not root_pre_existed
        pre_existing = set(build_root.iterdir()) if root_pre_existed else set()

    # Build COMPILED_AUGMENTED_PYTHON before PURE_CYTHON (if both are
    # requested) so PURE_CYTHON can reuse the compiled .so instead of
    # cythonizing + compiling the identical source twice.
    ordered = sorted(
        wanted, key=lambda v: 0 if v is CythonModuleType.COMPILED_AUGMENTED_PYTHON else 1
    )

    results: dict[CythonModuleType, Path] = {}
    compiled_so: Path | None = None
    try:
        for variant in ordered:
            if variant is CythonModuleType.COMPILED_AUGMENTED_PYTHON:
                results[variant] = _build_compiled_augmented_python(source_py, build_root, name)
                compiled_so = results[variant]
            elif variant is CythonModuleType.PURE_CYTHON:
                results[variant] = _build_pure_cython(
                    source_py, build_root, name, reuse_so=compiled_so
                )
            else:
                results[variant] = _BUILDERS[variant](source_py, build_root, name)
    except Exception:
        if full_cleanup_on_failure:
            shutil.rmtree(build_root, ignore_errors=True)
        else:
            _remove_new_entries(build_root, pre_existing)
        raise

    artifacts = VariantArtifacts({variant: results[variant] for variant in wanted})
    artifacts.build_root = build_root
    return artifacts
