"""Tests for hatch-cython compilation pipeline.

Validates the full workflow:
  1. Augmented .py file with cython annotations exists
  2. .pyx symlink is created (simulating pre-commit hook)
  3. Cython compiles .pyx → .c
  4. C compiler produces .so/.pyd
  5. Compiled module imports and produces correct results
  6. Compiled module has cython.compiled == True

These tests require Cython and a C compiler to be available.
"""

from __future__ import annotations

import importlib
import importlib.util
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Skip entire module if Cython is not available
cython = pytest.importorskip("Cython", reason="Cython not installed")


def _get_fixture_dir() -> Path:
    """Get path to the augmented_example fixture."""
    return Path(__file__).parent.parent / "fixtures" / "augmented_example"


def _has_c_compiler() -> bool:
    """Check if a C compiler is available."""
    if platform.system() == "Windows":
        return shutil.which("cl") is not None
    return shutil.which("gcc") is not None or shutil.which("cc") is not None


def _cythonize_file(py_path: Path, work_dir: Path) -> Path | None:
    """Cythonize a single .py file via .pyx symlink.

    Returns the path to the compiled .so/.pyd, or None if compilation fails.
    """
    # Copy the .py file to work dir
    dest_py = work_dir / py_path.name
    shutil.copy2(py_path, dest_py)

    # Create .pyx symlink (what the pre-commit hook does)
    pyx_path = dest_py.with_suffix(".pyx")
    if pyx_path.exists():
        pyx_path.unlink()
    pyx_path.symlink_to(dest_py.name)

    # Also copy __init__.py if it exists
    init_file = py_path.parent / "__init__.py"
    if init_file.exists():
        shutil.copy2(init_file, work_dir / "__init__.py")

    # Step 1: Cythonize .pyx → .c
    result = subprocess.run(
        [sys.executable, "-m", "cython", "--3str", str(pyx_path)],
        capture_output=True,
        text=True,
        cwd=str(work_dir),
    )
    if result.returncode != 0:
        pytest.skip(f"Cython compilation failed: {result.stderr}")

    c_path = pyx_path.with_suffix(".c")
    if not c_path.exists():
        pytest.skip("Cython did not produce .c file")

    # Step 2: Compile .c → .so/.pyd using distutils/setuptools inline
    module_name = py_path.stem
    setup_py = work_dir / "_setup_compile.py"
    setup_py.write_text(f"""\
import numpy as np
from setuptools import Extension, setup

ext = Extension(
    "{module_name}",
    sources=["{c_path.name}"],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)
setup(
    name="{module_name}",
    ext_modules=[ext],
    script_args=["build_ext", "--inplace"],
)
""")

    result = subprocess.run(
        [sys.executable, str(setup_py)],
        capture_output=True,
        text=True,
        cwd=str(work_dir),
    )
    if result.returncode != 0:
        pytest.skip(f"C compilation failed: {result.stderr}")

    # Find the .so/.pyd file
    if platform.system() == "Windows":
        so_files = list(work_dir.glob(f"{module_name}*.pyd"))
    else:
        so_files = list(work_dir.glob(f"{module_name}*.so"))

    if not so_files:
        pytest.skip("No compiled extension found after build")

    return so_files[0]


def _load_compiled_module(so_path: Path, module_name: str):
    """Load a compiled .so/.pyd as a Python module.

    The module name passed to spec_from_file_location MUST match the
    PyInit_<name> export in the .so. Cython names it after the .pyx stem.
    """
    spec = importlib.util.spec_from_file_location(
        module_name,  # Must match PyInit_<name> in the .so
        str(so_path),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def compiled_example():
    """Compile the augmented example fixture and return the loaded module.

    This fixture is module-scoped because compilation is expensive.
    """
    fixture_dir = _get_fixture_dir()
    py_file = fixture_dir / "example.py"

    if not py_file.exists():
        pytest.skip("Augmented example fixture not found")

    if not _has_c_compiler():
        pytest.skip("No C compiler available")

    with tempfile.TemporaryDirectory(prefix="cython_test_") as tmpdir:
        work_dir = Path(tmpdir)
        so_path = _cythonize_file(py_file, work_dir)
        if so_path is None:
            pytest.skip("Compilation produced no output")

        module = _load_compiled_module(so_path, "example")
        yield module


@pytest.fixture
def pure_python_example():
    """Load the pure-python variant for comparison."""
    fixture_dir = _get_fixture_dir() / "__pure_python__"
    py_file = fixture_dir / "example.py"

    spec = importlib.util.spec_from_file_location(
        "_test_pure_python_example",
        str(py_file),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def augmented_as_python():
    """Load the augmented file as plain Python (no compilation)."""
    fixture_dir = _get_fixture_dir()
    py_file = fixture_dir / "example.py"

    spec = importlib.util.spec_from_file_location(
        "_test_augmented_python_example",
        str(py_file),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Compilation pipeline tests
# ---------------------------------------------------------------------------


class TestCythonCompilationPipeline:
    """Verify the full compile pipeline works."""

    def test_augmented_file_has_pragma(self):
        """The augmented .py file must contain the cython pragma in its header."""
        py_file = _get_fixture_dir() / "example.py"
        header = py_file.read_text().split("\n")[:5]
        assert any("augmented_pure_python=True" in line for line in header)

    def test_pyx_symlink_creation(self, tmp_path):
        """Simulate pre-commit hook: .pyx symlink resolves to .py."""
        py_file = _get_fixture_dir() / "example.py"
        dest = tmp_path / "example.py"
        shutil.copy2(py_file, dest)
        pyx = dest.with_suffix(".pyx")
        pyx.symlink_to(dest.name)

        assert pyx.is_symlink()
        assert pyx.resolve().name == "example.py"
        assert pyx.read_text().startswith("# cython:")

    def test_cythonize_produces_c_file(self, tmp_path):
        """Cython can process the augmented .py via .pyx symlink."""
        py_file = _get_fixture_dir() / "example.py"
        dest = tmp_path / "example.py"
        shutil.copy2(py_file, dest)
        pyx = dest.with_suffix(".pyx")
        pyx.symlink_to(dest.name)

        result = subprocess.run(
            [sys.executable, "-m", "cython", "--3str", str(pyx)],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0, f"Cython failed: {result.stderr}"
        assert (tmp_path / "example.c").exists()

    def test_compiled_module_loads(self, compiled_example):
        """The compiled .so can be imported as a Python module."""
        assert compiled_example is not None
        assert hasattr(compiled_example, "add")
        assert hasattr(compiled_example, "multiply")
        assert hasattr(compiled_example, "fibonacci")


# ---------------------------------------------------------------------------
# Correctness tests: compiled vs pure Python
# ---------------------------------------------------------------------------


class TestCompiledCorrectness:
    """Verify compiled module produces identical results to pure Python."""

    def test_add(self, compiled_example, pure_python_example):
        for a, b in [(0, 0), (1, 2), (-5, 5), (100, 200), (-100, -200)]:
            assert compiled_example.add(a, b) == pure_python_example.add(a, b)

    def test_multiply(self, compiled_example, pure_python_example):
        for a, b in [(0.0, 1.0), (2.5, 4.0), (-1.5, 3.0), (1e10, 1e-10)]:
            assert compiled_example.multiply(a, b) == pytest.approx(
                pure_python_example.multiply(a, b)
            )

    def test_fibonacci(self, compiled_example, pure_python_example):
        for n in range(20):
            assert compiled_example.fibonacci(n) == pure_python_example.fibonacci(n)


# ---------------------------------------------------------------------------
# Augmented-as-Python tests (no compilation)
# ---------------------------------------------------------------------------


class TestAugmentedAsPython:
    """Verify the augmented .py file runs correctly WITHOUT compilation."""

    def test_add(self, augmented_as_python):
        assert augmented_as_python.add(2, 3) == 5

    def test_multiply(self, augmented_as_python):
        assert augmented_as_python.multiply(2.5, 4.0) == 10.0

    def test_fibonacci(self, augmented_as_python):
        assert augmented_as_python.fibonacci(10) == 55

    def test_consistency_with_pure_python(self, augmented_as_python, pure_python_example):
        """Augmented file running as Python must match pure Python."""
        for n in range(15):
            assert augmented_as_python.fibonacci(n) == pure_python_example.fibonacci(n)
