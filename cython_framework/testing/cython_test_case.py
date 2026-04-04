"""
Cython Testing Framework
=======================

A framework for testing and comparing different implementations of Python modules:
- Pure Python implementations
- Augmented Pure Python (Cython-optimized Python)
- Compiled versions of Augmented Python
- Pure Cython implementations

Key Components
-------------
CythonTestCase:
    Base test class for synchronous tests of multiple implementations

CythonIsoAsyncioTestCase:
    Base test class for asynchronous tests of multiple implementations

Usage
-----
1. Basic Test Case:
```python
class TestMyModule(CythonTestCase):
    MODULE_PATH = "hummingbot.data_feed.my_module"  # Full path from project root
    MODULE_NAME = "utils"  # Module name without extension

    @cython_test_implementations()
    def test_my_function(self, module, implementation):
        result = module.my_function()
        self.assertEqual(expected, result)

2. Async Test Case:
class TestAsyncModule(CythonIsoAsyncioTestCase):
    MODULE_PATH = "hummingbot.data_feed.my_module"
    MODULE_NAME = "async_utils"

    @async_cython_test_implementations()
    async def test_async_function(self, module, implementation):
        result = await module.async_function()
        self.assertEqual(expected, result)

3. Benchmarking:
class TestPerformance(BenchmarkTestCase):
    MODULE_PATH = "hummingbot.data_feed.my_module"
    MODULE_NAME = "utils"

    def test_performance(self):
        results = self.benchmark_all('my_function')
        for impl, result in results.items():
            print(f"{impl}: {result}")

4. Project Structure:
hummingbot/
    .../
        my_module/
          __init__.py           # Package exports
          utils.py              # Current implementation
          __pure_python__/      # Reference implementation
            __init__.py
            utils.py
          __pure_cython__/      # Performance implementation
            __init__.py
            utils.pyx
"""

import functools
import importlib.util
import logging
import sys
import time
import types
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, ClassVar

from .isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase


def _run_implementation_tests(tests_func, self, results, impl, module, args, kwargs):
    """Run tests for one implementation and store results"""
    try:
        if args or kwargs:
            tests_func(self, module, impl, *args, **kwargs)
        else:
            tests_func(self, module, impl)
        results[impl] = "PASS"
    except AssertionError as e:
        results[impl] = str(e)


def _report_results(results, modules):
    """Report test results for all implementations"""
    print("\nTest Results:")
    for impl, result in results.items():
        print(f"\t{modules[impl][1]}: {result}")


def _check_failures(results, strict_all_variants=False):
    """Check if any tests failed.

    When strict_all_variants is False (default), only raises if
    AUGMENTED_PYTHON is not present or didn't pass AND other variants failed.
    When strict_all_variants is True, raises on ANY variant failure.
    """
    if strict_all_variants:
        for impl, result in results.items():
            if result != "PASS":
                raise AssertionError(f"Failed for {impl}: {result}")
    else:
        if (
            CythonModuleType.AUGMENTED_PYTHON not in results
            or results[CythonModuleType.AUGMENTED_PYTHON] != "PASS"
        ):
            for impl, result in results.items():
                if result != "PASS":
                    raise AssertionError(f"Failed for {impl}: {result}")


def cython_test_implementations(*test_impls):
    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(self, *args, **kwargs):
            results = {}
            impls = test_impls or self.IMPLEMENTATIONS
            for impl in impls:
                if impl in self.modules:
                    _run_implementation_tests(
                        test_func, self, results, impl, self.modules[impl][0], args, kwargs
                    )
            _report_results(results, self.modules)
            _check_failures(results, getattr(self, "STRICT_ALL_VARIANTS", False))

        return wrapper

    return decorator


async def _run_async_implementation_tests(tests_func, self, results, impl, module, args, kwargs):
    """Run async tests for one implementation and store results"""
    try:
        if args or kwargs:
            await tests_func(self, module, impl, *args, **kwargs)
        else:
            await tests_func(self, module, impl)
        results[impl] = "PASS"
    except AssertionError as e:
        results[impl] = str(e)


def async_cython_test_implementations(*test_impls):
    def decorator(test_func):
        @functools.wraps(test_func)
        async def wrapper(self, *args, **kwargs):
            results = {}
            impls = test_impls or self.IMPLEMENTATIONS
            for impl in impls:
                if impl in self.modules:
                    await _run_async_implementation_tests(
                        test_func, self, results, impl, self.modules[impl][0], args, kwargs
                    )
            _report_results(results, self.modules)
            _check_failures(results, getattr(self, "STRICT_ALL_VARIANTS", False))

        return wrapper

    return decorator


class CythonModuleType(Enum):
    """
    Enumeration of supported Cython-related module types.

    PURE_PYTHON: Pure Python implementation
    AUGMENTED_PYTHON: Augmented Python implementation
    COMPILED_AUGMENTED_PYTHON: Compiled Augmented Python implementation
    PURE_CYTHON: Pure Cython implementation
    """

    PURE_PYTHON = auto()  # __pure_python__/module.py
    AUGMENTED_PYTHON = auto()  # module.py
    COMPILED_AUGMENTED_PYTHON = auto()  # module.*.so
    PURE_CYTHON = auto()  # __pure_cython__/module.*.so


class CythonModuleLoader:
    """
    Manages loading of Cython modules for testing.
    Helper class to load Cython modules for CythonTestCase derived classes.
    It is intended to test an Augmented Pure Python module and compare its behavior
    with a Pure Python, Compiled Augmented Python, and Pure Cython implementations.
    The Pure Python implementation is expected to be under
    <module.py dir>/__pure_python__ directory.
    The Pure Cython implementation is expected to be under
    <module.py dir>/__pure_cython__ directory.
    The Compiled Augmented Python implementation is expected to be a .so file in <module.py dir>.

    :param module_path: Path to the module
    :param module_name: Name of the module
    :param pure_python_deps: List of dependency module names to load before
        the main module in __pure_python__ (e.g. ["cython_definitions", "candle_data"])
    """

    def __init__(
        self, module_path: str, module_name: str, pure_python_deps: list[str] | None = None
    ) -> None:
        self.module_path = module_path
        self.module_name = module_name
        self.pure_python_deps = pure_python_deps or []
        self.project_root = self._get_project_root()
        path_parts = self.module_path.split(".")
        self.module_dir = self.project_root / Path(*path_parts)
        self._validate_paths()

    def _get_project_root(self) -> Path:
        """Get the project root by walking up to find pyproject.toml or .git."""
        current = Path(__file__).resolve().parent

        while current != current.parent:
            if (current / "pyproject.toml").exists() or (current / ".git").exists():
                return current
            current = current.parent

        raise FileNotFoundError("Could not find project root (no pyproject.toml or .git found)")

    def _setup_package(self):
        """Setup package structure for imports"""
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        parts = self.module_path.split(".")
        current_pkg = ""
        current_path = self.project_root

        for part in parts:
            current_path = current_path / part
            current_pkg = f"{current_pkg}.{part}" if current_pkg else part

            if current_pkg not in sys.modules:
                module = types.ModuleType(current_pkg)
                module.__path__ = [str(current_path)]
                module.__package__ = current_pkg
                module.__file__ = str(current_path / "__init__.py")
                module.__spec__ = importlib.util.spec_from_file_location(
                    current_pkg, str(current_path / "__init__.py")
                )
                sys.modules[current_pkg] = module

    def _validate_paths(self) -> None:
        if not self.module_dir.exists():
            raise FileNotFoundError(f"Module directory not found: {self.module_dir}")

    def _find_so_file(self, directory: Path) -> Path:
        pattern = f"{self.module_name}.cpython-*.so"
        if matches := list(directory.glob(pattern)):
            return matches[0]
        else:
            raise FileNotFoundError(f"No .so file found matching {pattern} in {directory}")

    def load_module(self, mode: CythonModuleType) -> tuple[Any, str]:
        """
        Load the specified module type.

        :param mode: Module type to load
        :return: Tuple of loaded module and implementation name
        """
        self._setup_package()
        loaders = {
            CythonModuleType.PURE_PYTHON: self._load_from_pure_python,
            CythonModuleType.AUGMENTED_PYTHON: self._load_augmented_python,
            CythonModuleType.COMPILED_AUGMENTED_PYTHON: self._load_augmented_python_compiled,
            CythonModuleType.PURE_CYTHON: self._load_from_pure_cython,
        }
        return loaders[mode]()

    def _load_from_pure_python(self) -> tuple[Any, str]:
        """Load current implementation as Pure Python"""
        pure_py_dir = self.module_dir / "__pure_python__"
        if not pure_py_dir.exists():
            raise FileNotFoundError(f"Pure Python directory not found: {pure_py_dir}")

        # Register __pure_python__ package and make its modules available
        pure_pkg = f"{self.module_path}.__pure_python__"

        # First load any dependencies (configurable via PURE_PYTHON_DEPS)
        for dep in self.pure_python_deps:
            dep_path = pure_py_dir / f"{dep}.py"
            if dep_path.exists():
                dep_module = self._import_from_path(
                    dep_path,
                    variant=CythonModuleType.PURE_PYTHON,
                    module_name_override=dep,
                )
                sys.modules[f"{pure_pkg}.{dep}"] = dep_module

        py_path = pure_py_dir / f"{self.module_name}.py"
        if not py_path.exists():
            raise FileNotFoundError(f"Pure Python module not found: {py_path}")

        return self._import_from_path(py_path, variant=CythonModuleType.PURE_PYTHON), "Pure Python"

    def _load_augmented_python(self) -> tuple[Any, str]:
        """Load current implementation as Python, bypassing any .so files.

        Uses importlib.util.spec_from_file_location with an explicit .py
        path so the .so file is never considered, avoiding the fragile
        rename-to-.so.bak trick.
        """
        py_path = self.module_dir / f"{self.module_name}.py"
        if not py_path.exists():
            raise FileNotFoundError(f"Augmented Python source not found: {py_path}")
        return self._import_from_path(
            py_path, variant=CythonModuleType.AUGMENTED_PYTHON
        ), "Augmented Python"

    def _load_augmented_python_compiled(self) -> tuple[Any, str]:
        """Load current implementation as Compiled Augmented Python"""
        so_path = self._find_so_file(self.module_dir)
        return self._import_from_path(
            so_path, variant=CythonModuleType.COMPILED_AUGMENTED_PYTHON
        ), "Compiled Augmented Python"

    def _load_from_pure_cython(self) -> tuple[Any, str]:
        """Load current implementation as Pure Cython"""
        cython_dir = self.module_dir / "__pure_cython__"
        if not cython_dir.exists():
            raise FileNotFoundError(f"Pure Cython directory not found: {cython_dir}")
        so_path = self._find_so_file(cython_dir)
        sys.modules.pop(self.module_name, None)
        return self._import_from_path(so_path, variant=CythonModuleType.PURE_CYTHON), "Pure Cython"

    def _import_from_path(
        self,
        path: Path,
        *,
        variant: CythonModuleType | None = None,
        module_name_override: str | None = None,
    ) -> Any:
        """Import module with proper package context.

        Uses namespaced sys.modules keys per variant to avoid collisions
        between different implementations of the same module.
        """
        parent_pkg = self.module_path
        mod_name = module_name_override or self.module_name

        # Namespace sys.modules keys per variant to avoid collisions
        if variant is not None:
            full_module_name = f"{parent_pkg}.__{variant.name}__.{mod_name}"
        else:
            full_module_name = f"{parent_pkg}.{mod_name}"

        spec = importlib.util.spec_from_file_location(full_module_name, path)
        if not spec or not spec.loader:
            raise ImportError(f"Failed to load {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[full_module_name] = module
        module.__package__ = parent_pkg

        spec.loader.exec_module(module)
        return module


class CythonTestMixin:
    MODULE_PATH: ClassVar[str]
    MODULE_NAME: ClassVar[str]
    IMPLEMENTATIONS: ClassVar[list[CythonModuleType]] = [
        CythonModuleType.AUGMENTED_PYTHON,
        CythonModuleType.COMPILED_AUGMENTED_PYTHON,
        CythonModuleType.PURE_PYTHON,
        CythonModuleType.PURE_CYTHON,
    ]
    PURE_PYTHON_DEPS: ClassVar[list[str]] = []
    STRICT_ALL_VARIANTS: ClassVar[bool] = False

    @classmethod
    def _check_class_attributes(cls) -> None:
        if not hasattr(cls, "MODULE_PATH") or not hasattr(cls, "MODULE_NAME"):
            raise NotImplementedError("Define MODULE_PATH and MODULE_NAME")

    def _init_module_loader(self) -> None:
        self.loader = CythonModuleLoader(
            self.MODULE_PATH,
            self.MODULE_NAME,
            pure_python_deps=self.PURE_PYTHON_DEPS,
        )
        self.modules = {}
        print("\n")
        for impl in self.IMPLEMENTATIONS:
            try:
                module, impl_name = self.loader.load_module(impl)
                self.modules[impl] = (module, impl_name)
            except (FileNotFoundError, ImportError) as e:
                print(f"\tSkipped: {impl}")
                logging.warning(f"Failed to load {impl}: {e}")


class CythonTestCase(unittest.TestCase, CythonTestMixin):
    """
    Base test case class for testing Cython-related modules.
    It is intended to be used with a CythonModuleLoader instance.
    The derived class should define MODULE_PATH and MODULE_NAME.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._check_class_attributes()
        super().setUpClass()

    def setUp(self) -> None:
        super().setUp()
        self._init_module_loader()


class CythonIsoAsyncioTestCase(IsolatedAsyncioWrapperTestCase, CythonTestMixin):
    """
    Base async test case class for testing Cython-related modules.
    It is intended to be used with a CythonModuleLoader instance.
    The derived class should define MODULE_PATH and MODULE_NAME.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._check_class_attributes()
        super().setUpClass()

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self._init_module_loader()


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    implementation: str
    total_time: float
    runs: int
    avg_time: float

    def __str__(self) -> str:
        return (
            f"{self.implementation}:\n"
            f"  Total time: {self.total_time:.3f}s\n"
            f"  Average time: {self.avg_time * 1000:.3f}ms per run\n"
            f"  Runs: {self.runs}"
        )


class BenchmarkMixin:
    """Mixin providing benchmarking capabilities to test cases."""

    def run_benchmark(
        self,
        func: Callable,
        *args,
        num_runs: int = 10000,
        warmup_runs: int = 100,
        implementation: str = "unknown",
        **kwargs,
    ) -> BenchmarkResult:
        """
        Run a benchmark on the specified function.

        :param func: Function to benchmark
        :param args: Positional arguments to pass to the function
        :param num_runs: Number of benchmark iterations
        :param warmup_runs: Number of warmup runs before timing
        :param kwargs: Keyword arguments to pass to the function
        :return: Benchmark results
        """
        # Warmup runs
        for _ in range(warmup_runs):
            func(*args, **kwargs)

        # Timed runs
        start_time = time.perf_counter()
        for _ in range(num_runs):
            func(*args, **kwargs)
        total_time = time.perf_counter() - start_time

        return BenchmarkResult(
            implementation=implementation,
            total_time=total_time,
            runs=num_runs,
            avg_time=total_time / num_runs,
        )


class BenchmarkTestCase(CythonTestCase, BenchmarkMixin):
    def benchmark_all(
        self, func_name: str, *args, **kwargs
    ) -> dict[CythonModuleType, BenchmarkResult]:
        results = {}
        for impl, (module, impl_name) in self.modules.items():
            func = getattr(module, func_name)
            result = self.run_benchmark(func, *args, **kwargs)
            result.implementation = impl_name
            results[impl] = result
        return results
