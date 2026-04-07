"""Validation helpers for Cython migration testing.

Provides base test classes that extend the core testing framework with
equivalence and performance assertion utilities.
"""

from __future__ import annotations

import inspect
from typing import Any, ClassVar

import numpy.testing

from cython_framework.testing.cython_test_case import BenchmarkTestCase, CythonTestCase


class ValidationTestCase(CythonTestCase):
    """Base test case with cross-implementation equivalence assertions."""

    def assert_implementations_equal(self, func_name: str, *args: Any, **kwargs: Any) -> None:
        """Assert that all loaded implementations return identical results.

        Calls *func_name* with the given arguments across every loaded
        implementation and raises ``AssertionError`` if any two results differ.
        """
        results: dict[str, Any] = {}
        for _impl, (module, impl_name) in self.modules.items():
            func = getattr(module, func_name)
            results[impl_name] = func(*args, **kwargs)

        values = list(results.values())
        if len(values) < 2:
            return

        reference_name, reference_value = next(iter(results.items())), values[0]
        for impl_name, value in list(results.items())[1:]:
            self.assertEqual(
                reference_value,
                value,
                msg=(
                    f"{func_name}: result mismatch between "
                    f"'{reference_name}' ({reference_value!r}) and "
                    f"'{impl_name}' ({value!r})"
                ),
            )

    def assert_implementations_close(
        self,
        func_name: str,
        *args: Any,
        rtol: float = 1e-7,
        atol: float = 0,
        **kwargs: Any,
    ) -> None:
        """Assert that all loaded implementations return numerically close results.

        Uses ``numpy.testing.assert_allclose`` for float comparison.
        """
        results: dict[str, Any] = {}
        for _impl, (module, impl_name) in self.modules.items():
            func = getattr(module, func_name)
            results[impl_name] = func(*args, **kwargs)

        values = list(results.values())
        if len(values) < 2:
            return

        impl_names = list(results.keys())
        reference_name = impl_names[0]
        reference_value = values[0]

        for impl_name, value in zip(impl_names[1:], values[1:], strict=True):
            numpy.testing.assert_allclose(
                actual=value,
                desired=reference_value,
                rtol=rtol,
                atol=atol,
                err_msg=(
                    f"{func_name}: result mismatch between "
                    f"'{reference_name}' ({reference_value!r}) and "
                    f"'{impl_name}' ({value!r})"
                ),
            )

    def assert_signature_matches(self, func_name: str) -> None:
        """Assert that parameter names match across all loaded implementations.

        Compares the parameter names (not annotations or defaults) of
        *func_name* across every loaded implementation.
        """
        signatures: dict[str, list[str]] = {}
        for _impl, (module, impl_name) in self.modules.items():
            func = getattr(module, func_name)
            sig = inspect.signature(func)
            signatures[impl_name] = list(sig.parameters.keys())

        if len(signatures) < 2:
            return

        impl_names = list(signatures.keys())
        reference_name = impl_names[0]
        reference_params = signatures[reference_name]

        for impl_name, params in zip(impl_names[1:], list(signatures.values())[1:], strict=True):
            self.assertEqual(
                reference_params,
                params,
                msg=(
                    f"{func_name}: signature mismatch between "
                    f"'{reference_name}' ({reference_params}) and "
                    f"'{impl_name}' ({params})"
                ),
            )


class PerformanceTestCase(BenchmarkTestCase):
    """Base test case with performance threshold assertions.

    Set ``MANIFEST_PATH`` to a manifest JSON path to enable threshold checks.
    """

    MANIFEST_PATH: ClassVar[str | None] = None

    def assert_within_threshold(self, func_name: str, *args: Any, **kwargs: Any) -> None:
        """Compare benchmark results against manifest thresholds.

        Skips if ``MANIFEST_PATH`` is not set.  Reads the manifest lazily so
        the import only occurs when actually needed.
        """
        if self.MANIFEST_PATH is None:
            self.skipTest("No MANIFEST_PATH configured; skipping threshold check.")
            return

        from cython_framework.manifest import load as manifest_load  # noqa: PLC0415

        manifest = manifest_load(self.MANIFEST_PATH)
        results = self.benchmark_all(func_name, *args, **kwargs)

        for _impl, benchmark_result in results.items():
            impl_name = benchmark_result.implementation
            entry = manifest.modules.get(self.MODULE_NAME)
            if entry is None:
                self.skipTest(f"Module '{self.MODULE_NAME}' not found in manifest.")
                return

            threshold = getattr(entry, "performance_threshold_ms", None)
            if threshold is None:
                continue

            avg_ms = benchmark_result.avg_time * 1000
            self.assertLessEqual(
                avg_ms,
                threshold,
                msg=(
                    f"{func_name} [{impl_name}]: avg {avg_ms:.3f}ms exceeds "
                    f"threshold {threshold:.3f}ms"
                ),
            )
