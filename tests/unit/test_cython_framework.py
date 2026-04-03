"""Meta-tests for the Cython test framework itself.

Validates that CythonTestCase, module loading, and the
@cython_test_implementations decorator work correctly using
a minimal augmented example module.
"""

from cython_framework.testing import CythonModuleType, CythonTestCase, cython_test_implementations


class TestCythonFrameworkLoading(CythonTestCase):
    """Test that the framework can load augmented and pure-python variants."""

    MODULE_PATH = "tests.fixtures.augmented_example"
    MODULE_NAME = "example"
    IMPLEMENTATIONS = [
        CythonModuleType.AUGMENTED_PYTHON,
        CythonModuleType.PURE_PYTHON,
    ]

    def test_modules_loaded(self):
        """At minimum, AUGMENTED_PYTHON and PURE_PYTHON should load."""
        self.assertIn(CythonModuleType.AUGMENTED_PYTHON, self.modules)
        self.assertIn(CythonModuleType.PURE_PYTHON, self.modules)

    @cython_test_implementations()
    def test_add(self, module, implementation):
        self.assertEqual(module.add(2, 3), 5)
        self.assertEqual(module.add(-1, 1), 0)
        self.assertEqual(module.add(0, 0), 0)

    @cython_test_implementations()
    def test_multiply(self, module, implementation):
        self.assertAlmostEqual(module.multiply(2.5, 4.0), 10.0)
        self.assertAlmostEqual(module.multiply(0.0, 100.0), 0.0)

    @cython_test_implementations()
    def test_fibonacci(self, module, implementation):
        self.assertEqual(module.fibonacci(0), 0)
        self.assertEqual(module.fibonacci(1), 1)
        self.assertEqual(module.fibonacci(10), 55)
        self.assertEqual(module.fibonacci(20), 6765)

    @cython_test_implementations(CythonModuleType.PURE_PYTHON)
    def test_selective_implementation(self, module, implementation):
        """Test that decorator can target a specific implementation."""
        self.assertEqual(implementation, CythonModuleType.PURE_PYTHON)
        self.assertEqual(module.add(1, 1), 2)


class TestCythonFrameworkConsistency(CythonTestCase):
    """Test that augmented and pure-python produce identical results."""

    MODULE_PATH = "tests.fixtures.augmented_example"
    MODULE_NAME = "example"
    IMPLEMENTATIONS = [
        CythonModuleType.AUGMENTED_PYTHON,
        CythonModuleType.PURE_PYTHON,
    ]

    @cython_test_implementations()
    def test_add_consistency(self, module, implementation):
        """All implementations should produce the same result."""
        for a, b in [(0, 0), (1, 2), (-5, 5), (100, 200)]:
            self.assertEqual(module.add(a, b), a + b)

    @cython_test_implementations()
    def test_fibonacci_consistency(self, module, implementation):
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        for n, expected_val in enumerate(expected):
            self.assertEqual(module.fibonacci(n), expected_val)
