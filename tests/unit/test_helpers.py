"""Tests for cython_framework.validation.helpers."""

from cython_framework.testing import CythonModuleType
from cython_framework.validation.helpers import ValidationTestCase


class TestValidationHelpers(ValidationTestCase):
    MODULE_PATH = "tests.fixtures.augmented_example"
    MODULE_NAME = "example"
    IMPLEMENTATIONS = [
        CythonModuleType.AUGMENTED_PYTHON,
        CythonModuleType.PURE_PYTHON,
    ]

    def test_assert_implementations_equal_passes(self):
        self.assert_implementations_equal("add", 2, 3)

    def test_assert_implementations_equal_multi_args(self):
        self.assert_implementations_equal("multiply", 2.5, 4.0)

    def test_assert_implementations_close_passes(self):
        self.assert_implementations_close("multiply", 2.5, 4.0)

    def test_assert_signature_matches(self):
        self.assert_signature_matches("add")
