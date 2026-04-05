"""Pytest migration rules for unittest -> modern pytest conversion.

Usage:
    python cython_framework/refactors/pytest_rules.py <test_path> --apply

Rules applied:
    1. RemoveUnittestInheritance - Remove TestCase base classes
    2. RemoveUnittestImport - Remove unittest imports
    3. RemoveTestWrapperImport - Remove wrapper imports
    4. ConvertAssertions - self.assertEqual -> assert
    5. ConvertSetUpTearDown - setUp/tearDown -> @pytest.fixture
    6. ConvertAssertRaises - assertRaises -> pytest.raises
    7. ConvertUnittestDecorators - @unittest.skip -> @pytest.mark.skip
    8. AddPytestImport - Add pytest import when needed
"""

from refactor import run
from refactor.rules.pytest_migration import (
    AddPytestImport,
    ConvertAssertions,
    ConvertAssertRaises,
    ConvertSetUpTearDown,
    ConvertUnittestDecorators,
    RemoveTestWrapperImport,
    RemoveUnittestImport,
    RemoveUnittestInheritance,
)

ALL_RULES = [
    RemoveUnittestInheritance,
    RemoveUnittestImport,
    RemoveTestWrapperImport,
    ConvertAssertions,
    ConvertSetUpTearDown,
    ConvertAssertRaises,
    ConvertUnittestDecorators,
    AddPytestImport,
]

if __name__ == "__main__":
    raise SystemExit(run(ALL_RULES))
