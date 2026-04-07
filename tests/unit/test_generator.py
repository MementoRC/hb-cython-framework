"""Tests for the validation test generator.

Verifies that generate_validation_tests() produces syntactically valid,
structurally correct test files from inspecting an augmented Python module.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURE_MODULE_PATH = "tests.fixtures.augmented_example"
FIXTURE_MODULE_NAME = "example"


def _generate(output_dir: Path) -> Path:
    """Call the generator against the augmented_example fixture."""
    from cython_framework.validation.generator import generate_validation_tests

    return generate_validation_tests(
        module_path=FIXTURE_MODULE_PATH,
        module_name=FIXTURE_MODULE_NAME,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateValidationTests:
    """Tests for generate_validation_tests()."""

    def test_output_path_returned(self, tmp_path: Path) -> None:
        """Return value must be a Path that exists on disk."""
        result = _generate(tmp_path)

        assert isinstance(result, Path)
        assert result.exists()

    def test_generated_file_is_named_correctly(self, tmp_path: Path) -> None:
        """Generated file must follow the test_validate_<module_name>.py convention."""
        result = _generate(tmp_path)

        assert result.name == f"test_validate_{FIXTURE_MODULE_NAME}.py"

    def test_generates_valid_python_file(self, tmp_path: Path) -> None:
        """Generated file must be syntactically valid Python (compile without error)."""
        result = _generate(tmp_path)
        source = result.read_text()

        # compile() raises SyntaxError if source is not valid Python
        try:
            compile(source, str(result), "exec")
        except SyntaxError as exc:
            pytest.fail(f"Generated file is not valid Python: {exc}")

    def test_generated_file_contains_test_class(self, tmp_path: Path) -> None:
        """Generated file must contain a class TestValidate<ModuleName>."""
        result = _generate(tmp_path)
        source = result.read_text()

        assert "class TestValidateExample" in source

    def test_generated_file_has_module_path(self, tmp_path: Path) -> None:
        """Generated class must declare MODULE_PATH matching the input."""
        result = _generate(tmp_path)
        source = result.read_text()

        assert f'MODULE_PATH = "{FIXTURE_MODULE_PATH}"' in source

    def test_generated_file_has_module_name(self, tmp_path: Path) -> None:
        """Generated class must declare MODULE_NAME matching the input."""
        result = _generate(tmp_path)
        source = result.read_text()

        assert f'MODULE_NAME = "{FIXTURE_MODULE_NAME}"' in source

    def test_generated_file_has_implementations(self, tmp_path: Path) -> None:
        """Generated class must declare an IMPLEMENTATIONS list."""
        result = _generate(tmp_path)
        source = result.read_text()

        assert "IMPLEMENTATIONS" in source

    def test_generated_file_has_function_tests(self, tmp_path: Path) -> None:
        """Generated file must contain test methods for each discovered function."""
        result = _generate(tmp_path)
        source = result.read_text()

        assert "test_add" in source
        assert "test_multiply" in source
        assert "test_fibonacci" in source

    def test_generated_file_has_signature_tests(self, tmp_path: Path) -> None:
        """Generated file must call assert_signature_matches for each function."""
        result = _generate(tmp_path)
        source = result.read_text()

        assert "assert_signature_matches" in source
        assert '"add"' in source
        assert '"multiply"' in source
        assert '"fibonacci"' in source

    def test_generated_file_has_implementations_equal_tests(self, tmp_path: Path) -> None:
        """Generated file must call assert_implementations_equal for each function."""
        result = _generate(tmp_path)
        source = result.read_text()

        assert "assert_implementations_equal" in source

    def test_generated_file_imports_validation_helpers(self, tmp_path: Path) -> None:
        """Generated file must import ValidationTestCase."""
        result = _generate(tmp_path)
        source = result.read_text()

        assert "ValidationTestCase" in source
        assert "cython_framework.validation.helpers" in source

    def test_generated_file_imports_testing_module(self, tmp_path: Path) -> None:
        """Generated file must import CythonModuleType."""
        result = _generate(tmp_path)
        source = result.read_text()

        assert "CythonModuleType" in source

    def test_generated_tests_are_parseable_ast(self, tmp_path: Path) -> None:
        """Generated file must produce a valid AST with expected structure."""
        result = _generate(tmp_path)
        source = result.read_text()

        tree = ast.parse(source)
        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert any("TestValidate" in name for name in class_names)

        # There should be multiple function defs (test methods)
        func_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        assert len(func_defs) >= 3  # at least one test per discovered function

    def test_generated_tests_actually_importable(self, tmp_path: Path) -> None:
        """Generated file must be importable as a Python module."""
        result = _generate(tmp_path)

        # Add tmp_path to sys.path so we can import the generated module
        sys.path.insert(0, str(tmp_path))
        try:
            module_name = result.stem
            # Remove cached version if present
            sys.modules.pop(module_name, None)
            mod = importlib.import_module(module_name)
            assert mod is not None
            # Verify the test class is accessible
            assert hasattr(mod, "TestValidateExample")
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop(result.stem, None)

    def test_custom_output_dir_is_used(self, tmp_path: Path) -> None:
        """Generator must write to the specified output directory."""
        sub_dir = tmp_path / "generated_tests"
        sub_dir.mkdir()

        result = _generate(sub_dir)

        assert result.parent == sub_dir

    def test_custom_implementations(self, tmp_path: Path) -> None:
        """Generator must accept a custom implementations list."""
        from cython_framework.testing import CythonModuleType
        from cython_framework.validation.generator import generate_validation_tests

        result = generate_validation_tests(
            module_path=FIXTURE_MODULE_PATH,
            module_name=FIXTURE_MODULE_NAME,
            output_dir=tmp_path,
            implementations=[CythonModuleType.AUGMENTED_PYTHON, CythonModuleType.PURE_PYTHON],
        )
        source = result.read_text()

        assert "AUGMENTED_PYTHON" in source
        assert "PURE_PYTHON" in source


class TestGeneratorCLI:
    """Tests for the __main__ CLI entry point."""

    def test_cli_generates_file(self, tmp_path: Path) -> None:
        """CLI invocation must produce a test file in the specified output dir."""
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cython_framework.validation.generator",
                "--module",
                FIXTURE_MODULE_PATH,
                "--name",
                FIXTURE_MODULE_NAME,
                "--output-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
        )

        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        expected = tmp_path / f"test_validate_{FIXTURE_MODULE_NAME}.py"
        assert expected.exists(), f"Expected {expected} to exist"
