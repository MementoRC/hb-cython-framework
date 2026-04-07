"""Validation test generator for Cython migration.

Inspects an augmented pure-Python module and auto-generates an equivalence
test file that exercises every public function against all loaded
implementations via ValidationTestCase helpers.

Usage
-----
Programmatic::

    from cython_framework.validation.generator import generate_validation_tests
    from pathlib import Path

    path = generate_validation_tests(
        module_path="hummingbot.data_feed.candles_feed.candles_base",
        module_name="candles_base",
        output_dir=Path("tests/generated"),
    )

CLI::

    python -m cython_framework.validation.generator \\
        --module hummingbot.data_feed.candles_feed.candles_base \\
        --name candles_base \\
        --output-dir tests/generated
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
import textwrap
from pathlib import Path
from typing import Any

from cython_framework.testing.cython_test_case import CythonModuleType

# ---------------------------------------------------------------------------
# Type annotation → sample argument mapping
# ---------------------------------------------------------------------------

_SAMPLE_ARGS: dict[Any, str] = {
    int: "1",
    float: "1.0",
    str: '"test"',
    bool: "True",
    bytes: 'b"test"',
}

_DEFAULT_SAMPLE = "0"


def _sample_arg_for_annotation(annotation: Any) -> str:
    """Return a simple literal string suitable as a sample argument.

    Falls back to ``0`` for unknown / un-annotated parameters.
    """
    if annotation is inspect.Parameter.empty:
        return _DEFAULT_SAMPLE
    return _SAMPLE_ARGS.get(annotation, _DEFAULT_SAMPLE)


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------


def _load_module(module_path: str, module_name: str) -> Any:
    """Import the augmented Python module from its source .py file.

    Walks up from the current working directory to locate the project root
    (pyproject.toml or .git), then resolves the dotted *module_path* to a
    filesystem path and loads the .py file directly to avoid picking up any
    compiled .so extension.
    """
    project_root = _find_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    path_parts = module_path.split(".")
    module_dir = project_root / Path(*path_parts)
    py_file = module_dir / f"{module_name}.py"

    if not py_file.exists():
        raise FileNotFoundError(
            f"Module source not found: {py_file}\n"
            f"(resolved module_path={module_path!r}, module_name={module_name!r})"
        )

    full_name = f"{module_path}._generator_inspect_.{module_name}"
    spec = importlib.util.spec_from_file_location(full_name, py_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {py_file}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = module_path
    sys.modules[full_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _find_project_root() -> Path:
    """Walk up from CWD to find the project root (pyproject.toml / .git)."""
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists() or (current / ".git").exists():
            return current
        current = current.parent
    # Fallback: also try relative to this file
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists() or (current / ".git").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root (no pyproject.toml or .git)")


# ---------------------------------------------------------------------------
# Function discovery
# ---------------------------------------------------------------------------


def _discover_public_functions(module: Any) -> list[tuple[str, inspect.Signature]]:
    """Return (name, signature) for every public function defined in *module*.

    Includes all public (no leading underscore) callables that are functions.
    Does not filter by __module__ to handle augmented modules whose __name__
    is a namespaced generator key rather than the original dotted path.
    """
    results = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue
        try:
            sig = inspect.signature(obj)
        except (ValueError, TypeError):
            sig = inspect.Signature()
        results.append((name, sig))
    return results


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------


def _implementations_repr(implementations: list[CythonModuleType]) -> str:
    """Render the IMPLEMENTATIONS list literal."""
    items = ", ".join(f"CythonModuleType.{impl.name}" for impl in implementations)
    return f"[{items}]"


def _generate_function_tests(func_name: str, sig: inspect.Signature) -> str:
    """Generate the two test methods for a single function."""
    params = [p for p in sig.parameters.values() if p.name not in ("self", "cls")]
    sample_args = ", ".join(_sample_arg_for_annotation(p.annotation) for p in params)

    signature_test = textwrap.dedent(f"""\
        def test_{func_name}_signature_matches(self) -> None:
            \"\"\"Parameter names must match across all implementations.\"\"\"
            self.assert_signature_matches("{func_name}")
    """)

    if sample_args:
        equal_call = f'self.assert_implementations_equal("{func_name}", {sample_args})'
    else:
        equal_call = f'self.assert_implementations_equal("{func_name}")'

    equal_test = textwrap.dedent(f"""\
        def test_{func_name}_implementations_equal(self) -> None:
            \"\"\"All implementations must return the same result for {func_name}.\"\"\"
            {equal_call}
    """)

    return signature_test + "\n" + equal_test


def _check_hypothesis_available() -> bool:
    """Return True if the hypothesis package is importable."""
    return importlib.util.find_spec("hypothesis") is not None


def _is_numeric_function(sig: inspect.Signature) -> bool:
    """Return True if all parameters are annotated as int or float (or unannotated)."""
    numeric_types = {int, float, inspect.Parameter.empty}
    for p in sig.parameters.values():
        if p.name in ("self", "cls"):
            continue
        if p.annotation not in numeric_types:
            return False
    return True


def _generate_hypothesis_test(func_name: str, sig: inspect.Signature) -> str:
    """Generate a property-based test using hypothesis for numeric functions."""
    params = [p for p in sig.parameters.values() if p.name not in ("self", "cls")]
    if not params:
        return ""

    strategies = []
    param_names = []
    for p in params:
        ann = p.annotation
        param_names.append(p.name)
        if ann is float:
            strategies.append("st.floats(allow_nan=False, allow_infinity=False)")
        else:
            strategies.append("st.integers(min_value=-1000, max_value=1000)")

    given_args = ", ".join(strategies)
    call_args = ", ".join(param_names)

    return textwrap.dedent(f"""\
        @given({given_args})
        def test_{func_name}_property_based(self, {call_args}) -> None:
            \"\"\"Property-based equivalence test for {func_name} (hypothesis).\"\"\"
            self.assert_implementations_equal("{func_name}", {call_args})
    """)


def _generate_test_file(
    module_path: str,
    module_name: str,
    functions: list[tuple[str, inspect.Signature]],
    implementations: list[CythonModuleType],
    *,
    use_hypothesis: bool = False,
) -> str:
    """Render the complete test file as a string."""
    class_name = "TestValidate" + module_name.replace("_", " ").title().replace(" ", "")

    # ---- imports ----
    imports = textwrap.dedent(f"""\
        \"\"\"Auto-generated validation tests for {module_name}.

        Generated by cython_framework.validation.generator.
        Run against all configured implementations to ensure equivalence.
        \"\"\"

        from __future__ import annotations

        from cython_framework.testing import CythonModuleType
        from cython_framework.validation.helpers import ValidationTestCase
    """)

    if use_hypothesis:
        imports += textwrap.dedent("""\
            from hypothesis import given
            from hypothesis import strategies as st
        """)

    imports += "\n"

    # ---- class header ----
    impl_repr = _implementations_repr(implementations)
    class_header = textwrap.dedent(f"""\
        class {class_name}(ValidationTestCase):
            \"\"\"Equivalence tests for {module_name} across all implementations.\"\"\"

            MODULE_PATH = "{module_path}"
            MODULE_NAME = "{module_name}"
            IMPLEMENTATIONS = {impl_repr}

    """)

    # ---- test methods ----
    method_blocks = []
    for func_name, sig in functions:
        block = _generate_function_tests(func_name, sig)
        # Indent each method 4 spaces for class body
        indented = textwrap.indent(block, "    ")
        method_blocks.append(indented)

        if use_hypothesis and _is_numeric_function(sig):
            hyp_block = _generate_hypothesis_test(func_name, sig)
            if hyp_block:
                method_blocks.append(textwrap.indent(hyp_block, "    "))

    if not method_blocks:
        # Ensure the class body is not empty
        method_blocks.append("    pass\n")

    return imports + class_header + "\n".join(method_blocks) + "\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_validation_tests(
    module_path: str,
    module_name: str,
    output_dir: Path | str,
    implementations: list[CythonModuleType] | None = None,
) -> Path:
    """Generate an equivalence test file for *module_name*.

    Parameters
    ----------
    module_path:
        Dotted package path to the directory containing the module, e.g.
        ``"tests.fixtures.augmented_example"``.
    module_name:
        The module file stem (without ``.py``), e.g. ``"example"``.
    output_dir:
        Directory where the generated test file will be written.
    implementations:
        List of :class:`CythonModuleType` variants to include.  Defaults to
        ``[AUGMENTED_PYTHON, PURE_PYTHON]``.

    Returns
    -------
    Path
        Absolute path to the generated test file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if implementations is None:
        implementations = [
            CythonModuleType.AUGMENTED_PYTHON,
            CythonModuleType.PURE_PYTHON,
        ]

    # Load module and discover functions
    module = _load_module(module_path, module_name)
    functions = _discover_public_functions(module)

    use_hypothesis = _check_hypothesis_available()

    source = _generate_test_file(
        module_path=module_path,
        module_name=module_name,
        functions=functions,
        implementations=implementations,
        use_hypothesis=use_hypothesis,
    )

    output_file = output_dir / f"test_validate_{module_name}.py"
    output_file.write_text(source, encoding="utf-8")
    return output_file


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate equivalence validation tests for an augmented Python module.",
        prog="python -m cython_framework.validation.generator",
    )
    parser.add_argument(
        "--module",
        required=True,
        metavar="DOTTED_PATH",
        help="Dotted path to the package directory containing the module.",
    )
    parser.add_argument(
        "--name",
        required=True,
        metavar="MODULE_NAME",
        help="Module file stem (without .py extension).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory where the generated test file will be written.",
    )
    parser.add_argument(
        "--implementations",
        nargs="*",
        metavar="IMPL",
        help=(
            "CythonModuleType variants to include "
            "(e.g. AUGMENTED_PYTHON PURE_PYTHON). "
            "Defaults to AUGMENTED_PYTHON and PURE_PYTHON."
        ),
    )

    args = parser.parse_args()

    impls: list[CythonModuleType] | None = None
    if args.implementations:
        impls = [CythonModuleType[name] for name in args.implementations]

    output_path = generate_validation_tests(
        module_path=args.module,
        module_name=args.name,
        output_dir=Path(args.output_dir),
        implementations=impls,
    )
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    _main()
