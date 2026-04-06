# hb-cython-framework

Augmented Pure Python testing and build tooling for hummingbot sub-packages.

## Overview

Provides the test framework and build configuration for progressive Cython migration in hummingbot. Enables modules to run as plain Python while optionally compiling to native C extensions via Cython.

## Features

- `CythonTestCase` — tests 4 module variants (augmented, pure Python, compiled, pure Cython)
- `@cython_test_implementations()` decorator for variant parametrization
- Pre-commit hook for `.pyx` symlink management
- hatch-cython build integration
- AST-based refactoring rules (via refactor-applications) for automated conversion

## Installation

```bash
pixi install
```

## Usage

```python
from cython_framework.testing import CythonTestCase, cython_test_implementations

class TestMyModule(CythonTestCase):
    MODULE_PATH = "my_package.my_module"
    MODULE_NAME = "my_module"

    @cython_test_implementations()
    def test_function(self, module, implementation):
        result = module.my_function(42)
        self.assertEqual(result, expected)
```

## Development

```bash
pixi run check         # lint + format + test
pixi run test          # run all tests
pixi run test-compile  # build + test compiled extensions
pixi run cython-convert-dry <src>  # preview cython annotations
```

## License

Apache-2.0 — see [LICENSE](LICENSE)
