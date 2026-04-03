#!/usr/bin/env python3
"""Create/cleanup .pyx symlinks for Cython augmented pure-python files.

Scans staged .py files for the `# cython: augmented_pure_python=True` pragma.
Creates relative .pyx symlinks so Cython build discovers them.
Removes stale .pyx symlinks pointing to non-augmented .py files.
"""
import subprocess
from pathlib import Path

PRAGMA = "# cython: augmented_pure_python=True"


def get_staged_py_files() -> list[Path]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True, text=True
    )
    return [Path(f) for f in result.stdout.strip().split("\n") if f.endswith(".py")]


def is_augmented(path: Path) -> bool:
    try:
        with open(path) as f:
            for line in f:
                if line.strip() == PRAGMA:
                    return True
                if not line.startswith("#") and line.strip():
                    break
    except (OSError, UnicodeDecodeError):
        pass
    return False


def main():
    for py_file in get_staged_py_files():
        pyx_file = py_file.with_suffix(".pyx")
        if is_augmented(py_file):
            if not pyx_file.is_symlink():
                pyx_file.symlink_to(py_file.name)
                print(f"  Created symlink: {pyx_file} -> {py_file.name}")
        else:
            if pyx_file.is_symlink() and pyx_file.resolve().name == py_file.name:
                pyx_file.unlink()
                print(f"  Removed stale symlink: {pyx_file}")


if __name__ == "__main__":
    main()
