"""Pre-commit hooks for Augmented Pure Python workflow."""

from cython_framework.hooks.link_augmented_pyx import is_augmented, main

__all__ = ["is_augmented", "main"]
