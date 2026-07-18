"""Local custom-hook shim for the augmented Cython build hook.

Lets a repository enable the hook by path
(``[tool.hatch.build.targets.wheel.hooks.custom] path = "hatch_build.py"``)
before it consumes hb-cython-framework as a registered plugin. The real logic
lives in :mod:`cython_framework.build.hook`; this module only re-exports the hook
class so hatchling's custom-hook loader discovers it.
"""
import os
import sys

# Ensure the in-tree package is importable while hatchling loads this shim.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cython_framework.buildhook.hook import AugmentedCythonBuildHook

__all__ = ["AugmentedCythonBuildHook"]
