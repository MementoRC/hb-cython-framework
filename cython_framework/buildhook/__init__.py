"""Build-time tooling: the env-gated, pragma-driven augmented Cython build hook.

Import :mod:`cython_framework.buildhook.hook` for the hook itself; it is
deliberately kept out of this package's ``__init__`` so importing
``cython_framework.buildhook`` does not require hatchling to be installed.
"""
