# cython: language_level=3str
# cython: augmented_pure_python=True

import cython


@cython.ccall
def add(a: cython.int, b: cython.int) -> cython.int:
    return a + b


@cython.ccall
def multiply(a: cython.double, b: cython.double) -> cython.double:
    return a * b


@cython.ccall
def fibonacci(n: cython.int) -> cython.int:
    if n <= 1:
        return n
    a: cython.int = 0
    b: cython.int = 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
