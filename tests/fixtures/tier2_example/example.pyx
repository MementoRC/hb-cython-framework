# cython: language_level=3str
"""Tier 2 example: heavy Cython with nogil and memoryviews."""
cimport cython
from libc.math cimport sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_norm(double[:] data) nogil:
    cdef int i
    cdef int n = data.shape[0]
    cdef double total = 0.0
    for i in range(n):
        total += data[i] * data[i]
    return sqrt(total)
