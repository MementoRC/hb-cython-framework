# cython: language_level=3str
# cython: augmented_pure_python=True
"""Phase 0 de-risk spike: first @cython.cclass in the framework. Throwaway."""

import cython


@cython.cclass
class RunningStats:
    _count: cython.int
    _sum: cython.double
    _sumsq: cython.double

    def __init__(self):
        self._count = 0
        self._sum = 0.0
        self._sumsq = 0.0

    @cython.ccall
    def add(self, x: cython.double) -> cython.int:
        self._count += 1
        self._sum += x
        self._sumsq += x * x
        return self._count

    @cython.ccall
    def mean(self) -> cython.double:
        if self._count == 0:
            return 0.0
        return self._sum / self._count

    @cython.cfunc
    def _variance(self) -> cython.double:
        if self._count == 0:
            return 0.0
        m: cython.double = self._sum / self._count
        # Clamp away tiny negative variance from FP cancellation so stddev's
        # sqrt never sees a negative input (would be complex/NaN depending
        # on variant, causing cross-variant mismatches).
        return max(0.0, self._sumsq / self._count - m * m)

    @cython.ccall
    def stddev(self) -> cython.double:
        v: cython.double = self._variance()
        return v**0.5


def bench_running_stats(n: int = 500) -> float:
    """Module-level benchmark entry point exercising RunningStats end-to-end.

    A plain callable (rather than the RunningStats cclass itself) is needed
    because BenchmarkTestCase.benchmark_all() resolves its target via
    getattr(module, func_name) and calls it directly -- see G3 benchmark_cli.
    """
    stats = RunningStats()
    for i in range(n):
        stats.add(float(i))
    stats.mean()
    return stats.stddev()
