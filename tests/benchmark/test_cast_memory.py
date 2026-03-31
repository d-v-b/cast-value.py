"""Memory benchmarks comparing numpy and rust cast_array implementations.

Uses tracemalloc to measure peak Python-side memory allocated during each call.
This captures numpy intermediate arrays but not rust/C-level allocations.

Run with:
    uv run --group bench pytest tests/benchmark/test_cast_memory.py -v -s
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from cast_value.core import cast_array as numpy_cast_array

try:
    from cast_value_rs import cast_array as rs_cast_array

    _has_rs = True
except ImportError:
    _has_rs = False

SIZES = [
    pytest.param(100, id="100"),
    pytest.param(10_000, id="10k"),
    pytest.param(1_000_000, id="1M"),
]

_rng = np.random.default_rng(42)

_results: list[tuple[str, int, str, int]] = []


def _measure_peak_bytes(fn, *args, **kwargs) -> int:
    """Call *fn* and return peak bytes allocated (Python-side only)."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def _record(name: str, size: int, impl: str, peak: int) -> None:
    _results.append((name, size, impl, peak))


def _numpy_cast(arr, target_dtype, **kwargs):
    return numpy_cast_array(
        arr,
        target_dtype=np.dtype(target_dtype),
        rounding_mode=kwargs.get("rounding_mode", "nearest-even"),
        out_of_range_mode=kwargs.get("out_of_range_mode"),
        scalar_map_entries=kwargs.get("scalar_map_entries"),
    )


def _rs_cast(arr, target_dtype, **kwargs):
    return rs_cast_array(
        arr=arr,
        target_dtype=target_dtype,
        rounding_mode=kwargs.get("rounding_mode", "nearest-even"),
        out_of_range_mode=kwargs.get("out_of_range_mode"),
        scalar_map_entries=kwargs.get("scalar_map_entries"),
    )


_IMPLEMENTATIONS = [pytest.param(_numpy_cast, "numpy", id="numpy")]
if _has_rs:
    _IMPLEMENTATIONS.append(pytest.param(_rs_cast, "rs", id="rs"))


@pytest.fixture(autouse=True, scope="session")
def _print_memory_summary():
    """Print a summary table of peak memory allocations after all tests."""
    yield
    if not _results:
        return
    print("\n")
    print("=" * 72)
    print("Peak Python memory allocations (tracemalloc)")
    print("=" * 72)
    print(f"{'Test':<40} {'Size':>8} {'Impl':>6} {'Peak':>12}")
    print("-" * 72)
    for name, size, impl, peak in sorted(_results):
        if peak < 1024:
            peak_str = f"{peak} B"
        elif peak < 1024 * 1024:
            peak_str = f"{peak / 1024:.1f} KB"
        else:
            peak_str = f"{peak / 1024 / 1024:.1f} MB"
        print(f"{name:<40} {size:>8,} {impl:>6} {peak_str:>12}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# float -> float with non-default rounding (numpy creates many intermediates)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("cast_fn", "impl_name"), _IMPLEMENTATIONS)
@pytest.mark.parametrize("size", SIZES)
def test_float_to_float_towards_zero_memory(cast_fn, impl_name, size):
    """Measure peak memory for float64 -> float32 towards-zero rounding."""
    arr = _rng.uniform(-100, 100, size=size).astype(np.float64)
    peak = _measure_peak_bytes(cast_fn, arr, "float32", rounding_mode="towards-zero")
    _record("float_to_float_towards_zero", size, impl_name, peak)


# ---------------------------------------------------------------------------
# int -> int with clamp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("cast_fn", "impl_name"), _IMPLEMENTATIONS)
@pytest.mark.parametrize("size", SIZES)
def test_int_to_int_clamp_memory(cast_fn, impl_name, size):
    """Measure peak memory for int32 -> int8 with clamp."""
    arr = _rng.integers(-200, 300, size=size, dtype=np.int32)
    peak = _measure_peak_bytes(cast_fn, arr, "int8", out_of_range_mode="clamp")
    _record("int_to_int_clamp", size, impl_name, peak)


# ---------------------------------------------------------------------------
# float -> int with scalar_map
# ---------------------------------------------------------------------------


_NUMPY_SCALAR_MAP_ENTRIES = [
    (np.float64(np.nan), np.float64(0.0)),
    (np.float64(np.inf), np.float64(1.0)),
    (np.float64(-np.inf), np.float64(2.0)),
]

_RS_SCALAR_MAP_ENTRIES = [
    [float("nan"), 0],
    [float("inf"), 1],
    [float("-inf"), 2],
]


def _make_scalar_map_array(size: int) -> np.ndarray:
    """Create a float64 array starting at 3.0 with NaN, +Inf, -Inf sprinkled in."""
    arr = _rng.uniform(3, 1000, size=size).astype(np.float64)
    arr[::10] = np.nan
    arr[1::20] = np.inf
    arr[3::20] = -np.inf
    return arr


@pytest.mark.parametrize("size", SIZES)
def test_scalar_map_numpy_memory(size):
    """Measure peak memory for float64 -> int32 with scalar_map (numpy)."""
    arr = _make_scalar_map_array(size)
    peak = _measure_peak_bytes(
        _numpy_cast, arr, "int32", scalar_map_entries=_NUMPY_SCALAR_MAP_ENTRIES
    )
    _record("scalar_map", size, "numpy", peak)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.skipif(not _has_rs, reason="cast-value-rs not installed")
def test_scalar_map_rs_memory(size):
    """Measure peak memory for float64 -> int32 with scalar_map (rust)."""
    arr = _make_scalar_map_array(size)
    peak = _measure_peak_bytes(
        _rs_cast, arr, "int32", scalar_map_entries=_RS_SCALAR_MAP_ENTRIES
    )
    _record("scalar_map", size, "rs", peak)
