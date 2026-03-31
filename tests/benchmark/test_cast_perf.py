"""Benchmarks comparing numpy and rust cast_array implementations.

Run with:
    uv run --group bench pytest tests/benchmark/ --benchmark-group-by=param:size
"""

from __future__ import annotations

from collections.abc import Callable

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

CastFn = Callable[..., np.ndarray]

_rng = np.random.default_rng(42)


def _numpy_cast(
    arr: np.ndarray,
    target_dtype: str,
    rounding_mode: str = "nearest-even",
    out_of_range_mode: str | None = None,
    scalar_map_entries: list | None = None,
) -> np.ndarray:
    """Call the numpy implementation."""
    return numpy_cast_array(
        arr,
        target_dtype=np.dtype(target_dtype),
        rounding_mode=rounding_mode,  # ty: ignore[invalid-argument-type]
        out_of_range_mode=out_of_range_mode,  # ty: ignore[invalid-argument-type]
        scalar_map_entries=scalar_map_entries,
    )


def _rs_cast(
    arr: np.ndarray,
    target_dtype: str,
    rounding_mode: str = "nearest-even",
    out_of_range_mode: str | None = None,
    scalar_map_entries: list | None = None,
) -> np.ndarray:
    """Call the rust implementation."""
    return rs_cast_array(
        arr=arr,
        target_dtype=target_dtype,  # ty: ignore[invalid-argument-type]
        rounding_mode=rounding_mode,  # ty: ignore[invalid-argument-type]
        out_of_range_mode=out_of_range_mode,  # ty: ignore[invalid-argument-type]
        scalar_map_entries=scalar_map_entries,
    )


_IMPLEMENTATIONS: list[tuple[str, CastFn]] = [("numpy", _numpy_cast)]
if _has_rs:
    _IMPLEMENTATIONS.append(("rs", _rs_cast))


@pytest.fixture(params=_IMPLEMENTATIONS, ids=[name for name, _ in _IMPLEMENTATIONS])
def cast_fn(request: pytest.FixtureRequest) -> CastFn:
    """Parametrized fixture yielding each available cast implementation."""
    _, fn = request.param
    return fn


# ---------------------------------------------------------------------------
# int -> int (narrowing with clamp)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", SIZES)
def test_int_to_int_clamp(benchmark, cast_fn, size):
    """Benchmark int32 -> int8 with clamp."""
    arr = _rng.integers(-200, 300, size=size, dtype=np.int32)
    benchmark(cast_fn, arr, "int8", out_of_range_mode="clamp")


# ---------------------------------------------------------------------------
# float -> int (with rounding)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", SIZES)
def test_float_to_int_round(benchmark, cast_fn, size):
    """Benchmark float64 -> int32 with nearest-even rounding."""
    arr = _rng.uniform(-1000, 1000, size=size).astype(np.float64)
    benchmark(cast_fn, arr, "int32", rounding_mode="nearest-even")


# ---------------------------------------------------------------------------
# float -> float (narrowing, nearest-even)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", SIZES)
def test_float_to_float_narrow(benchmark, cast_fn, size):
    """Benchmark float64 -> float32 narrowing."""
    arr = _rng.uniform(-100, 100, size=size).astype(np.float64)
    benchmark(cast_fn, arr, "float32")


# ---------------------------------------------------------------------------
# float -> float (narrowing, non-default rounding)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", SIZES)
def test_float_to_float_towards_zero(benchmark, cast_fn, size):
    """Benchmark float64 -> float32 with towards-zero rounding."""
    arr = _rng.uniform(-100, 100, size=size).astype(np.float64)
    benchmark(cast_fn, arr, "float32", rounding_mode="towards-zero")


# ---------------------------------------------------------------------------
# int -> float (widening)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", SIZES)
def test_int_to_float_widen(benchmark, cast_fn, size):
    """Benchmark int16 -> float64 widening."""
    arr = _rng.integers(-1000, 1000, size=size, dtype=np.int16)
    benchmark(cast_fn, arr, "float64")


# ---------------------------------------------------------------------------
# float -> int with scalar_map (NaN, +Inf, -Inf -> sentinel ints)
# ---------------------------------------------------------------------------


def _make_scalar_map_array(size: int) -> np.ndarray:
    """Create a float64 array starting at 3.0 with NaN, +Inf, -Inf sprinkled in."""
    arr = _rng.uniform(3, 1000, size=size).astype(np.float64)
    # ~10% NaN, ~5% +Inf, ~5% -Inf
    arr[::10] = np.nan
    arr[1::20] = np.inf
    arr[3::20] = -np.inf
    return arr


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


@pytest.mark.parametrize("size", SIZES)
def test_scalar_map_numpy(benchmark, size):
    """Benchmark float64 -> int32 with NaN/Inf/-Inf scalar_map (numpy)."""
    arr = _make_scalar_map_array(size)
    benchmark(_numpy_cast, arr, "int32", scalar_map_entries=_NUMPY_SCALAR_MAP_ENTRIES)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.skipif(not _has_rs, reason="cast-value-rs not installed")
def test_scalar_map_rs(benchmark, size):
    """Benchmark float64 -> int32 with NaN/Inf/-Inf scalar_map (rust)."""
    arr = _make_scalar_map_array(size)
    benchmark(_rs_cast, arr, "int32", scalar_map_entries=_RS_SCALAR_MAP_ENTRIES)
