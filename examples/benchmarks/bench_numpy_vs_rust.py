#!/usr/bin/env -S uv run --script

# /// script
# dependencies = ["cast-value[rs]", "numpy"]
# ///

"""Benchmark comparing numpy and rust cast_value implementations.

Measures throughput (elements/second) and peak Python memory allocation
for several codec configurations, showing where the rust backend shines.

Run with:
    uv run examples/benchmarks/bench_numpy_vs_rust.py
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
from cast_value_rs import cast_array as rs_cast_array

from cast_value.core import cast_array as numpy_cast_array

SIZE = 1_000_000
WARMUP = 3
REPEATS = 10

rng = np.random.default_rng(42)


def numpy_cast(arr, target_dtype, **kwargs):
    return numpy_cast_array(
        arr,
        target_dtype=np.dtype(target_dtype),
        rounding_mode=kwargs.get("rounding_mode", "nearest-even"),
        out_of_range_mode=kwargs.get("out_of_range_mode"),
        scalar_map_entries=kwargs.get("scalar_map_entries"),
    )


def rust_cast(arr, target_dtype, **kwargs):
    return rs_cast_array(
        arr=arr,
        target_dtype=target_dtype,
        rounding_mode=kwargs.get("rounding_mode", "nearest-even"),
        out_of_range_mode=kwargs.get("out_of_range_mode"),
        scalar_map_entries=kwargs.get("scalar_map_entries"),
    )


def measure_throughput(fn, arr, target_dtype, **kwargs) -> float:
    """Return median elements/second over REPEATS runs."""
    for _ in range(WARMUP):
        fn(arr, target_dtype, **kwargs)
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn(arr, target_dtype, **kwargs)
        times.append(time.perf_counter() - t0)
    median_time = sorted(times)[len(times) // 2]
    return len(arr) / median_time


def measure_peak_memory(fn, arr, target_dtype, **kwargs) -> int:
    """Return peak Python-side bytes allocated during one call."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn(arr, target_dtype, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def fmt_throughput(eps: float) -> str:
    if eps >= 1e9:
        return f"{eps / 1e9:.1f}G"
    if eps >= 1e6:
        return f"{eps / 1e6:.1f}M"
    if eps >= 1e3:
        return f"{eps / 1e3:.1f}K"
    return f"{eps:.0f}"


def fmt_bytes(b: int) -> str:
    if b >= 1024 * 1024:
        return f"{b / 1024 / 1024:.1f} MB"
    if b >= 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b} B"


def make_scalar_map_array() -> np.ndarray:
    """Float64 array starting at 3.0 with ~10% NaN, ~5% +Inf, ~5% -Inf."""
    arr = rng.uniform(3, 1000, size=SIZE).astype(np.float64)
    arr[::10] = np.nan
    arr[1::20] = np.inf
    arr[3::20] = -np.inf
    return arr


def build_configs() -> list[dict]:
    return [
        {
            "name": "float64 -> float32 (simple narrowing)",
            "arr": rng.uniform(-100, 100, size=SIZE).astype(np.float64),
            "target": "float32",
            "numpy_kw": {},
            "rust_kw": {},
        },
        {
            "name": "float64 -> int32 (round nearest-even)",
            "arr": rng.uniform(-1000, 1000, size=SIZE).astype(np.float64),
            "target": "int32",
            "numpy_kw": {"rounding_mode": "nearest-even"},
            "rust_kw": {"rounding_mode": "nearest-even"},
        },
        {
            "name": "float64 -> float32 (round towards-zero)",
            "arr": rng.uniform(-100, 100, size=SIZE).astype(np.float64),
            "target": "float32",
            "numpy_kw": {"rounding_mode": "towards-zero"},
            "rust_kw": {"rounding_mode": "towards-zero"},
        },
        {
            "name": "float64 -> uint8 (clamp, SIMD path)",
            "arr": rng.uniform(-50, 300, size=SIZE).astype(np.float64),
            "target": "uint8",
            "numpy_kw": {"out_of_range_mode": "clamp"},
            "rust_kw": {"out_of_range_mode": "clamp"},
        },
        {
            "name": "float64 -> int32 (scalar_map: NaN/Inf/-Inf)",
            "arr": make_scalar_map_array(),
            "target": "int32",
            "numpy_kw": {
                "scalar_map_entries": [
                    (np.float64(np.nan), np.float64(0.0)),
                    (np.float64(np.inf), np.float64(1.0)),
                    (np.float64(-np.inf), np.float64(2.0)),
                ],
            },
            "rust_kw": {
                "scalar_map_entries": [
                    [float("nan"), 0],
                    [float("inf"), 1],
                    [float("-inf"), 2],
                ],
            },
        },
    ]


def main() -> None:
    configs = build_configs()

    print(f"Array size: {SIZE:,} elements\n")
    header = f"{'Configuration':<46} {'Impl':>6} {'Throughput':>12} {'Memory':>10}"
    print(header)
    print("-" * len(header))

    for config in configs:
        arr = config["arr"]
        target = config["target"]

        for impl_name, fn, kwargs in [
            ("numpy", numpy_cast, config["numpy_kw"]),
            ("rust", rust_cast, config["rust_kw"]),
        ]:
            throughput = measure_throughput(fn, arr, target, **kwargs)
            memory = measure_peak_memory(fn, arr, target, **kwargs)
            print(
                f"{config['name']:<46} {impl_name:>6} "
                f"{fmt_throughput(throughput):>10}/s {fmt_bytes(memory):>10}"
            )
        print()


if __name__ == "__main__":
    main()
