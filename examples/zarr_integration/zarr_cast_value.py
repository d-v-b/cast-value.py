#!/usr/bin/env -S uv run --script

# /// script
# dependencies = ["zarr>=3.1.6", "cast-value[rs]", "numpy"]
# ///

"""Zarr-Python integration example for the cast_value codec.

This script demonstrates creating an in-memory zarr array that stores float64
data as uint8 using the cast_value codec, then reading it back.

Run with:
    uv run examples/zarr_integration/zarr_cast_value.py
"""

from __future__ import annotations

import numpy as np
import zarr
import zarr.storage

from cast_value import CastValueRustV1


def main() -> None:
    # Create an in-memory zarr array with float64 dtype, stored as uint8.
    # The cast_value codec handles the conversion: float64 -> uint8 on write,
    # uint8 -> float64 on read.
    codec = CastValueRustV1(
        data_type="uint8",
        rounding="nearest-even",
        out_of_range="clamp",
        scalar_map={
            "encode": [(np.nan, 0), (np.inf, 1), (-np.inf, 2)],
            "decode": [(0, np.nan), (1, np.inf), (2, -np.inf)],
        },
    )
    # Create array and write float64 data — values are rounded and clamped to [0, 255]
    data = np.array([np.nan, np.inf, -np.inf, 3.3, 4])
    arr = zarr.create_array(data=data, store=zarr.storage.MemoryStore(), filters=codec)

    # Read it back — comes back as float64, but with uint8 precision
    result = arr[:]

    print(f"Array dtype: {arr.dtype}")
    print(f"Values written: {data}")
    print(f"Values read:    {result}")


if __name__ == "__main__":
    main()
