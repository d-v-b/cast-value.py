"""Shared test case data for encode/decode tests across backends.

Each function takes a codec class and returns a list of Expect cases
constructed with that class. This allows the same cases to be reused
by both the numpy and rust backend test modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from conftest import Expect

from zarr_compat.v1._helpers import arrays_bytes_equal

if TYPE_CHECKING:
    from cast_value.zarr_compat.v1 import CastValueBaseV1


def encode_cases(
    cls: type[CastValueBaseV1],
) -> list[Expect[tuple[Any, np.ndarray, str], np.ndarray]]:
    """Test cases for encoding."""
    return [
        Expect(
            id="int32-to-uint8",
            input=(
                cls(data_type="uint8"),
                np.array([1, 2, 3, 4], dtype=np.int32),
                "int32",
            ),
            expected=np.array([1, 2, 3, 4], dtype=np.uint8),
            eq=arrays_bytes_equal,
        ),
        Expect(
            id="float64-to-float32",
            input=(
                cls(data_type="float32"),
                np.array([1.5, 2.5], dtype=np.float64),
                "float64",
            ),
            expected=np.array([1.5, 2.5], dtype=np.float32),
            eq=arrays_bytes_equal,
        ),
        Expect(
            id="float64-to-int32-nearest-even",
            input=(
                cls(data_type="int32", rounding="nearest-even"),
                np.array([1.5, 2.5], dtype=np.float64),
                "float64",
            ),
            expected=np.array([2, 2], dtype=np.int32),
            eq=arrays_bytes_equal,
        ),
        Expect(
            id="int32-to-int8-clamp",
            input=(
                cls(data_type="int8", out_of_range="clamp"),
                np.array([300, -200], dtype=np.int32),
                "int32",
            ),
            expected=np.array([127, -128], dtype=np.int8),
            eq=arrays_bytes_equal,
        ),
        Expect(
            id="int32-to-uint8-wrap",
            input=(
                cls(data_type="uint8", out_of_range="wrap"),
                np.array([256, -1], dtype=np.int32),
                "int32",
            ),
            expected=np.array([0, 255], dtype=np.uint8),
            eq=arrays_bytes_equal,
        ),
    ]


def decode_cases(
    cls: type[CastValueBaseV1],
) -> list[Expect[tuple[Any, np.ndarray, str], np.ndarray]]:
    """Test cases for decoding."""
    return [
        Expect(
            id="uint8-to-int32",
            input=(
                cls(data_type="uint8"),
                np.array([1, 2, 3], dtype=np.uint8),
                "int32",
            ),
            expected=np.array([1, 2, 3], dtype=np.int32),
            eq=arrays_bytes_equal,
        ),
        Expect(
            id="float32-to-float64",
            input=(
                cls(data_type="float32"),
                np.array([1.5, 2.5], dtype=np.float32),
                "float64",
            ),
            expected=np.array([1.5, 2.5], dtype=np.float64),
            eq=arrays_bytes_equal,
        ),
    ]


def roundtrip_cases(
    cls: type[CastValueBaseV1],
) -> list[Expect[tuple[str, str, np.ndarray, type[CastValueBaseV1]], np.ndarray]]:
    """Test cases for encode/decode roundtrip."""
    return [
        Expect(
            id="int32-to-uint8",
            input=("int32", "uint8", np.array([1, 2, 3, 4], dtype=np.int32), cls),
            expected=np.array([1, 2, 3, 4], dtype=np.int32),
            eq=arrays_bytes_equal,
        ),
        Expect(
            id="int32-to-int64",
            input=("int32", "int64", np.array([1, 2, 3], dtype=np.int32), cls),
            expected=np.array([1, 2, 3], dtype=np.int32),
            eq=arrays_bytes_equal,
        ),
        Expect(
            id="float64-to-float32",
            input=("float64", "float32", np.array([1.5, 2.5], dtype=np.float64), cls),
            expected=np.array([1.5, 2.5], dtype=np.float64),
            eq=arrays_bytes_equal,
        ),
        Expect(
            id="int16-to-float32",
            input=("int16", "float32", np.array([1, 2, 3], dtype=np.int16), cls),
            expected=np.array([1, 2, 3], dtype=np.int16),
            eq=arrays_bytes_equal,
        ),
        Expect(
            id="float64-to-int32-exact-integers",
            input=(
                "float64",
                "int32",
                np.array([1.0, 2.0, 3.0], dtype=np.float64),
                cls,
            ),
            expected=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            eq=arrays_bytes_equal,
        ),
    ]


def scalar_map_encode_cases(
    cls: type[CastValueBaseV1],
) -> list[Expect[tuple[Any, np.ndarray, str], np.ndarray]]:
    """Test cases for encode with scalar_map."""
    return [
        Expect(
            id="encode-maps-value",
            input=(
                cls(
                    data_type="float32",
                    scalar_map={"encode": [("1", "99")]},
                ),
                np.array([1, 2, 3], dtype=np.int32),
                "int32",
            ),
            expected=np.array([99.0, 2.0, 3.0], dtype=np.float32),
            eq=arrays_bytes_equal,
        ),
    ]


def scalar_map_decode_cases(
    cls: type[CastValueBaseV1],
) -> list[Expect[tuple[Any, np.ndarray, str], np.ndarray]]:
    """Test cases for decode with scalar_map."""
    return [
        Expect(
            id="decode-maps-value",
            input=(
                cls(
                    data_type="float32",
                    scalar_map={"decode": [("99", "1")]},
                ),
                np.array([99.0, 2.0, 3.0], dtype=np.float32),
                "int32",
            ),
            expected=np.array([1, 2, 3], dtype=np.int32),
            eq=arrays_bytes_equal,
        ),
    ]
