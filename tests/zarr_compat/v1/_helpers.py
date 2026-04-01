"""Shared helpers for zarr_compat v1 tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import NDBuffer, default_buffer_prototype
from zarr.core.dtype import get_data_type_from_json

if TYPE_CHECKING:
    from cast_value.zarr_compat.v1 import CastValueBaseV1


def make_spec(
    dtype_str: str, fill_value: Any, shape: tuple[int, ...] = (4,)
) -> ArraySpec:
    """Create an ArraySpec for testing."""
    zdtype = get_data_type_from_json(dtype_str, zarr_format=3)
    return ArraySpec(
        shape=shape,
        dtype=zdtype,
        fill_value=fill_value,
        config=ArrayConfig(order="C", write_empty_chunks=False),
        prototype=default_buffer_prototype(),
    )


def encode(
    codec: CastValueBaseV1,
    arr: np.ndarray,
    source_dtype_str: str,
    fill_value: Any = 0,
) -> np.ndarray:
    """Encode a numpy array through a CastValue codec, return the result as ndarray."""
    spec = make_spec(source_dtype_str, fill_value, shape=arr.shape)
    buf = NDBuffer.from_ndarray_like(arr)  # ty: ignore[invalid-argument-type]
    encoded = codec._encode_sync(buf, spec)
    assert encoded is not None
    return np.asarray(encoded.as_ndarray_like())


def decode(
    codec: CastValueBaseV1,
    arr: np.ndarray,
    source_dtype_str: str,
    fill_value: Any = 0,
) -> np.ndarray:
    """Decode a numpy array through a CastValue codec, return the result as ndarray."""
    spec = make_spec(source_dtype_str, fill_value, shape=arr.shape)
    buf = NDBuffer.from_ndarray_like(arr)  # ty: ignore[invalid-argument-type]
    decoded = codec._decode_sync(buf, spec)
    return np.asarray(decoded.as_ndarray_like())


def arrays_bytes_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Byte-level array comparison."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return (a.view(np.uint8) == b.view(np.uint8)).all().item()
