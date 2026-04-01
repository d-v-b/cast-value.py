"""Test that CastValueRustV1 fails clearly when cast-value-rs is not installed.

This test must run in an environment WITHOUT cast-value-rs to be meaningful.
It is skipped if cast-value-rs is installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from zarr.core.buffer import NDBuffer

from cast_value.zarr_compat.v1 import CastValueRustV1
from zarr_compat.v1._helpers import make_spec

try:
    import cast_value_rs  # noqa: F401

    _has_rust = True
except ImportError:
    _has_rust = False

pytestmark = pytest.mark.skipif(_has_rust, reason="cast-value-rs IS installed")


def test_rust_codec_importable_without_backend() -> None:
    """Test that CastValueRustV1 can be imported and constructed without cast-value-rs."""
    codec = CastValueRustV1(data_type="uint8")
    assert codec.to_dict() == {
        "name": "cast_value",
        "configuration": {"data_type": "uint8"},
    }


def test_rust_codec_encode_raises_without_backend() -> None:
    """Test that encoding with CastValueRustV1 raises ModuleNotFoundError without cast-value-rs."""
    codec = CastValueRustV1(data_type="uint8")
    arr = np.array([1, 2, 3], dtype=np.int32)
    spec = make_spec("int32", 0, shape=arr.shape)
    buf = NDBuffer.from_ndarray_like(arr)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ModuleNotFoundError, match="cast_value_rs"):
        codec._encode_sync(buf, spec)


def test_rust_codec_decode_raises_without_backend() -> None:
    """Test that decoding with CastValueRustV1 raises ModuleNotFoundError without cast-value-rs."""
    codec = CastValueRustV1(data_type="uint8")
    arr = np.array([1, 2, 3], dtype=np.uint8)
    spec = make_spec("int32", 0, shape=arr.shape)
    buf = NDBuffer.from_ndarray_like(arr)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ModuleNotFoundError, match="cast_value_rs"):
        codec._decode_sync(buf, spec)
