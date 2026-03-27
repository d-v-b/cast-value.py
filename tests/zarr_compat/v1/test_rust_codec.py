"""Tests for CastValueRust — encode/decode via the cast-value-rs backend."""

from __future__ import annotations

import pytest

from cast_value.zarr_compat.v1 import CastValueRust
from zarr_compat.v1._cases import (
    decode_cases,
    encode_cases,
    roundtrip_cases,
    scalar_map_decode_cases,
    scalar_map_encode_cases,
)
from zarr_compat.v1._helpers import decode, encode

_has_rust = True
try:
    import cast_value_rs  # noqa: F401
except ImportError:
    _has_rust = False

pytestmark = pytest.mark.skipif(not _has_rust, reason="cast-value-rs not installed")


@pytest.mark.parametrize("case", encode_cases(CastValueRust))
def test_encode(case):
    """Test that CastValueRust.encode produces expected dtype and values."""
    codec, arr, source_dtype_str = case.input
    result = encode(codec, arr, source_dtype_str)
    assert case.eq(result, case.expected)


@pytest.mark.parametrize("case", decode_cases(CastValueRust))
def test_decode(case):
    """Test that CastValueRust.decode produces expected dtype and values."""
    codec, arr, source_dtype_str = case.input
    result = decode(codec, arr, source_dtype_str)
    assert case.eq(result, case.expected)


@pytest.mark.parametrize("case", roundtrip_cases(CastValueRust))
def test_encode_decode_roundtrip(case):
    """Test that encoding then decoding recovers the original values."""
    source_dtype_str, target_dtype_str, arr, cls = case.input
    codec = cls(data_type=target_dtype_str)
    encoded = encode(codec, arr, source_dtype_str)
    decoded = decode(codec, encoded, source_dtype_str)
    assert case.eq(decoded, case.expected)


@pytest.mark.parametrize("case", scalar_map_encode_cases(CastValueRust))
def test_scalar_map_encode(case):
    """Test encode with scalar_map parameter."""
    codec, arr, source_dtype_str = case.input
    result = encode(codec, arr, source_dtype_str)
    assert case.eq(result, case.expected)


@pytest.mark.parametrize("case", scalar_map_decode_cases(CastValueRust))
def test_scalar_map_decode(case):
    """Test decode with scalar_map parameter."""
    codec, arr, source_dtype_str = case.input
    result = decode(codec, arr, source_dtype_str)
    assert case.eq(result, case.expected)
