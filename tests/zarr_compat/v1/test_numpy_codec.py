"""Tests for CastValueNumpyV1 — encode/decode via the pure-numpy backend."""

from __future__ import annotations

import pytest

from cast_value.zarr_compat.v1 import CastValueNumpyV1
from zarr_compat.v1._cases import (
    decode_cases,
    encode_cases,
    roundtrip_cases,
    scalar_map_decode_cases,
    scalar_map_encode_cases,
)
from zarr_compat.v1._helpers import decode, encode


@pytest.mark.parametrize("case", encode_cases(CastValueNumpyV1))
def test_encode(case):
    """Test that CastValueNumpyV1.encode produces expected dtype and values."""
    codec, arr, source_dtype_str = case.input
    result = encode(codec, arr, source_dtype_str)
    assert case.eq(result, case.expected)


@pytest.mark.parametrize("case", decode_cases(CastValueNumpyV1))
def test_decode(case):
    """Test that CastValueNumpyV1.decode produces expected dtype and values."""
    codec, arr, source_dtype_str = case.input
    result = decode(codec, arr, source_dtype_str)
    assert case.eq(result, case.expected)


@pytest.mark.parametrize("case", roundtrip_cases(CastValueNumpyV1))
def test_encode_decode_roundtrip(case):
    """Test that encoding then decoding recovers the original values."""
    source_dtype_str, target_dtype_str, arr, cls = case.input
    codec = cls(data_type=target_dtype_str)
    encoded = encode(codec, arr, source_dtype_str)
    decoded = decode(codec, encoded, source_dtype_str)
    assert case.eq(decoded, case.expected)


@pytest.mark.parametrize("case", scalar_map_encode_cases(CastValueNumpyV1))
def test_scalar_map_encode(case):
    """Test encode with scalar_map parameter."""
    codec, arr, source_dtype_str = case.input
    result = encode(codec, arr, source_dtype_str)
    assert case.eq(result, case.expected)


@pytest.mark.parametrize("case", scalar_map_decode_cases(CastValueNumpyV1))
def test_scalar_map_decode(case):
    """Test decode with scalar_map parameter."""
    codec, arr, source_dtype_str = case.input
    result = decode(codec, arr, source_dtype_str)
    assert case.eq(result, case.expected)
