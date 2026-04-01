"""Tests for CastValueNumpyV1Base — shared behavior independent of backend."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest
from conftest import Expect, ExpectFail
from zarr.core.buffer import NDBuffer
from zarr.core.chunk_grids import RegularChunkGrid
from zarr.core.dtype import get_data_type_from_json

from cast_value.zarr_compat.v1 import CastValueNumpyV1, parse_map_entries
from zarr_compat.v1._helpers import arrays_bytes_equal, make_spec

# ---------------------------------------------------------------------------
# from_dict / to_dict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            id="minimal",
            input={"name": "cast_value", "configuration": {"data_type": "uint8"}},
            expected={"name": "cast_value", "configuration": {"data_type": "uint8"}},
        ),
        Expect(
            id="with-rounding",
            input={
                "name": "cast_value",
                "configuration": {"data_type": "int32", "rounding": "towards-zero"},
            },
            expected={
                "name": "cast_value",
                "configuration": {"data_type": "int32", "rounding": "towards-zero"},
            },
        ),
        Expect(
            id="with-out-of-range",
            input={
                "name": "cast_value",
                "configuration": {"data_type": "uint8", "out_of_range": "clamp"},
            },
            expected={
                "name": "cast_value",
                "configuration": {"data_type": "uint8", "out_of_range": "clamp"},
            },
        ),
        Expect(
            id="with-scalar-map",
            input={
                "name": "cast_value",
                "configuration": {
                    "data_type": "float32",
                    "scalar_map": {"encode": [("1", "99")]},
                },
            },
            expected={
                "name": "cast_value",
                "configuration": {
                    "data_type": "float32",
                    "scalar_map": {"encode": [("1", "99")]},
                },
            },
        ),
    ],
)
def test_from_dict_to_dict_roundtrip(
    case: Expect[dict, dict],
) -> None:
    """Test that from_dict followed by to_dict recovers the original dict."""
    codec = CastValueNumpyV1.from_dict(case.input)
    result = codec.to_dict()
    assert result == case.expected


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            id="int32-to-uint8",
            input=("int32", "uint8", None),
            expected=None,
        ),
        Expect(
            id="float64-to-float32",
            input=("float64", "float32", None),
            expected=None,
        ),
        Expect(
            id="int16-to-float64",
            input=("int16", "float64", None),
            expected=None,
        ),
        Expect(
            id="float32-to-int16",
            input=("float32", "int16", None),
            expected=None,
        ),
        Expect(
            id="wrap-on-int-target",
            input=("int32", "uint8", "wrap"),
            expected=None,
        ),
        Expect(
            id="int64-to-float32",
            input=("int64", "float32", None),
            expected=None,
        ),
        Expect(
            id="float32-to-int64",
            input=("float32", "int64", None),
            expected=None,
        ),
    ],
)
def test_validate(
    case: Expect[tuple[str, str, str | None], None],
) -> None:
    """Test that validate accepts valid type combinations."""
    source_dtype_str, target_dtype_str, out_of_range = case.input
    codec = CastValueNumpyV1(data_type=target_dtype_str, out_of_range=out_of_range)  # ty: ignore[invalid-argument-type]
    source_zdtype = get_data_type_from_json(source_dtype_str, zarr_format=3)
    codec.validate(
        shape=(4,),
        dtype=source_zdtype,
        chunk_grid=RegularChunkGrid(chunk_shape=(4,)),
    )


@pytest.mark.parametrize(
    "case",
    [
        ExpectFail(
            id="wrap-on-float-target",
            input=("int32", "float32", "wrap"),
            err=ValueError,
            msg="out_of_range='wrap' is only valid for integer",
        ),
    ],
)
def test_validate_fail(
    case: ExpectFail[tuple[str, str, str | None]],
) -> None:
    """Test that validate rejects invalid type combinations."""
    source_dtype_str, target_dtype_str, out_of_range = case.input
    codec = CastValueNumpyV1(data_type=target_dtype_str, out_of_range=out_of_range)  # ty: ignore[invalid-argument-type]
    source_zdtype = get_data_type_from_json(source_dtype_str, zarr_format=3)
    with pytest.raises(case.err, match=case.msg):
        codec.validate(
            shape=(4,),
            dtype=source_zdtype,
            chunk_grid=RegularChunkGrid(chunk_shape=(4,)),
        )


# ---------------------------------------------------------------------------
# resolve_metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            id="int32-to-uint8-fill-0",
            input=("int32", "uint8", np.int32(0)),
            expected=(np.dtype(np.uint8), np.uint8(0)),
        ),
        Expect(
            id="float64-to-int32-fill-1p5-rounds",
            input=("float64", "int32", np.float64(1.5)),
            expected=(np.dtype(np.int32), np.int32(2)),
        ),
        Expect(
            id="int16-to-float64-fill-42",
            input=("int16", "float64", np.int16(42)),
            expected=(np.dtype(np.float64), np.float64(42.0)),
        ),
    ],
)
def test_resolve_metadata(
    case: Expect[tuple[str, str, Any], tuple[np.dtype, Any]],
) -> None:
    """Test that resolve_metadata transforms fill_value and dtype correctly."""
    source_dtype_str, target_dtype_str, fill_value = case.input
    codec = CastValueNumpyV1(data_type=target_dtype_str)
    spec = make_spec(source_dtype_str, fill_value)
    result = codec.resolve_metadata(spec)
    expected_dtype, expected_fill = case.expected
    assert result.dtype.to_native_dtype() == expected_dtype
    assert result.fill_value == expected_fill


# ---------------------------------------------------------------------------
# parse_map_entries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            id="int-to-int-map",
            input=(
                {"1": "10", "2": "20"},
                "int32",
                "uint8",
            ),
            expected=[(np.int32(1), np.uint8(10)), (np.int32(2), np.uint8(20))],
        ),
        Expect(
            id="float-to-float-map",
            input=(
                {"1.5": "2.5"},
                "float64",
                "float32",
            ),
            expected=[(np.float64(1.5), np.float32(2.5))],
        ),
        Expect(
            id="empty-map",
            input=(
                {},
                "int32",
                "int32",
            ),
            expected=[],
        ),
    ],
)
def test_parse_map_entries(
    case: Expect[tuple[dict[str, str], str, str], list],
) -> None:
    """Test that parse_map_entries deserializes scalar mappings via zarr dtypes."""
    mapping, src_dtype_str, tgt_dtype_str = case.input
    src_zdtype = get_data_type_from_json(src_dtype_str, zarr_format=3)
    tgt_zdtype = get_data_type_from_json(tgt_dtype_str, zarr_format=3)
    result = parse_map_entries(mapping, src_zdtype, tgt_zdtype)
    assert len(result) == len(case.expected)
    for (rs, rt), (es, et) in zip(result, case.expected, strict=True):
        assert rs == es
        assert rt == et


# ---------------------------------------------------------------------------
# compute_encoded_size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            id="int32-to-uint8",
            input=("int32", "uint8", 16),
            expected=4,
        ),
        Expect(
            id="int8-to-int64",
            input=("int8", "int64", 4),
            expected=32,
        ),
        Expect(
            id="zero-length",
            input=("int32", "uint8", 0),
            expected=0,
        ),
    ],
)
def test_compute_encoded_size(
    case: Expect[tuple[str, str, int], int],
) -> None:
    """Test that compute_encoded_size calculates byte length correctly."""
    source_dtype_str, target_dtype_str, input_bytes = case.input
    codec = CastValueNumpyV1(data_type=target_dtype_str)
    spec = make_spec(source_dtype_str, 0)
    result = codec.compute_encoded_size(input_bytes, spec)
    assert result == case.expected


# ---------------------------------------------------------------------------
# __init__ with ZDType
# ---------------------------------------------------------------------------


def test_init_with_zdtype() -> None:
    """Test that CastValueNumpyV1 can be constructed with a ZDType instead of a string."""
    zdtype = get_data_type_from_json("uint8", zarr_format=3)
    codec = CastValueNumpyV1(data_type=zdtype)
    assert codec.dtype is zdtype


# ---------------------------------------------------------------------------
# async encode / decode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            id="async-encode-int32-to-uint8",
            input=(
                CastValueNumpyV1(data_type="uint8"),
                np.array([1, 2, 3, 4], dtype=np.int32),
                "int32",
            ),
            expected=np.array([1, 2, 3, 4], dtype=np.uint8),
            eq=arrays_bytes_equal,
        ),
    ],
)
def test_encode_single(
    case: Expect[tuple[CastValueNumpyV1, np.ndarray, str], np.ndarray],
) -> None:
    """Test that the async _encode_single path produces the same result as sync."""
    codec, arr, source_dtype_str = case.input
    spec = make_spec(source_dtype_str, 0, shape=arr.shape)
    buf = NDBuffer.from_ndarray_like(arr)  # ty: ignore[invalid-argument-type]
    result_buf = asyncio.run(codec._encode_single(buf, spec))
    assert result_buf is not None
    result = np.asarray(result_buf.as_ndarray_like())
    assert case.eq(result, case.expected)


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            id="async-decode-uint8-to-int32",
            input=(
                CastValueNumpyV1(data_type="uint8"),
                np.array([1, 2, 3], dtype=np.uint8),
                "int32",
            ),
            expected=np.array([1, 2, 3], dtype=np.int32),
            eq=arrays_bytes_equal,
        ),
    ],
)
def test_decode_single(
    case: Expect[tuple[CastValueNumpyV1, np.ndarray, str], np.ndarray],
) -> None:
    """Test that the async _decode_single path produces the same result as sync."""
    codec, arr, source_dtype_str = case.input
    spec = make_spec(source_dtype_str, 0, shape=arr.shape)
    buf = NDBuffer.from_ndarray_like(arr)  # ty: ignore[invalid-argument-type]
    result_buf = asyncio.run(codec._decode_single(buf, spec))
    result = np.asarray(result_buf.as_ndarray_like())
    assert case.eq(result, case.expected)
