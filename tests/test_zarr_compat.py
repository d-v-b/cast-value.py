from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from conftest import Expect, ExpectFail
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import NDBuffer, default_buffer_prototype
from zarr.core.chunk_grids import RegularChunkGrid
from zarr.core.dtype import get_data_type_from_json

from cast_value.zarr_compat import CastValue
from cast_value.zarr_compat.v1 import _parse_map_entries


def _make_spec(
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


def _encode(
    codec: CastValue,
    arr: np.ndarray,
    source_dtype_str: str,
    fill_value: Any = 0,
) -> np.ndarray:
    """Encode a numpy array through a CastValue codec, return the result as ndarray."""
    spec = _make_spec(source_dtype_str, fill_value, shape=arr.shape)
    buf = NDBuffer.from_ndarray_like(arr)
    encoded = codec._encode_sync(buf, spec)
    assert encoded is not None
    return np.asarray(encoded.as_ndarray_like())


def _decode(
    codec: CastValue,
    arr: np.ndarray,
    source_dtype_str: str,
    fill_value: Any = 0,
) -> np.ndarray:
    """Decode a numpy array through a CastValue codec, return the result as ndarray."""
    spec = _make_spec(source_dtype_str, fill_value, shape=arr.shape)
    buf = NDBuffer.from_ndarray_like(arr)
    decoded = codec._decode_sync(buf, spec)
    return np.asarray(decoded.as_ndarray_like())


def _arrays_bytes_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Byte-level array comparison."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return (a.view(np.uint8) == b.view(np.uint8)).all().item()


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            id="int32-to-uint8",
            input=(
                CastValue(data_type="uint8"),
                np.array([1, 2, 3, 4], dtype=np.int32),
                "int32",
            ),
            expected=np.array([1, 2, 3, 4], dtype=np.uint8),
            eq=_arrays_bytes_equal,
        ),
        Expect(
            id="float64-to-float32",
            input=(
                CastValue(data_type="float32"),
                np.array([1.5, 2.5], dtype=np.float64),
                "float64",
            ),
            expected=np.array([1.5, 2.5], dtype=np.float32),
            eq=_arrays_bytes_equal,
        ),
        Expect(
            id="float64-to-int32-nearest-even",
            input=(
                CastValue(data_type="int32", rounding="nearest-even"),
                np.array([1.5, 2.5], dtype=np.float64),
                "float64",
            ),
            expected=np.array([2, 2], dtype=np.int32),
            eq=_arrays_bytes_equal,
        ),
        Expect(
            id="int32-to-int8-clamp",
            input=(
                CastValue(data_type="int8", out_of_range="clamp"),
                np.array([300, -200], dtype=np.int32),
                "int32",
            ),
            expected=np.array([127, -128], dtype=np.int8),
            eq=_arrays_bytes_equal,
        ),
        Expect(
            id="int32-to-uint8-wrap",
            input=(
                CastValue(data_type="uint8", out_of_range="wrap"),
                np.array([256, -1], dtype=np.int32),
                "int32",
            ),
            expected=np.array([0, 255], dtype=np.uint8),
            eq=_arrays_bytes_equal,
        ),
    ],
)
def test_encode(
    case: Expect[tuple[CastValue, np.ndarray, str], np.ndarray],
) -> None:
    """Test that CastValue.encode produces expected dtype and values."""
    codec, arr, source_dtype_str = case.input
    result = _encode(codec, arr, source_dtype_str)
    assert case.eq(result, case.expected)


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            id="uint8-to-int32",
            input=(
                CastValue(data_type="uint8"),
                np.array([1, 2, 3], dtype=np.uint8),
                "int32",
            ),
            expected=np.array([1, 2, 3], dtype=np.int32),
            eq=_arrays_bytes_equal,
        ),
        Expect(
            id="float32-to-float64",
            input=(
                CastValue(data_type="float32"),
                np.array([1.5, 2.5], dtype=np.float32),
                "float64",
            ),
            expected=np.array([1.5, 2.5], dtype=np.float64),
            eq=_arrays_bytes_equal,
        ),
    ],
)
def test_decode(
    case: Expect[tuple[CastValue, np.ndarray, str], np.ndarray],
) -> None:
    """Test that CastValue.decode produces expected dtype and values."""
    codec, arr, source_dtype_str = case.input
    result = _decode(codec, arr, source_dtype_str)
    assert case.eq(result, case.expected)


# ---------------------------------------------------------------------------
# encode/decode roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            id="int32-to-uint8",
            input=("int32", "uint8", np.array([1, 2, 3, 4], dtype=np.int32)),
            expected=np.array([1, 2, 3, 4], dtype=np.int32),
            eq=_arrays_bytes_equal,
        ),
        Expect(
            id="int32-to-int64",
            input=("int32", "int64", np.array([1, 2, 3], dtype=np.int32)),
            expected=np.array([1, 2, 3], dtype=np.int32),
            eq=_arrays_bytes_equal,
        ),
        Expect(
            id="float64-to-float32",
            input=("float64", "float32", np.array([1.5, 2.5], dtype=np.float64)),
            expected=np.array([1.5, 2.5], dtype=np.float64),
            eq=_arrays_bytes_equal,
        ),
        Expect(
            id="int16-to-float32",
            input=("int16", "float32", np.array([1, 2, 3], dtype=np.int16)),
            expected=np.array([1, 2, 3], dtype=np.int16),
            eq=_arrays_bytes_equal,
        ),
        Expect(
            id="float64-to-int32-exact-integers",
            input=("float64", "int32", np.array([1.0, 2.0, 3.0], dtype=np.float64)),
            expected=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            eq=_arrays_bytes_equal,
        ),
    ],
)
def test_encode_decode_roundtrip(
    case: Expect[tuple[str, str, np.ndarray], np.ndarray],
) -> None:
    """Test that encoding then decoding recovers the original values."""
    source_dtype_str, target_dtype_str, arr = case.input
    codec = CastValue(data_type=target_dtype_str)
    encoded = _encode(codec, arr, source_dtype_str)
    decoded = _decode(codec, encoded, source_dtype_str)
    assert case.eq(decoded, case.expected)


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
    codec = CastValue.from_dict(case.input)
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
    ],
)
def test_validate(
    case: Expect[tuple[str, str, str | None], None],
) -> None:
    """Test that validate accepts valid type combinations."""
    source_dtype_str, target_dtype_str, out_of_range = case.input
    codec = CastValue(data_type=target_dtype_str, out_of_range=out_of_range)
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
            id="int64-to-float32-precision-loss",
            input=("int64", "float32", None),
            err=ValueError,
            msg="may silently lose precision",
        ),
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
    codec = CastValue(data_type=target_dtype_str, out_of_range=out_of_range)
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
    codec = CastValue(data_type=target_dtype_str)
    spec = _make_spec(source_dtype_str, fill_value)
    result = codec.resolve_metadata(spec)
    expected_dtype, expected_fill = case.expected
    assert result.dtype.to_native_dtype() == expected_dtype
    assert result.fill_value == expected_fill


# ---------------------------------------------------------------------------
# encode with scalar_map
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            id="encode-maps-value",
            input=(
                CastValue(
                    data_type="float32",
                    scalar_map={"encode": [("1", "99")]},
                ),
                np.array([1, 2, 3], dtype=np.int32),
                "int32",
            ),
            expected=np.array([99.0, 2.0, 3.0], dtype=np.float32),
            eq=_arrays_bytes_equal,
        ),
        Expect(
            id="decode-maps-value",
            input=(
                CastValue(
                    data_type="float32",
                    scalar_map={"decode": [("99", "1")]},
                ),
                np.array([99.0, 2.0, 3.0], dtype=np.float32),
                "int32",
            ),
            expected=np.array([1, 2, 3], dtype=np.int32),
            eq=_arrays_bytes_equal,
        ),
    ],
)
def test_scalar_map(
    case: Expect[tuple[CastValue, np.ndarray, str], np.ndarray],
) -> None:
    """Test encode and decode with scalar_map parameter."""
    codec, arr, source_dtype_str = case.input
    if "encode" in (case.id):
        result = _encode(codec, arr, source_dtype_str)
    else:
        result = _decode(codec, arr, source_dtype_str)
    assert case.eq(result, case.expected)


# ---------------------------------------------------------------------------
# _parse_map_entries
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
    """Test that _parse_map_entries deserializes scalar mappings via zarr dtypes."""
    mapping, src_dtype_str, tgt_dtype_str = case.input
    src_zdtype = get_data_type_from_json(src_dtype_str, zarr_format=3)
    tgt_zdtype = get_data_type_from_json(tgt_dtype_str, zarr_format=3)
    result = _parse_map_entries(mapping, src_zdtype, tgt_zdtype)
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
    codec = CastValue(data_type=target_dtype_str)
    spec = _make_spec(source_dtype_str, 0)
    result = codec.compute_encoded_size(input_bytes, spec)
    assert result == case.expected
