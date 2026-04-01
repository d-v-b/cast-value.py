from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from conftest import Expect, ExpectFail

from cast_value.impl._numpy import (
    apply_scalar_map,
    cast_array,
    check_int_range,
    round_inplace,
)
from cast_value.zarr_compat._parsing import extract_raw_map

if TYPE_CHECKING:
    from cast_value.types import ScalarMapEntry, ScalarMapJSON


# ---------------------------------------------------------------------------
# apply_scalar_map
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            id="single-int-replacement",
            input=(
                np.array([1, 2, 3], dtype=np.int64),
                [(np.int64(1), np.int64(10))],
            ),
            expected=np.array([10, 2, 3], dtype=np.int64),
        ),
        Expect(
            id="replaces-all-occurrences",
            input=(
                np.array([1, 2, 1, 3], dtype=np.int64),
                [(np.int64(1), np.int64(99))],
            ),
            expected=np.array([99, 2, 99, 3], dtype=np.int64),
        ),
        Expect(
            id="multiple-entries",
            input=(
                np.array([1, 2, 3], dtype=np.int64),
                [
                    (np.int64(1), np.int64(10)),
                    (np.int64(2), np.int64(20)),
                ],
            ),
            expected=np.array([10, 20, 3], dtype=np.int64),
        ),
        Expect(
            id="nan-source-replaced",
            input=(
                np.array([1.0, np.nan, 3.0], dtype=np.float64),
                [(np.float64(np.nan), np.float64(0.0))],
            ),
            expected=np.array([1.0, 0.0, 3.0], dtype=np.float64),
        ),
        Expect(
            id="nan-target",
            input=(
                np.array([1.0, 2.0, 3.0], dtype=np.float64),
                [(np.float64(2.0), np.float64(np.nan))],
            ),
            expected=np.array([1.0, np.nan, 3.0], dtype=np.float64),
        ),
        Expect(
            id="empty-entries-noop",
            input=(
                np.array([1, 2, 3], dtype=np.int64),
                [],
            ),
            expected=np.array([1, 2, 3], dtype=np.int64),
        ),
        Expect(
            id="no-match-noop",
            input=(
                np.array([5, 6, 7], dtype=np.int64),
                [(np.int64(99), np.int64(0))],
            ),
            expected=np.array([5, 6, 7], dtype=np.int64),
        ),
        Expect(
            id="all-nan-replaced",
            input=(
                np.array([np.nan, np.nan, np.nan], dtype=np.float64),
                [(np.float64(np.nan), np.float64(-1.0))],
            ),
            expected=np.array([-1.0, -1.0, -1.0], dtype=np.float64),
        ),
        Expect(
            id="inf-source-replaced",
            input=(
                np.array([1.0, np.inf, 3.0], dtype=np.float64),
                [(np.float64(np.inf), np.float64(0.0))],
            ),
            expected=np.array([1.0, 0.0, 3.0], dtype=np.float64),
        ),
        Expect(
            id="neg-inf-source-replaced",
            input=(
                np.array([1.0, -np.inf, 3.0], dtype=np.float64),
                [(np.float64(-np.inf), np.float64(0.0))],
            ),
            expected=np.array([1.0, 0.0, 3.0], dtype=np.float64),
        ),
        Expect(
            id="chain-order-first-wins",
            input=(
                np.array([1, 2, 3], dtype=np.int64),
                [
                    (np.int64(1), np.int64(2)),
                    (np.int64(2), np.int64(3)),
                ],
            ),
            expected=np.array([3, 3, 3], dtype=np.int64),
        ),
    ],
)
def test_apply_scalar_map(
    case: Expect[tuple[np.ndarray, list[ScalarMapEntry]], np.ndarray],
) -> None:
    """Test that apply_scalar_map modifies the array in-place according to entries."""
    work, entries = case.input
    apply_scalar_map(work, entries)
    assert case.eq(work, case.expected)


# ---------------------------------------------------------------------------
# round_inplace
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            id="nearest-even-half-values",
            input=(np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64), "nearest-even"),
            expected=np.array([0.0, 2.0, 2.0, 4.0], dtype=np.float64),
        ),
        Expect(
            id="nearest-even-non-half",
            input=(np.array([1.3, 2.7, -1.3, -2.7], dtype=np.float64), "nearest-even"),
            expected=np.array([1.0, 3.0, -1.0, -3.0], dtype=np.float64),
        ),
        Expect(
            id="towards-zero",
            input=(np.array([1.9, -1.9, 0.1, -0.1], dtype=np.float64), "towards-zero"),
            expected=np.array([1.0, -1.0, 0.0, -0.0], dtype=np.float64),
        ),
        Expect(
            id="towards-positive",
            input=(
                np.array([1.1, -1.1, 0.0, -0.9], dtype=np.float64),
                "towards-positive",
            ),
            expected=np.array([2.0, -1.0, 0.0, -0.0], dtype=np.float64),
        ),
        Expect(
            id="towards-negative",
            input=(
                np.array([1.9, -1.1, 0.0, 0.9], dtype=np.float64),
                "towards-negative",
            ),
            expected=np.array([1.0, -2.0, 0.0, 0.0], dtype=np.float64),
        ),
        Expect(
            id="nearest-away-half-values",
            input=(np.array([0.5, 1.5, -0.5, -1.5], dtype=np.float64), "nearest-away"),
            expected=np.array([1.0, 2.0, -1.0, -2.0], dtype=np.float64),
        ),
        Expect(
            id="nearest-away-non-half",
            input=(np.array([1.3, 2.7, -1.3, -2.7], dtype=np.float64), "nearest-away"),
            expected=np.array([1.0, 3.0, -1.0, -3.0], dtype=np.float64),
        ),
        Expect(
            id="already-integer-values",
            input=(np.array([3.0, -2.0, 0.0], dtype=np.float64), "nearest-even"),
            expected=np.array([3.0, -2.0, 0.0], dtype=np.float64),
        ),
        Expect(
            id="empty-array",
            input=(np.array([], dtype=np.float64), "nearest-even"),
            expected=np.array([], dtype=np.float64),
        ),
    ],
)
def test_round_inplace(case: Expect[tuple[np.ndarray, str], np.ndarray]) -> None:
    """Test that round_inplace rounds according to the specified mode."""
    arr, mode = case.input
    result = round_inplace(arr, mode)  # ty: ignore[invalid-argument-type]
    assert case.eq(result, case.expected)


@pytest.mark.parametrize(
    "case",
    [
        ExpectFail(
            id="invalid-mode",
            input=(np.array([1.0], dtype=np.float64), "bogus"),
            err=ValueError,
            msg="Unknown rounding mode",
        ),
    ],
)
def test_round_inplace_fail(case: ExpectFail[tuple[np.ndarray, str]]) -> None:
    """Test that round_inplace raises on invalid rounding modes."""
    arr, mode = case.input
    with pytest.raises(case.err, match=case.msg):
        round_inplace(arr, mode)  # ty: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# check_int_range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            id="in-range-uint8",
            input=(
                np.array([0, 100, 255], dtype=np.int64),
                np.dtype(np.uint8),
                None,
            ),
            expected=np.array([0, 100, 255], dtype=np.uint8),
        ),
        Expect(
            id="clamp-uint8",
            input=(
                np.array([-10, 300], dtype=np.int64),
                np.dtype(np.uint8),
                "clamp",
            ),
            expected=np.array([0, 255], dtype=np.uint8),
        ),
        Expect(
            id="wrap-uint8",
            input=(
                np.array([256, -1, 512], dtype=np.int64),
                np.dtype(np.uint8),
                "wrap",
            ),
            expected=np.array([0, 255, 0], dtype=np.uint8),
        ),
        Expect(
            id="wrap-int8",
            input=(
                np.array([128, -129], dtype=np.int64),
                np.dtype(np.int8),
                "wrap",
            ),
            expected=np.array([-128, 127], dtype=np.int8),
        ),
        Expect(
            id="clamp-int8",
            input=(
                np.array([-1000, 1000], dtype=np.int64),
                np.dtype(np.int8),
                "clamp",
            ),
            expected=np.array([-128, 127], dtype=np.int8),
        ),
        Expect(
            id="widen-int32-to-int64",
            input=(
                np.array([10, 20], dtype=np.int32),
                np.dtype(np.int64),
                None,
            ),
            expected=np.array([10, 20], dtype=np.int64),
        ),
        Expect(
            id="wrap-uint16",
            input=(
                np.array([65536, -1], dtype=np.int64),
                np.dtype(np.uint16),
                "wrap",
            ),
            expected=np.array([0, 65535], dtype=np.uint16),
        ),
        Expect(
            id="clamp-all-negative-to-uint8",
            input=(
                np.array([-5, -100, -1], dtype=np.int64),
                np.dtype(np.uint8),
                "clamp",
            ),
            expected=np.array([0, 0, 0], dtype=np.uint8),
        ),
        Expect(
            id="same-type-int8-noop",
            input=(
                np.array([-1, 0, 1], dtype=np.int64),
                np.dtype(np.int8),
                None,
            ),
            expected=np.array([-1, 0, 1], dtype=np.int8),
        ),
    ],
)
def test_check_int_range(
    case: Expect[tuple[np.ndarray, np.dtype, str | None], np.ndarray],
) -> None:
    """Test that check_int_range casts with correct out-of-range handling."""
    work, target_dtype, out_of_range = case.input
    result = check_int_range(
        work,
        target_dtype=target_dtype,
        out_of_range=out_of_range,  # ty: ignore[invalid-argument-type]
    )
    assert case.eq(result, case.expected)
    assert result.dtype == case.expected.dtype


@pytest.mark.parametrize(
    "case",
    [
        ExpectFail(
            id="error-on-out-of-range-none",
            input=(
                np.array([256], dtype=np.int64),
                np.dtype(np.uint8),
                None,
            ),
            err=ValueError,
            msg="Values out of range",
        ),
        ExpectFail(
            id="error-negative-for-unsigned",
            input=(
                np.array([-1], dtype=np.int64),
                np.dtype(np.uint8),
                None,
            ),
            err=ValueError,
            msg="Values out of range",
        ),
    ],
)
def test_check_int_range_fail(
    case: ExpectFail[tuple[np.ndarray, np.dtype, str | None]],
) -> None:
    """Test that check_int_range raises ValueError when values are out of range and out_of_range is None."""
    work, target_dtype, out_of_range = case.input
    with pytest.raises(case.err, match=case.msg):
        check_int_range(
            work,
            target_dtype=target_dtype,
            out_of_range=out_of_range,  # ty: ignore[invalid-argument-type]
        )


# ---------------------------------------------------------------------------
# extract_raw_map
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            id="none-data-returns-none",
            input=(None, "encode"),
            expected=None,
        ),
        Expect(
            id="encode-direction",
            input=({"encode": [("1", "10"), ("2", "20")]}, "encode"),
            expected={"1": "10", "2": "20"},
        ),
        Expect(
            id="decode-direction",
            input=({"decode": [("5", "50")]}, "decode"),
            expected={"5": "50"},
        ),
        Expect(
            id="missing-direction-returns-none",
            input=({"encode": [("1", "10")]}, "decode"),
            expected=None,
        ),
        Expect(
            id="empty-pairs-returns-none",
            input=({"encode": []}, "encode"),
            expected=None,
        ),
        Expect(
            id="both-directions-selects-encode",
            input=(
                {"encode": [("1", "10")], "decode": [("10", "1")]},
                "encode",
            ),
            expected={"1": "10"},
        ),
        Expect(
            id="both-directions-selects-decode",
            input=(
                {"encode": [("1", "10")], "decode": [("10", "1")]},
                "decode",
            ),
            expected={"10": "1"},
        ),
        Expect(
            id="non-string-values-stringified",
            input=({"encode": [(1, 10)]}, "encode"),
            expected={"1": "10"},
        ),
    ],
)
def test_extract_raw_map(
    case: Expect[tuple[ScalarMapJSON | None, str], dict[str, str] | None],
) -> None:
    """Test that extract_raw_map extracts the correct direction from scalar_map JSON."""
    data, direction = case.input
    result = extract_raw_map(data, direction)
    assert result == case.expected


# ===========================================================================
# cast_array — split by direction
# ===========================================================================


def _call_cast(
    arr: np.ndarray,
    target_dtype: np.dtype,
    rounding_mode: str = "nearest-even",
    out_of_range_mode: str | None = None,
    scalar_map_entries: list[ScalarMapEntry] | None = None,
) -> np.ndarray:
    return cast_array(
        arr,
        target_dtype=target_dtype,
        rounding_mode=rounding_mode,  # ty: ignore[invalid-argument-type]
        out_of_range_mode=out_of_range_mode,  # ty: ignore[invalid-argument-type]
        scalar_map_entries=scalar_map_entries,
    )


# ---------------------------------------------------------------------------
# cast_array: int → int
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        # --- narrowing ---
        Expect(
            id="narrow-int32-to-int8-in-range",
            input=(
                np.array([0, 127, -128], dtype=np.int32),
                np.dtype(np.int8),
                None,
                None,
            ),
            expected=np.array([0, 127, -128], dtype=np.int8),
        ),
        Expect(
            id="narrow-int32-to-int8-clamp",
            input=(
                np.array([0, 300, -200], dtype=np.int32),
                np.dtype(np.int8),
                "clamp",
                None,
            ),
            expected=np.array([0, 127, -128], dtype=np.int8),
        ),
        Expect(
            id="narrow-int32-to-int8-wrap",
            input=(
                np.array([128, -129], dtype=np.int32),
                np.dtype(np.int8),
                "wrap",
                None,
            ),
            expected=np.array([-128, 127], dtype=np.int8),
        ),
        # --- widening ---
        Expect(
            id="widen-int8-to-int32",
            input=(
                np.array([1, -1, 127, -128], dtype=np.int8),
                np.dtype(np.int32),
                None,
                None,
            ),
            expected=np.array([1, -1, 127, -128], dtype=np.int32),
        ),
        Expect(
            id="widen-int8-to-int64",
            input=(
                np.array([1, 2], dtype=np.int8),
                np.dtype(np.int64),
                None,
                None,
            ),
            expected=np.array([1, 2], dtype=np.int64),
        ),
        # --- cross-sign ---
        Expect(
            id="cross-int32-to-uint8-in-range",
            input=(
                np.array([0, 100, 200], dtype=np.int32),
                np.dtype(np.uint8),
                None,
                None,
            ),
            expected=np.array([0, 100, 200], dtype=np.uint8),
        ),
        Expect(
            id="cross-int32-to-uint8-clamp",
            input=(
                np.array([-10, 300], dtype=np.int32),
                np.dtype(np.uint8),
                "clamp",
                None,
            ),
            expected=np.array([0, 255], dtype=np.uint8),
        ),
        Expect(
            id="cross-int32-to-uint8-wrap",
            input=(
                np.array([256, -1], dtype=np.int32),
                np.dtype(np.uint8),
                "wrap",
                None,
            ),
            expected=np.array([0, 255], dtype=np.uint8),
        ),
        Expect(
            id="cross-uint8-to-int32",
            input=(
                np.array([0, 128, 255], dtype=np.uint8),
                np.dtype(np.int32),
                None,
                None,
            ),
            expected=np.array([0, 128, 255], dtype=np.int32),
        ),
        Expect(
            id="cross-uint16-to-int16-clamp",
            input=(
                np.array([0, 32767, 65535], dtype=np.uint16),
                np.dtype(np.int16),
                "clamp",
                None,
            ),
            expected=np.array([0, 32767, 32767], dtype=np.int16),
        ),
        # --- identity ---
        Expect(
            id="same-int32-to-int32",
            input=(
                np.array([1, 2, 3], dtype=np.int32),
                np.dtype(np.int32),
                None,
                None,
            ),
            expected=np.array([1, 2, 3], dtype=np.int32),
        ),
        # --- with scalar_map ---
        Expect(
            id="narrow-with-scalar-map",
            input=(
                np.array([1, 2, 3], dtype=np.int32),
                np.dtype(np.int8),
                None,
                [(np.int64(2), np.int64(20))],
            ),
            expected=np.array([1, 20, 3], dtype=np.int8),
        ),
        Expect(
            id="scalar-map-brings-value-in-range",
            input=(
                np.array([1, 200, 3], dtype=np.int32),
                np.dtype(np.int8),
                None,
                [(np.int64(200), np.int64(50))],
            ),
            expected=np.array([1, 50, 3], dtype=np.int8),
        ),
        Expect(
            id="scalar-map-plus-clamp",
            input=(
                np.array([1, 200, 3], dtype=np.int32),
                np.dtype(np.int8),
                "clamp",
                [(np.int64(1), np.int64(10))],
            ),
            expected=np.array([10, 127, 3], dtype=np.int8),
        ),
    ],
)
def test_cast_array_int_to_int(
    case: Expect[
        tuple[np.ndarray, np.dtype, str | None, list[ScalarMapEntry] | None],
        np.ndarray,
    ],
) -> None:
    """Test cast_array for integer-to-integer casts across narrowing, widening, and cross-sign directions."""
    arr, target_dtype, out_of_range_mode, scalar_map_entries = case.input
    result = _call_cast(
        arr,
        target_dtype=target_dtype,
        out_of_range_mode=out_of_range_mode,
        scalar_map_entries=scalar_map_entries,
    )
    assert case.eq(result, case.expected)
    assert result.dtype == case.expected.dtype


@pytest.mark.parametrize(
    "case",
    [
        ExpectFail(
            id="narrow-out-of-range-error",
            input=(
                np.array([300], dtype=np.int32),
                np.dtype(np.int8),
            ),
            err=ValueError,
            msg="Values out of range",
        ),
        ExpectFail(
            id="negative-for-unsigned-error",
            input=(
                np.array([-1], dtype=np.int32),
                np.dtype(np.uint8),
            ),
            err=ValueError,
            msg="Values out of range",
        ),
    ],
)
def test_cast_array_int_to_int_fail(
    case: ExpectFail[tuple[np.ndarray, np.dtype]],
) -> None:
    """Test that cast_array raises ValueError for out-of-range integer casts with no out_of_range mode."""
    arr, target_dtype = case.input
    with pytest.raises(case.err, match=case.msg):
        _call_cast(arr, target_dtype=target_dtype)


# ---------------------------------------------------------------------------
# cast_array: int → float
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            id="int16-to-float32",
            input=(
                np.array([1, -1, 0, 32767], dtype=np.int16),
                np.dtype(np.float32),
                None,
            ),
            expected=np.array([1.0, -1.0, 0.0, 32767.0], dtype=np.float32),
        ),
        Expect(
            id="int8-to-float64",
            input=(
                np.array([-128, 0, 127], dtype=np.int8),
                np.dtype(np.float64),
                None,
            ),
            expected=np.array([-128.0, 0.0, 127.0], dtype=np.float64),
        ),
        Expect(
            id="int32-to-float64",
            input=(
                np.array([1, 2, 3], dtype=np.int32),
                np.dtype(np.float64),
                None,
            ),
            expected=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        ),
        Expect(
            id="uint8-to-float32",
            input=(
                np.array([0, 128, 255], dtype=np.uint8),
                np.dtype(np.float32),
                None,
            ),
            expected=np.array([0.0, 128.0, 255.0], dtype=np.float32),
        ),
        Expect(
            id="uint16-to-float64",
            input=(
                np.array([0, 65535], dtype=np.uint16),
                np.dtype(np.float64),
                None,
            ),
            expected=np.array([0.0, 65535.0], dtype=np.float64),
        ),
        # --- with scalar_map ---
        Expect(
            id="with-scalar-map",
            input=(
                np.array([1, 2, 3], dtype=np.int32),
                np.dtype(np.float64),
                [(np.int64(2), np.float64(99.0))],
            ),
            expected=np.array([1.0, 99.0, 3.0], dtype=np.float64),
        ),
        Expect(
            id="map-sentinel-to-nan",
            input=(
                np.array([1, -999, 3], dtype=np.int32),
                np.dtype(np.float64),
                [(np.int64(-999), np.float64(np.nan))],
            ),
            expected=np.array([1.0, np.nan, 3.0], dtype=np.float64),
        ),
        # --- empty array ---
        Expect(
            id="empty-array",
            input=(
                np.array([], dtype=np.int32),
                np.dtype(np.float64),
                None,
            ),
            expected=np.array([], dtype=np.float64),
        ),
    ],
)
def test_cast_array_int_to_float(
    case: Expect[
        tuple[np.ndarray, np.dtype, list[ScalarMapEntry] | None],
        np.ndarray,
    ],
) -> None:
    """Test cast_array for integer-to-float casts."""
    arr, target_dtype, scalar_map_entries = case.input
    result = _call_cast(
        arr,
        target_dtype=target_dtype,
        scalar_map_entries=scalar_map_entries,
    )
    assert case.eq(result, case.expected)
    assert result.dtype == case.expected.dtype


# ---------------------------------------------------------------------------
# cast_array: float → int
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        # --- all 5 rounding modes ---
        Expect(
            id="nearest-even",
            input=(
                np.array([1.5, 2.5, 3.7, -1.2], dtype=np.float64),
                np.dtype(np.int32),
                "nearest-even",
                None,
                None,
            ),
            expected=np.array([2, 2, 4, -1], dtype=np.int32),
        ),
        Expect(
            id="towards-zero",
            input=(
                np.array([1.9, -1.9], dtype=np.float64),
                np.dtype(np.int32),
                "towards-zero",
                None,
                None,
            ),
            expected=np.array([1, -1], dtype=np.int32),
        ),
        Expect(
            id="towards-positive",
            input=(
                np.array([1.1, -1.9], dtype=np.float64),
                np.dtype(np.int32),
                "towards-positive",
                None,
                None,
            ),
            expected=np.array([2, -1], dtype=np.int32),
        ),
        Expect(
            id="towards-negative",
            input=(
                np.array([1.9, -1.1], dtype=np.float64),
                np.dtype(np.int32),
                "towards-negative",
                None,
                None,
            ),
            expected=np.array([1, -2], dtype=np.int32),
        ),
        Expect(
            id="nearest-away",
            input=(
                np.array([0.5, -0.5, 1.5, -1.5], dtype=np.float64),
                np.dtype(np.int32),
                "nearest-away",
                None,
                None,
            ),
            expected=np.array([1, -1, 2, -2], dtype=np.int32),
        ),
        # --- out_of_range ---
        Expect(
            id="clamp-int8",
            input=(
                np.array([300.0, -300.0], dtype=np.float64),
                np.dtype(np.int8),
                "nearest-even",
                "clamp",
                None,
            ),
            expected=np.array([127, -128], dtype=np.int8),
        ),
        Expect(
            id="wrap-int8",
            input=(
                np.array([300.0, -300.0], dtype=np.float64),
                np.dtype(np.int8),
                "nearest-even",
                "wrap",
                None,
            ),
            expected=np.array([44, -44], dtype=np.int8),
        ),
        Expect(
            id="clamp-uint8",
            input=(
                np.array([-1.5, 300.5], dtype=np.float64),
                np.dtype(np.uint8),
                "nearest-even",
                "clamp",
                None,
            ),
            expected=np.array([0, 255], dtype=np.uint8),
        ),
        Expect(
            id="wrap-uint8",
            input=(
                np.array([256.0, -1.0], dtype=np.float64),
                np.dtype(np.uint8),
                "towards-zero",
                "wrap",
                None,
            ),
            expected=np.array([0, 255], dtype=np.uint8),
        ),
        # --- float32 source (promotes to float64 internally) ---
        Expect(
            id="float32-source-promotes",
            input=(
                np.array([1.6, 2.4], dtype=np.float32),
                np.dtype(np.int32),
                "nearest-even",
                None,
                None,
            ),
            expected=np.array([2, 2], dtype=np.int32),
        ),
        # --- exact integer values need no rounding ---
        Expect(
            id="exact-integers-no-rounding-needed",
            input=(
                np.array([1.0, -2.0, 0.0], dtype=np.float64),
                np.dtype(np.int32),
                "nearest-even",
                None,
                None,
            ),
            expected=np.array([1, -2, 0], dtype=np.int32),
        ),
        # --- scalar_map ---
        Expect(
            id="map-nan-to-zero",
            input=(
                np.array([1.0, np.nan, 3.0], dtype=np.float64),
                np.dtype(np.int32),
                "nearest-even",
                None,
                [(np.float64(np.nan), np.float64(0.0))],
            ),
            expected=np.array([1, 0, 3], dtype=np.int32),
        ),
        Expect(
            id="map-inf-to-sentinel",
            input=(
                np.array([1.0, np.inf], dtype=np.float64),
                np.dtype(np.int32),
                "nearest-even",
                None,
                [(np.float64(np.inf), np.float64(-1.0))],
            ),
            expected=np.array([1, -1], dtype=np.int32),
        ),
        Expect(
            id="map-neg-inf-to-sentinel",
            input=(
                np.array([-np.inf, 1.0], dtype=np.float64),
                np.dtype(np.int32),
                "nearest-even",
                None,
                [(np.float64(-np.inf), np.float64(-1.0))],
            ),
            expected=np.array([-1, 1], dtype=np.int32),
        ),
        Expect(
            id="map-plus-clamp",
            input=(
                np.array([1.5, np.nan, 300.0], dtype=np.float64),
                np.dtype(np.int8),
                "nearest-even",
                "clamp",
                [(np.float64(np.nan), np.float64(0.0))],
            ),
            expected=np.array([2, 0, 127], dtype=np.int8),
        ),
    ],
)
def test_cast_array_float_to_int(
    case: Expect[
        tuple[np.ndarray, np.dtype, str, str | None, list[ScalarMapEntry] | None],
        np.ndarray,
    ],
) -> None:
    """Test cast_array for float-to-integer casts across rounding modes, out-of-range, and scalar_map."""
    arr, target_dtype, rounding_mode, out_of_range_mode, scalar_map_entries = case.input
    result = _call_cast(
        arr,
        target_dtype=target_dtype,
        rounding_mode=rounding_mode,
        out_of_range_mode=out_of_range_mode,
        scalar_map_entries=scalar_map_entries,
    )
    assert case.eq(result, case.expected)
    assert result.dtype == case.expected.dtype


@pytest.mark.parametrize(
    "case",
    [
        ExpectFail(
            id="nan-no-map",
            input=(
                np.array([np.nan], dtype=np.float64),
                np.dtype(np.int32),
                "nearest-even",
                None,
                None,
            ),
            err=ValueError,
            msg="Cannot cast NaN or Infinity",
        ),
        ExpectFail(
            id="inf-no-map",
            input=(
                np.array([np.inf], dtype=np.float64),
                np.dtype(np.int32),
                "nearest-even",
                None,
                None,
            ),
            err=ValueError,
            msg="Cannot cast NaN or Infinity",
        ),
        ExpectFail(
            id="neg-inf-no-map",
            input=(
                np.array([-np.inf], dtype=np.float64),
                np.dtype(np.int32),
                "nearest-even",
                None,
                None,
            ),
            err=ValueError,
            msg="Cannot cast NaN or Infinity",
        ),
        ExpectFail(
            id="nan-remains-after-partial-map",
            input=(
                np.array([np.nan, 1.0, np.nan], dtype=np.float64),
                np.dtype(np.int32),
                "nearest-even",
                None,
                [(np.float64(1.0), np.float64(2.0))],
            ),
            err=ValueError,
            msg="Cannot cast NaN or Infinity",
        ),
        ExpectFail(
            id="oor-after-rounding",
            input=(
                np.array([127.6], dtype=np.float64),
                np.dtype(np.int8),
                "towards-positive",
                None,
                None,
            ),
            err=ValueError,
            msg="Values out of range",
        ),
        ExpectFail(
            id="float32-nan-no-map",
            input=(
                np.array([np.nan], dtype=np.float32),
                np.dtype(np.int32),
                "nearest-even",
                None,
                None,
            ),
            err=ValueError,
            msg="Cannot cast NaN or Infinity",
        ),
    ],
)
def test_cast_array_float_to_int_fail(
    case: ExpectFail[
        tuple[np.ndarray, np.dtype, str, str | None, list[ScalarMapEntry] | None]
    ],
) -> None:
    """Test that cast_array raises ValueError for invalid float-to-integer casts."""
    arr, target_dtype, rounding_mode, out_of_range_mode, scalar_map_entries = case.input
    with pytest.raises(case.err, match=case.msg):
        _call_cast(
            arr,
            target_dtype=target_dtype,
            rounding_mode=rounding_mode,
            out_of_range_mode=out_of_range_mode,
            scalar_map_entries=scalar_map_entries,
        )


# ---------------------------------------------------------------------------
# cast_array: float → float
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        # --- narrowing ---
        Expect(
            id="narrow-float64-to-float32",
            input=(
                np.array([1.5, 2.5, 3.5], dtype=np.float64),
                np.dtype(np.float32),
                None,
            ),
            expected=np.array([1.5, 2.5, 3.5], dtype=np.float32),
        ),
        Expect(
            id="narrow-float64-to-float16",
            input=(
                np.array([1.0, 0.5, -0.25], dtype=np.float64),
                np.dtype(np.float16),
                None,
            ),
            expected=np.array([1.0, 0.5, -0.25], dtype=np.float16),
        ),
        # --- widening ---
        Expect(
            id="widen-float32-to-float64",
            input=(
                np.array([1.0, 2.0], dtype=np.float32),
                np.dtype(np.float64),
                None,
            ),
            expected=np.array([1.0, 2.0], dtype=np.float64),
        ),
        Expect(
            id="widen-float16-to-float64",
            input=(
                np.array([1.0, -1.0], dtype=np.float16),
                np.dtype(np.float64),
                None,
            ),
            expected=np.array([1.0, -1.0], dtype=np.float64),
        ),
        # --- identity ---
        Expect(
            id="same-float32-to-float32",
            input=(
                np.array([1.0, 2.0], dtype=np.float32),
                np.dtype(np.float32),
                None,
            ),
            expected=np.array([1.0, 2.0], dtype=np.float32),
        ),
        # --- special values ---
        Expect(
            id="nan-propagates",
            input=(
                np.array([1.0, np.nan, 3.0], dtype=np.float64),
                np.dtype(np.float32),
                None,
            ),
            expected=np.array([1.0, np.nan, 3.0], dtype=np.float32),
        ),
        Expect(
            id="inf-propagates",
            input=(
                np.array([np.inf, -np.inf, 1.0], dtype=np.float64),
                np.dtype(np.float32),
                None,
            ),
            expected=np.array([np.inf, -np.inf, 1.0], dtype=np.float32),
        ),
        Expect(
            id="signed-zero-preserved",
            input=(
                np.array([-0.0, 0.0], dtype=np.float64),
                np.dtype(np.float32),
                None,
            ),
            expected=np.array([-0.0, 0.0], dtype=np.float32),
        ),
        # --- with scalar_map ---
        Expect(
            id="with-scalar-map",
            input=(
                np.array([1.0, 2.0, 3.0], dtype=np.float64),
                np.dtype(np.float32),
                [(np.float64(2.0), np.float64(99.0))],
            ),
            expected=np.array([1.0, 99.0, 3.0], dtype=np.float32),
        ),
        Expect(
            id="map-nan-to-value",
            input=(
                np.array([np.nan, 1.0], dtype=np.float64),
                np.dtype(np.float32),
                [(np.float64(np.nan), np.float64(-1.0))],
            ),
            expected=np.array([-1.0, 1.0], dtype=np.float32),
        ),
        Expect(
            id="map-inf-to-value",
            input=(
                np.array([np.inf, 1.0], dtype=np.float64),
                np.dtype(np.float32),
                [(np.float64(np.inf), np.float64(0.0))],
            ),
            expected=np.array([0.0, 1.0], dtype=np.float32),
        ),
        # --- empty array ---
        Expect(
            id="empty-array",
            input=(
                np.array([], dtype=np.float64),
                np.dtype(np.float32),
                None,
            ),
            expected=np.array([], dtype=np.float32),
        ),
    ],
)
def test_cast_array_float_to_float(
    case: Expect[
        tuple[np.ndarray, np.dtype, list[ScalarMapEntry] | None],
        np.ndarray,
    ],
) -> None:
    """Test cast_array for float-to-float casts including special values and scalar_map."""
    arr, target_dtype, scalar_map_entries = case.input
    result = _call_cast(
        arr,
        target_dtype=target_dtype,
        scalar_map_entries=scalar_map_entries,
    )
    assert case.eq(result, case.expected)
    assert result.dtype == case.expected.dtype


# ---------------------------------------------------------------------------
# cast_array: rounding for precision-losing casts (float→float and int→float)
# ---------------------------------------------------------------------------
#
# 1.3 is not exactly representable in any narrower float type, making it a
# good probe value.  For each (source_dtype, target_dtype) narrowing pair we
# pre-compute the two adjacent representable values in the target type (lo, hi)
# and derive the expected result per rounding mode.


def _expected_for_rounding_mode(
    lo: float,
    hi: float,
    mode: str,
    ne_val: float,
) -> float:
    """Return the expected result of rounding a value that lies between *lo* and *hi*.

    *ne_val* is the nearest-even result from numpy (used only for the
    ``"nearest-even"`` case, since reproducing IEEE 754 tie-breaking from
    scratch is error-prone).
    """
    match mode:
        case "nearest-even":
            return ne_val
        case "towards-zero":
            return lo if abs(lo) <= abs(hi) else hi
        case "towards-positive":
            return hi
        case "towards-negative":
            return lo
        case "nearest-away":
            return hi if abs(hi) >= abs(lo) else lo
    raise ValueError(mode)  # pragma: no cover


_NARROWING_FLOAT_PAIRS: list[tuple[type[np.floating], type[np.floating]]] = [
    (np.float64, np.float32),
    (np.float64, np.float16),
    (np.float32, np.float16),
]

_ROUNDING_MODES = [
    "nearest-even",
    "towards-zero",
    "towards-positive",
    "towards-negative",
    "nearest-away",
]


def _make_float_rounding_cases() -> list[Expect]:
    """Generate Expect cases for every (dtype_pair, sign, rounding_mode) combo."""
    cases: list[Expect] = []
    for src_dt, tgt_dt in _NARROWING_FLOAT_PAIRS:
        for sign in [1.0, -1.0]:
            val = src_dt(sign * 1.3)
            ne = tgt_dt(val)  # nearest-even candidate
            ne_wide = src_dt(ne)
            toward = tgt_dt(-np.inf) if ne_wide > val else tgt_dt(np.inf)
            other = np.nextafter(ne, toward)

            lo = min(ne, other, key=float)
            hi = max(ne, other, key=float)

            for mode in _ROUNDING_MODES:
                expected_val = _expected_for_rounding_mode(
                    float(lo), float(hi), mode, float(ne)
                )
                sign_label = "pos" if sign > 0 else "neg"
                cases.append(
                    Expect(
                        id=f"{src_dt.__name__}-{tgt_dt.__name__}-{sign_label}-{mode}",
                        input=(
                            np.array([val], dtype=src_dt),
                            np.dtype(tgt_dt),
                            mode,
                        ),
                        expected=np.array([expected_val], dtype=tgt_dt),
                    )
                )
    return cases


@pytest.mark.parametrize("case", _make_float_rounding_cases())
def test_cast_array_float_rounding(
    case: Expect[tuple[np.ndarray, np.dtype, str], np.ndarray],
) -> None:
    """Test that float-narrowing casts respect the rounding mode for every dtype pair and sign."""
    arr, target_dtype, rounding_mode = case.input
    result = _call_cast(arr, target_dtype=target_dtype, rounding_mode=rounding_mode)
    assert case.eq(result, case.expected)
    assert result.dtype == case.expected.dtype


# ---------------------------------------------------------------------------
# cast_array: int → float rounding (precision loss)
# ---------------------------------------------------------------------------
#
# float32 has 23 mantissa bits, so 2**24 + 1 = 16777217 is not exactly
# representable.  The two adjacent float32 values are 16777216 and 16777218.


def _make_int_to_float_rounding_cases() -> list[Expect]:
    """Generate Expect cases for int-to-float precision loss, rounding mode, and sign."""
    cases: list[Expect] = []
    for sign in [1, -1]:
        val = np.int64(sign * 16777217)
        for mode in _ROUNDING_MODES:
            ne = np.float32(val)
            ne_wide = np.float64(ne)
            toward = (
                np.float32(-np.inf) if ne_wide > np.float64(val) else np.float32(np.inf)
            )
            other = np.nextafter(ne, toward)

            lo = min(ne, other, key=float)
            hi = max(ne, other, key=float)

            expected_val = _expected_for_rounding_mode(
                float(lo), float(hi), mode, float(ne)
            )
            sign_label = "pos" if sign > 0 else "neg"
            cases.append(
                Expect(
                    id=f"int64-float32-{sign_label}-{mode}",
                    input=(
                        np.array([val], dtype=np.int64),
                        np.dtype(np.float32),
                        mode,
                    ),
                    expected=np.array([expected_val], dtype=np.float32),
                )
            )
    return cases


@pytest.mark.parametrize("case", _make_int_to_float_rounding_cases())
def test_cast_array_int_to_float_rounding(
    case: Expect[tuple[np.ndarray, np.dtype, str], np.ndarray],
) -> None:
    """Test that int-to-float casts respect the rounding mode when precision is lost."""
    arr, target_dtype, rounding_mode = case.input
    result = _call_cast(arr, target_dtype=target_dtype, rounding_mode=rounding_mode)
    assert case.eq(result, case.expected)
    assert result.dtype == case.expected.dtype
