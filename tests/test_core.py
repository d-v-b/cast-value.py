from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from conftest import Expect, ExpectFail

from cast_value.core import (
    apply_scalar_map,
    cast_array,
    check_int_range,
    extract_raw_map,
    round_inplace,
)

if TYPE_CHECKING:
    from cast_value.types import MapEntry, ScalarMapJSON


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
    ],
)
def test_apply_scalar_map(
    case: Expect[tuple[np.ndarray, list[MapEntry]], np.ndarray],
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
            input=(np.array([1.1, -1.1, 0.0, -0.9], dtype=np.float64), "towards-positive"),
            expected=np.array([2.0, -1.0, 0.0, -0.0], dtype=np.float64),
        ),
        Expect(
            id="towards-negative",
            input=(np.array([1.9, -1.1, 0.0, 0.9], dtype=np.float64), "towards-negative"),
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
    ],
)
def test_round_inplace(case: Expect[tuple[np.ndarray, str], np.ndarray]) -> None:
    """Test that round_inplace rounds according to the specified mode."""
    arr, mode = case.input
    result = round_inplace(arr, mode)
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
        round_inplace(arr, mode)


# ---------------------------------------------------------------------------
# check_int_range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            id="in-range-no-action",
            input=(
                np.array([0, 100, 255], dtype=np.int64),
                np.dtype(np.uint8),
                None,
            ),
            expected=np.array([0, 100, 255], dtype=np.uint8),
        ),
        Expect(
            id="clamp-out-of-range",
            input=(
                np.array([-10, 300], dtype=np.int64),
                np.dtype(np.uint8),
                "clamp",
            ),
            expected=np.array([0, 255], dtype=np.uint8),
        ),
        Expect(
            id="wrap-unsigned",
            input=(
                np.array([256, -1, 512], dtype=np.int64),
                np.dtype(np.uint8),
                "wrap",
            ),
            expected=np.array([0, 255, 0], dtype=np.uint8),
        ),
        Expect(
            id="wrap-signed",
            input=(
                np.array([128, -129], dtype=np.int64),
                np.dtype(np.int8),
                "wrap",
            ),
            expected=np.array([-128, 127], dtype=np.int8),
        ),
        Expect(
            id="clamp-signed",
            input=(
                np.array([-1000, 1000], dtype=np.int64),
                np.dtype(np.int8),
                "clamp",
            ),
            expected=np.array([-128, 127], dtype=np.int8),
        ),
        Expect(
            id="widen-no-range-issue",
            input=(
                np.array([10, 20], dtype=np.int32),
                np.dtype(np.int64),
                None,
            ),
            expected=np.array([10, 20], dtype=np.int64),
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
        out_of_range=out_of_range,
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
            out_of_range=out_of_range,
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


# ---------------------------------------------------------------------------
# cast_array
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            id="float64-to-float32",
            input=(
                np.array([1.5, 2.5, 3.5], dtype=np.float64),
                np.dtype(np.float32),
                "nearest-even",
                None,
                None,
            ),
            expected=np.array([1.5, 2.5, 3.5], dtype=np.float32),
        ),
        Expect(
            id="float32-to-float64",
            input=(
                np.array([1.0, 2.0], dtype=np.float32),
                np.dtype(np.float64),
                "nearest-even",
                None,
                None,
            ),
            expected=np.array([1.0, 2.0], dtype=np.float64),
        ),
        Expect(
            id="int32-to-float64",
            input=(
                np.array([1, 2, 3], dtype=np.int32),
                np.dtype(np.float64),
                "nearest-even",
                None,
                None,
            ),
            expected=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        ),
        Expect(
            id="int32-to-int32-identity",
            input=(
                np.array([1, 2, 3], dtype=np.int32),
                np.dtype(np.int32),
                "nearest-even",
                None,
                None,
            ),
            expected=np.array([1, 2, 3], dtype=np.int32),
        ),
        Expect(
            id="int32-to-int8-in-range",
            input=(
                np.array([0, 127, -128], dtype=np.int32),
                np.dtype(np.int8),
                "nearest-even",
                None,
                None,
            ),
            expected=np.array([0, 127, -128], dtype=np.int8),
        ),
        Expect(
            id="int32-to-int8-clamp",
            input=(
                np.array([0, 300, -200], dtype=np.int32),
                np.dtype(np.int8),
                "nearest-even",
                "clamp",
                None,
            ),
            expected=np.array([0, 127, -128], dtype=np.int8),
        ),
        Expect(
            id="int32-to-uint8-wrap",
            input=(
                np.array([256, -1], dtype=np.int32),
                np.dtype(np.uint8),
                "nearest-even",
                "wrap",
                None,
            ),
            expected=np.array([0, 255], dtype=np.uint8),
        ),
        Expect(
            id="float64-to-int32-nearest-even",
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
            id="float64-to-int32-towards-zero",
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
            id="float64-to-int32-towards-positive",
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
            id="float64-to-int32-towards-negative",
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
            id="float64-to-int32-nearest-away",
            input=(
                np.array([0.5, -0.5, 1.5, -1.5], dtype=np.float64),
                np.dtype(np.int32),
                "nearest-away",
                None,
                None,
            ),
            expected=np.array([1, -1, 2, -2], dtype=np.int32),
        ),
        Expect(
            id="float64-to-int8-clamp",
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
            id="float32-to-int32-promotes-to-float64",
            input=(
                np.array([1.6, 2.4], dtype=np.float32),
                np.dtype(np.int32),
                "nearest-even",
                None,
                None,
            ),
            expected=np.array([2, 2], dtype=np.int32),
        ),
        Expect(
            id="int-to-float-with-scalar-map",
            input=(
                np.array([1, 2, 3], dtype=np.int32),
                np.dtype(np.float64),
                "nearest-even",
                None,
                [(np.int64(2), np.float64(99.0))],
            ),
            expected=np.array([1.0, 99.0, 3.0], dtype=np.float64),
        ),
        Expect(
            id="float-to-float-with-scalar-map",
            input=(
                np.array([1.0, 2.0, 3.0], dtype=np.float64),
                np.dtype(np.float32),
                "nearest-even",
                None,
                [(np.float64(2.0), np.float64(99.0))],
            ),
            expected=np.array([1.0, 99.0, 3.0], dtype=np.float32),
        ),
        Expect(
            id="int-to-int-with-scalar-map",
            input=(
                np.array([1, 2, 3], dtype=np.int32),
                np.dtype(np.int8),
                "nearest-even",
                None,
                [(np.int64(2), np.int64(20))],
            ),
            expected=np.array([1, 20, 3], dtype=np.int8),
        ),
        Expect(
            id="float-to-int-scalar-map-nan-to-zero",
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
            id="int8-to-int64-widen",
            input=(
                np.array([1, 2], dtype=np.int8),
                np.dtype(np.int64),
                "nearest-even",
                None,
                None,
            ),
            expected=np.array([1, 2], dtype=np.int64),
        ),
        Expect(
            id="int32-to-uint8-in-range",
            input=(
                np.array([0, 100, 200], dtype=np.int32),
                np.dtype(np.uint8),
                "nearest-even",
                None,
                None,
            ),
            expected=np.array([0, 100, 200], dtype=np.uint8),
        ),
    ],
)
def test_cast_array(
    case: Expect[
        tuple[np.ndarray, np.dtype, str, str | None, list[MapEntry] | None],
        np.ndarray,
    ],
) -> None:
    """Test that cast_array produces the expected output for various type combinations."""
    arr, target_dtype, rounding_mode, out_of_range_mode, scalar_map_entries = case.input
    result = cast_array(
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
            id="int-narrowing-out-of-range-no-mode",
            input=(
                np.array([300], dtype=np.int32),
                np.dtype(np.int8),
                "nearest-even",
                None,
                None,
            ),
            err=ValueError,
            msg="Values out of range",
        ),
        ExpectFail(
            id="float-nan-to-int-no-scalar-map",
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
            id="float-inf-to-int-no-scalar-map",
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
            id="float-neg-inf-to-int-no-scalar-map",
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
    ],
)
def test_cast_array_fail(
    case: ExpectFail[
        tuple[np.ndarray, np.dtype, str, str | None, list[MapEntry] | None]
    ],
) -> None:
    """Test that cast_array raises ValueError for invalid casts."""
    arr, target_dtype, rounding_mode, out_of_range_mode, scalar_map_entries = case.input
    with pytest.raises(case.err, match=case.msg):
        cast_array(
            arr,
            target_dtype=target_dtype,
            rounding_mode=rounding_mode,
            out_of_range_mode=out_of_range_mode,
            scalar_map_entries=scalar_map_entries,
        )
