from __future__ import annotations

from typing import Any, Literal

import numpy as np

from cast_value.types import MapEntry, OutOfRangeMode, RoundingMode, ScalarMapJSON


def apply_scalar_map(work: np.ndarray[Any, np.dtype[Any]], entries: list[MapEntry]) -> None:
    """Apply scalar map entries in-place. Single pass per entry."""
    for src, tgt in entries:
        if isinstance(src, (float, np.floating)) and np.isnan(src):
            mask = np.isnan(work)
        else:
            mask = work == src
        work[mask] = tgt


def round_inplace(
    arr: np.ndarray[Any, np.dtype[Any]], mode: RoundingMode
) -> np.ndarray[Any, np.dtype[Any]]:
    """Round array, returning result (may or may not be a new array).

    For nearest-away, requires 3 numpy ops. All others are a single op.
    """
    match mode:
        case "nearest-even":
            return np.rint(arr)  # type: ignore [no-any-return]
        case "towards-zero":
            return np.trunc(arr)  # type: ignore [no-any-return]
        case "towards-positive":
            return np.ceil(arr)  # type: ignore [no-any-return]
        case "towards-negative":
            return np.floor(arr)  # type: ignore [no-any-return]
        case "nearest-away":
            return np.sign(arr) * np.floor(np.abs(arr) + 0.5)  # type: ignore [no-any-return]
    raise ValueError(f"Unknown rounding mode: {mode}")


def cast_array(
    arr: np.ndarray[Any, np.dtype[Any]],
    *,
    target_dtype: np.dtype[Any],
    rounding_mode: RoundingMode,
    out_of_range_mode: OutOfRangeMode | None,
    scalar_map_entries: list[MapEntry] | None,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Cast an array to target_dtype with rounding, out-of-range, and scalar_map handling.

    Optimized to minimize allocations and passes over the data.
    For the simple case (no scalar_map, no rounding needed, no out-of-range),
    this is essentially just ``arr.astype(target_dtype)``.

    All casts are performed under ``np.errstate(over='raise', invalid='raise')``
    so that numpy overflow or invalid-value warnings become hard errors instead
    of being silently swallowed.
    """
    with np.errstate(over="raise", invalid="raise"):
        return _cast_array_impl(
            arr,
            target_dtype=target_dtype,
            rounding=rounding_mode,
            out_of_range=out_of_range_mode,
            scalar_map_entries=scalar_map_entries,
        )


def check_int_range(
    work: np.ndarray[Any, np.dtype[Any]],
    *,
    target_dtype: np.dtype[Any],
    out_of_range: OutOfRangeMode | None,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Check integer range and apply out-of-range handling, then cast."""
    info = np.iinfo(target_dtype)
    lo, hi = int(info.min), int(info.max)
    w_min, w_max = int(work.min()), int(work.max())
    if w_min >= lo and w_max <= hi:
        return work.astype(target_dtype)
    match out_of_range:
        case "clamp":
            return np.clip(work, lo, hi).astype(target_dtype)
        case "wrap":
            range_size = hi - lo + 1
            return ((work.astype(np.int64) - lo) % range_size + lo).astype(target_dtype)
        case None:
            oor_vals = work[(work < lo) | (work > hi)]
            raise ValueError(
                f"Values out of range for {target_dtype} (valid range: [{lo}, {hi}]), "
                f"got values in [{w_min}, {w_max}]. "
                f"Out-of-range values: {oor_vals.ravel()!r}. "
                f"Set out_of_range='clamp' or out_of_range='wrap' to handle this."
            )


def _cast_array_impl(
    arr: np.ndarray[Any, np.dtype[Any]],
    *,
    target_dtype: np.dtype[Any],
    rounding: RoundingMode,
    out_of_range: OutOfRangeMode | None,
    scalar_map_entries: list[MapEntry] | None,
) -> np.ndarray[Any, np.dtype[Any]]:
    src_type: Literal["int", "float"] = "int" if np.issubdtype(arr.dtype, np.integer) else "float"
    tgt_type: Literal["int", "float"] = (
        "int" if np.issubdtype(target_dtype, np.integer) else "float"
    )
    has_map = bool(scalar_map_entries)

    match (src_type, tgt_type, has_map):
        # float→float or int→float without scalar_map — single astype
        case (_, "float", False):
            return arr.astype(target_dtype)

        # int→float with scalar_map — widen to float64, apply map, cast
        case ("int", "float", True):
            work = arr.astype(np.float64)
            apply_scalar_map(work, scalar_map_entries)  # type: ignore[arg-type]
            return work.astype(target_dtype)

        # float→float with scalar_map — copy, apply map, cast
        case ("float", "float", True):
            work = arr.copy()
            apply_scalar_map(work, scalar_map_entries)  # type: ignore[arg-type]
            return work.astype(target_dtype)

        # int→int without scalar_map — range check then astype
        case ("int", "int", False):
            if arr.dtype.itemsize > target_dtype.itemsize or arr.dtype != target_dtype:
                return check_int_range(arr, target_dtype=target_dtype, out_of_range=out_of_range)
            return arr.astype(target_dtype)

        # int→int with scalar_map — widen to int64, apply map, range check
        case ("int", "int", True):
            work = arr.astype(np.int64)
            apply_scalar_map(work, scalar_map_entries)  # type: ignore[arg-type]
            return check_int_range(work, target_dtype=target_dtype, out_of_range=out_of_range)

        # float→int (with or without scalar_map) — rounding + range check
        case ("float", "int", _):
            if arr.dtype != np.float64:
                work = arr.astype(np.float64)
            else:
                work = arr.copy()

            if scalar_map_entries:
                apply_scalar_map(work, scalar_map_entries)

            bad = np.isnan(work) | np.isinf(work)
            if bad.any():
                raise ValueError("Cannot cast NaN or Infinity to integer type without scalar_map")

            work = round_inplace(work, rounding)
            return check_int_range(work, target_dtype=target_dtype, out_of_range=out_of_range)

    raise AssertionError(
        f"Unhandled type combination: src={src_type}, tgt={tgt_type}"
    )  # pragma: no cover


def extract_raw_map(data: ScalarMapJSON | None, direction: str) -> dict[str, str] | None:
    """Extract raw string mapping from scalar_map JSON for 'encode' or 'decode'."""
    if data is None:
        return None
    raw: dict[str, str] = {}
    pairs = data.get(direction, [])
    for src, tgt in pairs:  # type: ignore[attr-defined]
        raw[str(src)] = str(tgt)
    return raw or None
