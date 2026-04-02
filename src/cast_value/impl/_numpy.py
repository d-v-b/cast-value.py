"""NumPy-based implementation of the cast_value transformation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cast_value.types import (
        OutOfRangeMode,
        RoundingMode,
        ScalarMapEntries,
        ScalarMapEntry,
    )


def _normalize_scalar_map(
    entries: ScalarMapEntries | None,
) -> tuple[ScalarMapEntry, ...]:
    """Normalize scalar map entries to a tuple of (src, tgt) pairs.

    Accepts an iterable of (src, tgt) pairs or a mapping of src -> tgt.
    Returns an empty tuple if entries is None.
    """
    if entries is None:
        return ()
    if isinstance(entries, Mapping):
        return tuple(entries.items())  # ty: ignore[invalid-return-type]
    return tuple(entries)


def apply_scalar_map(work: np.ndarray, entries: Iterable[ScalarMapEntry]) -> None:
    """Apply scalar map entries in-place. Single pass per entry."""
    for src, tgt in entries:
        if isinstance(src, (float, np.floating)) and np.isnan(src):
            mask = np.isnan(work)
        else:
            mask = work == src
        work[mask] = tgt


def round_inplace(arr: np.ndarray, mode: RoundingMode) -> np.ndarray:
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
    msg = f"Unknown rounding mode: {mode}"
    raise ValueError(msg)


def cast_array(
    arr: np.ndarray,
    *,
    target_dtype: np.dtype,
    rounding_mode: RoundingMode,
    out_of_range_mode: OutOfRangeMode | None,
    scalar_map_entries: ScalarMapEntries | None = None,
) -> np.ndarray:
    """Cast an array to target_dtype with rounding, out-of-range, and scalar_map handling.

    Optimized to minimize allocations and passes over the data.
    For the simple case (no scalar_map, no rounding needed, no out-of-range),
    this is essentially just ``arr.astype(target_dtype)``.

    All casts are performed under ``np.errstate(over='raise', invalid='raise')``
    so that numpy overflow or invalid-value warnings become hard errors instead
    of being silently swallowed.
    """
    entries = _normalize_scalar_map(scalar_map_entries)
    with np.errstate(over="raise", invalid="raise"):
        return _cast_array_impl(
            arr,
            target_dtype=target_dtype,
            rounding=rounding_mode,
            out_of_range=out_of_range_mode,
            scalar_map_entries=entries or None,
        )


def check_int_range(
    work: np.ndarray,
    *,
    target_dtype: np.dtype,
    out_of_range: OutOfRangeMode | None,
) -> np.ndarray:
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
            msg = (
                f"Values out of range for {target_dtype} (valid range: [{lo}, {hi}]), "
                f"got values in [{w_min}, {w_max}]. "
                f"Out-of-range values: {oor_vals.ravel()!r}. "
                f"Set out_of_range='clamp' or out_of_range='wrap' to handle this."
            )
            raise ValueError(msg)


def _cast_float(
    arr: np.ndarray,
    target_dtype: np.dtype,
    rounding: RoundingMode,
) -> np.ndarray:
    """Cast a float (or int) array to a float target dtype, respecting the rounding mode.

    numpy's ``astype`` always uses nearest-even. For other rounding modes we
    detect which values lost precision and correct them by choosing between
    the two adjacent representable values in the target dtype.
    """
    result = arr.astype(target_dtype)

    if rounding == "nearest-even":
        return result

    # Widen source to a float type so we can compare. For integer sources,
    # float64 is the widest available; for float sources, keep the original dtype.
    wide_dtype = np.float64 if np.issubdtype(arr.dtype, np.integer) else arr.dtype

    wide_src = arr.astype(wide_dtype)
    roundtrip = result.astype(wide_dtype)
    inexact = roundtrip != wide_src

    # Skip NaN/Inf — they are exact in any float type that supports them.
    if np.issubdtype(wide_dtype, np.floating):
        inexact &= np.isfinite(wide_src)

    if not inexact.any():
        return result

    # For inexact values, ``result`` holds the nearest-even candidate.
    # The other adjacent representable value is one ULP towards the original.
    ne = result[inexact]  # nearest-even candidates
    src = wide_src[inexact]  # original values in wide dtype
    ne_wide = ne.astype(wide_dtype)

    # If nearest-even rounded up (ne > original), the other candidate is one ULP lower.
    # If nearest-even rounded down (ne < original), the other candidate is one ULP higher.
    toward = np.where(ne_wide > src, np.float64(-np.inf), np.float64(np.inf)).astype(
        target_dtype
    )
    other = np.nextafter(ne, toward)
    other_wide = other.astype(wide_dtype)

    match rounding:
        case "towards-zero":
            use_other = np.abs(other_wide) < np.abs(ne_wide)
        case "towards-positive":
            use_other = other_wide > ne_wide
        case "towards-negative":
            use_other = other_wide < ne_wide
        case "nearest-away":
            d_ne = np.abs(ne_wide - src)
            d_other = np.abs(other_wide - src)
            # Pick the closer candidate; break ties away from zero
            use_other = (d_other < d_ne) | (
                (d_other == d_ne) & (np.abs(other_wide) > np.abs(ne_wide))
            )

    corrected = result.copy()
    indices = np.where(inexact)[0]
    corrected[indices[use_other]] = other[use_other]
    return corrected


def _cast_array_impl(
    arr: np.ndarray,
    *,
    target_dtype: np.dtype,
    rounding: RoundingMode,
    out_of_range: OutOfRangeMode | None,
    scalar_map_entries: Iterable[ScalarMapEntry] | None,
) -> np.ndarray:
    src_type: Literal["int", "float"] = (
        "int" if np.issubdtype(arr.dtype, np.integer) else "float"
    )
    tgt_type: Literal["int", "float"] = (
        "int" if np.issubdtype(target_dtype, np.integer) else "float"
    )
    has_map = bool(scalar_map_entries)

    match (src_type, tgt_type, has_map):
        # float→float or int→float without scalar_map
        case (_, "float", False):
            return _cast_float(arr, target_dtype, rounding)

        # int→float with scalar_map — widen to float64, apply map, cast
        case ("int", "float", True):
            assert scalar_map_entries is not None
            work = arr.astype(np.float64)
            apply_scalar_map(work, scalar_map_entries)
            return _cast_float(work, target_dtype, rounding)

        # float→float with scalar_map — copy, apply map, cast
        case ("float", "float", True):
            assert scalar_map_entries is not None
            work = arr.copy()
            apply_scalar_map(work, scalar_map_entries)
            return _cast_float(work, target_dtype, rounding)

        # int→int without scalar_map — range check then astype
        case ("int", "int", False):
            if arr.dtype.itemsize > target_dtype.itemsize or arr.dtype != target_dtype:
                return check_int_range(
                    arr, target_dtype=target_dtype, out_of_range=out_of_range
                )
            return arr.astype(target_dtype)

        # int→int with scalar_map — widen to int64, apply map, range check
        case ("int", "int", True):
            assert scalar_map_entries is not None
            work = arr.astype(np.int64)
            apply_scalar_map(work, scalar_map_entries)
            return check_int_range(
                work, target_dtype=target_dtype, out_of_range=out_of_range
            )

        # float→int (with or without scalar_map) — rounding + range check
        case ("float", "int", _):
            work = arr.astype(np.float64) if arr.dtype != np.float64 else arr.copy()

            if scalar_map_entries:
                apply_scalar_map(work, scalar_map_entries)

            bad = np.isnan(work) | np.isinf(work)
            if bad.any():
                msg = "Cannot cast NaN or Infinity to integer type without scalar_map"
                raise ValueError(msg)

            work = round_inplace(work, rounding)
            return check_int_range(
                work, target_dtype=target_dtype, out_of_range=out_of_range
            )

    msg = f"Unhandled type combination: src={src_type}, tgt={tgt_type}"  # pragma: no cover
    raise AssertionError(msg)  # pragma: no cover
