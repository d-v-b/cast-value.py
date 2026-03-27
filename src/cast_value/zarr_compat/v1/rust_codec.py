"""Cast-value codec backed by the cast-value-rs Rust implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from cast_value.zarr_compat.v1._base import CastValueBase

if TYPE_CHECKING:
    from cast_value.types import MapEntry


def _dtype_to_str(dtype: np.dtype[Any]) -> str:
    """Convert a numpy dtype to the string name expected by cast-value-rs."""
    return dtype.name


def _convert_scalar_map(
    entries: list[MapEntry] | None,
) -> list[tuple[int | float, int | float]] | None:
    """Convert scalar map entries to plain Python types for cast-value-rs.

    cast-value-rs accepts Python int/float but rejects np.floating scalars.
    """
    if entries is None:
        return None
    result: list[tuple[int | float, int | float]] = []
    for src, tgt in entries:
        src_py: int | float = int(src) if isinstance(src, np.integer) else float(src)
        tgt_py: int | float = int(tgt) if isinstance(tgt, np.integer) else float(tgt)
        result.append((src_py, tgt_py))
    return result


@dataclass(frozen=True, init=False)
class CastValueRust(CastValueBase):
    """Cast-value codec backed by the cast-value-rs Rust implementation."""

    def _cast_array(
        self,
        arr: np.ndarray[Any, np.dtype[Any]],
        *,
        target_dtype: np.dtype[Any],
        scalar_map_entries: list[MapEntry] | None,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        from cast_value_rs import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
            cast_array as rs_cast_array,
        )

        return rs_cast_array(
            arr=arr,
            target_dtype=_dtype_to_str(target_dtype),  # ty: ignore[invalid-argument-type]
            rounding_mode=self.rounding,
            out_of_range_mode=self.out_of_range,
            scalar_map_entries=_convert_scalar_map(scalar_map_entries),
        )
