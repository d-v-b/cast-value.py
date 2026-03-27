"""Cast-value codec backed by the pure-numpy implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cast_value.core import cast_array
from cast_value.zarr_compat.v1._base import CastValueBase

if TYPE_CHECKING:
    import numpy as np

    from cast_value.types import MapEntry


@dataclass(frozen=True, init=False)
class CastValueNumpy(CastValueBase):
    """Cast-value codec backed by the pure-numpy implementation."""

    def _cast_array(
        self,
        arr: np.ndarray,
        *,
        target_dtype: np.dtype,
        scalar_map_entries: list[MapEntry] | None,
    ) -> np.ndarray:
        return cast_array(
            arr,
            target_dtype=target_dtype,
            rounding_mode=self.rounding,
            out_of_range_mode=self.out_of_range,
            scalar_map_entries=scalar_map_entries,
        )
