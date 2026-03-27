"""cast_value codec implementations for the current zarr-python codec API.

Everything in this package depends on zarr-python's ``ArrayArrayCodec`` ABC
and related types. When zarr-python ships a new codec interface, a ``v2``
package can be added alongside this one.
"""

from __future__ import annotations

from cast_value.zarr_compat.v1._base import CastValueBase, parse_map_entries
from cast_value.zarr_compat.v1.numpy_codec import CastValueNumpy
from cast_value.zarr_compat.v1.rust_codec import CastValueRust

# Backwards-compatible alias
CastValue = CastValueNumpy

__all__ = [
    "CastValue",
    "CastValueBase",
    "CastValueNumpy",
    "CastValueRust",
    "parse_map_entries",
]
