"""Codec metadata parsing utilities shared across zarr API versions.

These depend on zarr's dtype system (``ZDType.from_json_scalar``) but not
on any specific version of the zarr codec API (``ArrayArrayCodec``, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType

    from cast_value.types import ScalarMapEntry, ScalarMapJSON


def extract_raw_map(
    data: ScalarMapJSON | None, direction: str
) -> dict[str, str] | None:
    """Extract raw string mapping from scalar_map JSON for 'encode' or 'decode'."""
    if data is None:
        return None
    raw: dict[str, str] = {}
    pairs = data.get(direction, [])
    for src, tgt in pairs:  # type: ignore[attr-defined]
        raw[str(src)] = str(tgt)
    return raw or None


def parse_map_entries(
    mapping: Mapping[str, str],
    src_dtype: ZDType[TBaseDType, TBaseScalar],
    tgt_dtype: ZDType[TBaseDType, TBaseScalar],
) -> tuple[ScalarMapEntry, ...]:
    """Pre-parse a scalar map dict into a tuple of (src, tgt) pairs.

    Each entry's source value is deserialized using ``src_dtype`` and its target
    value using ``tgt_dtype``, preserving full precision for both data types.
    """
    return tuple(  # type: ignore[return-value]  # ty: ignore[invalid-return-type]
        (
            src_dtype.from_json_scalar(src_str, zarr_format=3),
            tgt_dtype.from_json_scalar(tgt_str, zarr_format=3),
        )
        for src_str, tgt_str in mapping.items()
    )
