from __future__ import annotations

try:
    import zarr  # noqa: F401
except ImportError as e:
    _msg = (
        "The 'zarr' package is required to use cast_value.zarr_compat. "
        "Install it with: pip install 'cast-value[zarr]'"
    )
    raise ImportError(_msg) from e

from cast_value.zarr_compat.v1 import CastValue

__all__ = ["CastValue"]
