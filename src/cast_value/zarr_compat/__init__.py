from __future__ import annotations

try:
    import zarr  # noqa: F401
except ImportError as e:
    _MSG = (
        "The 'zarr' package is required to use cast_value.zarr_compat. "
        "Install it with: pip install 'cast-value[zarr]'"
    )
    raise ImportError(_MSG) from e

from cast_value.zarr_compat.v1 import CastValue, CastValueNumpy, CastValueRust

__all__ = ["CastValue", "CastValueNumpy", "CastValueRust"]
