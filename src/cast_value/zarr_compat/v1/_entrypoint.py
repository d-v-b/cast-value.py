"""Entrypoint for zarr codec discovery.

Exports ``CastValue`` — the best available codec implementation.
Uses the Rust backend if ``cast-value-rs`` is installed, otherwise
falls back to the pure-NumPy backend.

Registered as ``cast_value`` in the ``zarr.codecs`` entrypoint group
(see ``pyproject.toml``).
"""

from __future__ import annotations

try:
    import cast_value_rs as _  # noqa: F401

    from cast_value.zarr_compat.v1.rust_codec import CastValueRustV1 as CastValue
except ModuleNotFoundError:
    from cast_value.zarr_compat.v1.numpy_codec import (
        CastValueNumpyV1 as CastValue,  # type: ignore[assignment]
    )

__all__ = ["CastValue"]
