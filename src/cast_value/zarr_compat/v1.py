from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import numpy as np

from zarr.abc.codec import ArrayArrayCodec
from zarr.core.common import JSON, parse_named_configuration
from zarr.core.dtype import get_data_type_from_json

from cast_value.core import cast_array, extract_raw_map
from cast_value.types import MapEntry, OutOfRangeMode, RoundingMode, ScalarMapJSON

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import NDBuffer
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType


def _parse_map_entries(
    mapping: dict[str, str],
    src_dtype: ZDType[TBaseDType, TBaseScalar],
    tgt_dtype: ZDType[TBaseDType, TBaseScalar],
) -> list[MapEntry]:
    """Pre-parse a scalar map dict into a list of (src, tgt) tuples.

    Each entry's source value is deserialized using ``src_dtype`` and its target
    value using ``tgt_dtype``, preserving full precision for both data types.
    """
    entries: list[MapEntry] = []
    for src_str, tgt_str in mapping.items():
        src = src_dtype.from_json_scalar(src_str, zarr_format=3)
        tgt = tgt_dtype.from_json_scalar(tgt_str, zarr_format=3)
        entries.append((src, tgt))  # type: ignore[arg-type]
    return entries


@dataclass(frozen=True)
class CastValue(ArrayArrayCodec):
    """Cast-value array-to-array codec.

    Value-converts array elements to a new data type during encoding,
    and back to the original data type during decoding.

    Parameters
    ----------
    data_type : str
        Target zarr v3 data type name (e.g. "uint8", "float32").
    rounding : RoundingMode
        How to round when exact representation is impossible. Default is "nearest-even".
    out_of_range : OutOfRangeMode or None
        What to do when a value is outside the target's range.
        None means error. "clamp" clips to range. "wrap" uses modular arithmetic
        (only valid for integer types).
    scalar_map : dict or None
        Explicit value overrides as JSON: {"encode": [[src, tgt], ...], "decode": [[src, tgt], ...]}.
    """

    is_fixed_size = True

    dtype: ZDType[TBaseDType, TBaseScalar]
    rounding: RoundingMode
    out_of_range: OutOfRangeMode | None
    scalar_map: ScalarMapJSON | None

    def __init__(
        self,
        *,
        data_type: str | ZDType[TBaseDType, TBaseScalar],
        rounding: RoundingMode = "nearest-even",
        out_of_range: OutOfRangeMode | None = None,
        scalar_map: ScalarMapJSON | None = None,
    ) -> None:
        if isinstance(data_type, str):
            dtype = get_data_type_from_json(data_type, zarr_format=3)
        else:
            dtype = data_type
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "rounding", rounding)
        object.__setattr__(self, "out_of_range", out_of_range)
        object.__setattr__(self, "scalar_map", scalar_map)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, "cast_value", require_configuration=True
        )
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        config: dict[str, JSON] = {"data_type": cast(JSON, self.dtype.to_json(zarr_format=3))}
        if self.rounding != "nearest-even":
            config["rounding"] = self.rounding
        if self.out_of_range is not None:
            config["out_of_range"] = self.out_of_range
        if self.scalar_map is not None:
            config["scalar_map"] = cast(JSON, self.scalar_map)
        return {"name": "cast_value", "configuration": config}

    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGrid,
    ) -> None:
        source_native = dtype.to_native_dtype()
        target_native = self.dtype.to_native_dtype()
        for label, dt in [("source", source_native), ("target", target_native)]:
            if not np.issubdtype(dt, np.integer) and not np.issubdtype(dt, np.floating):
                raise ValueError(
                    f"cast_value codec only supports integer and floating-point data types. "
                    f"Got {label} dtype {dt}."
                )
        if self.out_of_range == "wrap" and not np.issubdtype(target_native, np.integer):
            raise ValueError("out_of_range='wrap' is only valid for integer target types.")
        # Check that int→float casts won't silently lose precision.
        # A float type with `m` mantissa bits can exactly represent all integers
        # in [-2**m, 2**m]. If the integer type's range exceeds that, the cast is lossy.
        if np.issubdtype(source_native, np.integer) and np.issubdtype(target_native, np.floating):
            int_info = np.iinfo(source_native)  # type: ignore[type-var]
            mantissa_bits = np.finfo(target_native).nmant  # type: ignore[arg-type]
            max_exact_int = 2**mantissa_bits
            if int_info.max > max_exact_int or int_info.min < -max_exact_int:
                raise ValueError(
                    f"Casting {source_native} to {target_native} may silently lose precision. "
                    f"{target_native} can only exactly represent integers up to 2**{mantissa_bits} "
                    f"({max_exact_int}), but {source_native} has range "
                    f"[{int_info.min}, {int_info.max}]."
                )
        # Same check for float→int decode direction
        if np.issubdtype(target_native, np.integer) and np.issubdtype(source_native, np.floating):
            int_info = np.iinfo(target_native)  # type: ignore[type-var]
            mantissa_bits = np.finfo(source_native).nmant  # type: ignore[arg-type]
            max_exact_int = 2**mantissa_bits
            if int_info.max > max_exact_int or int_info.min < -max_exact_int:
                raise ValueError(
                    f"Casting {source_native} to {target_native} may silently lose precision. "
                    f"{source_native} can only exactly represent integers up to 2**{mantissa_bits} "
                    f"({max_exact_int}), but {target_native} has range "
                    f"[{int_info.min}, {int_info.max}]."
                )

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        target_zdtype = self.dtype
        target_native = target_zdtype.to_native_dtype()
        source_native = chunk_spec.dtype.to_native_dtype()

        fill = chunk_spec.fill_value
        fill_arr = np.array([fill], dtype=source_native)

        encode_raw = extract_raw_map(self.scalar_map, "encode")
        encode_entries = (
            _parse_map_entries(encode_raw, chunk_spec.dtype, self.dtype) if encode_raw else None
        )

        new_fill_arr = cast_array(
            fill_arr,
            target_dtype=target_native,
            rounding_mode=self.rounding,
            out_of_range_mode=self.out_of_range,
            scalar_map_entries=encode_entries,
        )
        new_fill = target_native.type(new_fill_arr[0])

        return replace(chunk_spec, dtype=target_zdtype, fill_value=new_fill)

    def _encode_sync(
        self,
        chunk_array: NDBuffer,
        _chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        arr = chunk_array.as_ndarray_like()
        target_native = self.dtype.to_native_dtype()

        encode_raw = extract_raw_map(self.scalar_map, "encode")
        encode_entries = (
            _parse_map_entries(encode_raw, _chunk_spec.dtype, self.dtype) if encode_raw else None
        )

        result = cast_array(
            np.asarray(arr),
            target_dtype=target_native,
            rounding_mode=self.rounding,
            out_of_range_mode=self.out_of_range,
            scalar_map_entries=encode_entries,
        )
        return chunk_array.__class__.from_ndarray_like(result)

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        return self._encode_sync(chunk_array, chunk_spec)

    def _decode_sync(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        arr = chunk_array.as_ndarray_like()
        target_native = chunk_spec.dtype.to_native_dtype()

        decode_raw = extract_raw_map(self.scalar_map, "decode")
        decode_entries = (
            _parse_map_entries(decode_raw, self.dtype, chunk_spec.dtype) if decode_raw else None
        )

        result = cast_array(
            np.asarray(arr),
            target_dtype=target_native,
            rounding_mode=self.rounding,
            out_of_range_mode=self.out_of_range,
            scalar_map_entries=decode_entries,
        )
        return chunk_array.__class__.from_ndarray_like(result)

    async def _decode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        return self._decode_sync(chunk_array, chunk_spec)

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        source_itemsize = chunk_spec.dtype.to_native_dtype().itemsize
        target_itemsize = self.dtype.to_native_dtype().itemsize
        if source_itemsize == 0:
            return 0
        num_elements = input_byte_length // source_itemsize
        return num_elements * target_itemsize
