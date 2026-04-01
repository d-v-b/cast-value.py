from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal, NotRequired, TypeAlias, TypedDict

import numpy as np

NumericScalar: TypeAlias = np.integer | np.floating

RoundingMode = Literal[
    "nearest-even",
    "towards-zero",
    "towards-positive",
    "towards-negative",
    "nearest-away",
]

OutOfRangeMode = Literal["clamp", "wrap"]


class ScalarMapJSON(TypedDict):
    """
    JSON representation of the scalar_map codec configuration field.

    This type models permitted values for the [`scalar_map`](https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/cast_value#scalar_map) field in the
    [`configuration`](https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/cast_value#configuration) field of the `cast_value` codec metadata.

    See the [`cast_value` spec](https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/cast_value) for details.
    """

    encode: NotRequired[list[tuple[object, object]]]
    decode: NotRequired[list[tuple[object, object]]]


# Pre-parsed scalar map entry: (source_scalar, target_scalar)
ScalarMapEntry = tuple[NumericScalar, NumericScalar]

# Accepted types for scalar_map_entries parameters
ScalarMapEntries: TypeAlias = (
    Iterable[ScalarMapEntry] | Mapping[NumericScalar, NumericScalar]
)
