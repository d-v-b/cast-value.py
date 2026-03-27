from __future__ import annotations

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
    """JSON representation of the scalar_map codec configuration field."""

    encode: NotRequired[list[tuple[object, object]]]
    decode: NotRequired[list[tuple[object, object]]]


# Pre-parsed scalar map entry: (source_scalar, target_scalar)
MapEntry = tuple[NumericScalar, NumericScalar]
