from __future__ import annotations

from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

import numpy as np

NumericScalar: TypeAlias = np.integer[Any] | np.floating[Any]

RoundingMode = Literal[
    "nearest-even",
    "towards-zero",
    "towards-positive",
    "towards-negative",
    "nearest-away",
]

OutOfRangeMode = Literal["clamp", "wrap"]


class ScalarMapJSON(TypedDict):
    encode: NotRequired[tuple[tuple[object, object]]]
    decode: NotRequired[tuple[tuple[object, object]]]


# Pre-parsed scalar map entry: (source_scalar, target_scalar)
MapEntry = tuple[NumericScalar, NumericScalar]
