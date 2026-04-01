"""Copyright (c) 2026 Davis Bennett. All rights reserved.

cast-value: Python implementation of the `cast_value` codec.
"""

from __future__ import annotations

from ._version import version as __version__
from .impl._numpy import cast_array
from .types import (
    NumericScalar,
    OutOfRangeMode,
    RoundingMode,
    ScalarMapEntries,
    ScalarMapEntry,
    ScalarMapJSON,
)
from .zarr_compat.v1 import CastValueNumpyV1, CastValueRustV1

__all__ = [
    "CastValueNumpyV1",
    "CastValueRustV1",
    "NumericScalar",
    "OutOfRangeMode",
    "RoundingMode",
    "ScalarMapEntries",
    "ScalarMapEntry",
    "ScalarMapJSON",
    "__version__",
    "cast_array",
]
