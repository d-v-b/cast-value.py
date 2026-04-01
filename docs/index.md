# cast-value

Python implementation of the
[`cast_value` codec](https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/cast_value)
for [Zarr V3](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html).

The `cast_value` codec converts array elements between numeric data types during
encoding and decoding, with configurable rounding, out-of-range handling, and
explicit scalar mappings.

## Installation

```bash
pip install cast-value
```

For the optional Rust-accelerated backend:

```bash
pip install cast-value[rs]
```

## Quick start

```python
import numpy as np
import zarr
from cast_value import CastValueNumpyV1

zarr.registry.register_codec("cast_value", CastValueNumpyV1)

codec = CastValueNumpyV1(
    data_type="uint8",
    rounding="nearest-even",
    out_of_range="clamp",
)

arr = zarr.create(
    shape=(100,),
    dtype="float64",
    chunks=(10,),
    store=zarr.storage.MemoryStore(),
    codecs=[codec, zarr.codecs.BytesCodec()],
    fill_value=0.0,
)

arr[:] = np.linspace(0, 300, 100)
print(arr[:10])  # [0. 3. 6. 9. 12. 15. 18. 21. 24. 27.]
```

## Backends

Two backends are available:

- **`CastValueNumpyV1`** — Pure Python + NumPy. Always available.
- **`CastValueRustV1`** — Rust via
  [cast-value-rs](https://pypi.org/project/cast-value-rs/). Faster for
  non-default rounding modes and SIMD-accelerated float-to-integer casts with
  clamping, with more efficient memory usage.

Both implement the same codec interface and produce identical results.
