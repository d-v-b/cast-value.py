# cast-value

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

[![Coverage][coverage-badge]][coverage-link]

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/zarr-developers/cast-value/actions/workflows/ci.yml/badge.svg
[actions-link]:             https://github.com/zarr-developers/cast-value/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/cast-value
[conda-link]:               https://github.com/conda-forge/cast-value-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/zarr-developers/cast-value/discussions
[pypi-link]:                https://pypi.org/project/cast-value/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/cast-value
[pypi-version]:             https://img.shields.io/pypi/v/cast-value
[rtd-badge]:                https://readthedocs.org/projects/cast-value/badge/?version=latest
[rtd-link]:                 https://cast-value.readthedocs.io/en/latest/?badge=latest
[coverage-badge]:           https://codecov.io/github/zarr-developers/cast-value/branch/main/graph/badge.svg
[coverage-link]:            https://codecov.io/github/zarr-developers/cast-value

<!-- prettier-ignore-end -->

# cast-value.py

A python implementation of the `cast_value` codec for [Zarr](https://zarr.dev/),
with [zarr-python](https://zarr.readthedocs.io/en/stable/) integration.

## what

The `cast_value` codec defines how to _safely_ convert arrays between integer
and float data types. In Zarr terminology, this codec is an "array -> array"
codec, which means its input and output are both arrays.

You can find the
[specification for this codec](https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/cast_value)
in the
[zarr-extensions repository](https://github.com/zarr-developers/zarr-extensions).

## why

This codec is commonly used to for lossy data compression: when decoded data
should be high-precision floats, but the absolute range of the values fits
within the range of a smaller integer data type, then encoding the floats as
ints before writing data can vastly shrink the stored values.

For example, if your data is a sequence of `float64` values like
`[100.1, 120.3, 125.5]`, storing those values as `uint8`, e.g.
`[100, 120, 125]`, offers 8-fold reduction in storage size, provided the
precision lost due to rounding is acceptable.

## how

```python
# Create an in-memory zarr array with float64 dtype, stored as uint8.
# The cast_value codec handles the conversion: float64 -> uint8 on write,
# uint8 -> float64 on read.
codec = CastValueRustV1(
    data_type="uint8",
    rounding="nearest-even",
    out_of_range="clamp",
    scalar_map={
        "encode": [(np.nan, 0), (np.inf, 1), (-np.inf, 2)],
        "decode": [(0, np.nan), (1, np.inf), (2, -np.inf)],
    },
)
# Create array and write float64 data — values are rounded and clamped to [0, 255]
data = np.array([np.nan, np.inf, -np.inf, 3.3, 4])
arr = zarr.create_array(data=data, store=zarr.storage.MemoryStore(), filters=codec)

# Read it back — comes back as float64, but with uint8 precision
result = arr[:]

print(f"Array dtype: {arr.dtype}")
print(f"Values written: {data}")
print(f"Values read:    {result}")

"""
Array dtype: float64
Values written: [ nan  inf -inf  3.3  4. ]
Values read:    [ nan  inf -inf   3.   4.]
"""
```

# who

Davis Bennett (@d-v-b)
