# About

This repo contains a Python implementation for the `cast_value` codec for Zarr
V3. The spec for that codec is
[here](https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/cast_value).
The relevant section of the Zarr V3 spec is
[here](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-encoding).
Be advised that the Zarr V3 spec describes an API but that API is NOT NORMATIVE
and should be ignored as needed.

## Design

- Use a functional, declarative style with simple data structures and functions
  that transform them.
- `TypedDict`s are preferred over classes.
- Functions are preferred over methods.
- Accurate type annotations for everything.
- Never use the `Any` type. Always look for a more accurate alternative, like
  `object`.

## Architecture

This repo has two main parts:

1. One or more implementations of the `cast_value` codec. This follows an
   internal API defined in this library.
2. One or more implementations of a `Codec` class compatible with the
   [Zarr-Python codec API](https://zarr.readthedocs.io/en/stable/api/zarr/abc/codec/#zarr.abc.codec.__all__).
   The Zarr-Python codec API is likely to change in the future so the
   Zarr-Python compatibility layer needs to be robustly versioned.

## Tests

- Every parameter of every function must have a test.
- Use `@pytest.mark.parametrize` over pytest classes.
- Every test must have a docstring that explains what it tests.

### Actions

### Run Python script

`uv run <script>`

### Setup test env

`uv sync --group test`

### Run tests

`uv run pytest tests`
