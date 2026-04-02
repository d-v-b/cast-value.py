"""Tests that the example scripts run without errors."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize(
    "script",
    [
        pytest.param(
            EXAMPLES_DIR / "zarr_integration" / "zarr_cast_value.py",
            id="zarr-integration",
        ),
        pytest.param(
            EXAMPLES_DIR / "benchmarks" / "bench_numpy_vs_rust.py",
            id="benchmark",
        ),
    ],
)
def test_example_runs(script: Path) -> None:
    """Test that an example script exits successfully."""
    result = subprocess.run(
        ["uv", "run", "--reinstall-package", "cast-value", "python", str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"Example {script.name} failed with:\n{result.stderr}"
    )
