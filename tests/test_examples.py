from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load_module(example_path: Path):
    spec = spec_from_file_location(example_path.stem, example_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _entrypoint(module):
    if hasattr(module, "main"):
        return module.main
    for name in (
        "circular_baseline_example",
        "circular_geometry_error_compare",
        "journal_geometry_error_compare",
        "rectangular_velocity_compare",
        "custom_geometry_example",
        "experimental_comparison_example",
        "tilt_sweep_example",
    ):
        if hasattr(module, name):
            return getattr(module, name)
    raise AssertionError(f"No runnable entrypoint found in module {module.__name__}")


@pytest.mark.parametrize(
    "example_name",
    [
        "ex00_run_app_local.py",
        "ex01_circular_baseline.py",
        "ex02_circular_geometry_error_comparison.py",
        "ex03_rectangular_velocity_comparison.py",
        "ex04_custom_geometry_function.py",
        "ex05_experimental_comparison.py",
        "ex06_circular_tilt_sweep.py",
        "ex07_journal_geometry_error_comparison.py",
    ],
)
def test_examples_run(example_name, monkeypatch):
    example_path = EXAMPLES_DIR / example_name
    module = _load_module(example_path)
    run = _entrypoint(module)

    monkeypatch.setattr(plt, "show", lambda: None)

    # ex00 starts the Dash app; replace it with a no-op for tests.
    if example_name == "ex00_run_app_local.py":
        monkeypatch.setattr(module, "main", lambda: None)
        run = module.main

    run()
    plt.close("all")
