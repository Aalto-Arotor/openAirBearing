import io
import zipfile

import plotly.graph_objects as go

from openairbearing.app.ui_export_utils import build_export_figures, build_export_zip
from openairbearing.bearings import CircularBearing
from openairbearing.solution_analytic import solve_bearing_analytic


def test_build_export_figures_contains_shape_results_and_legend():
    bearing = CircularBearing(nx=14, nh=6)
    result = solve_bearing_analytic(bearing)

    figures = build_export_figures(bearing, [result])

    names = [name for name, _ in figures]
    assert any(name.startswith("shape_") for name in names)
    assert any(name.startswith("results_") for name in names)
    assert "results_legend" in names
    assert all(isinstance(fig, go.Figure) for _, fig in figures)


def test_build_export_zip_writes_csv_and_metadata_only():
    bearing = CircularBearing(nx=10, nh=5)
    result = solve_bearing_analytic(bearing)
    zip_bytes = build_export_zip(
        app_state={
            "case": "circular",
            "solvers": ["analytic"],
            "kappa": 1e-14,
            "Qsc": 2.5,
            "bearing_kwargs": {"pa": 101325.0, "nx": 10},
        },
        note="test note",
        bearing=bearing,
        results=[result],
    )

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        names = set(zf.namelist())
        assert "inputs.csv" in names
        assert not any(name.endswith(".pdf") for name in names)
        assert any(name.endswith("_results.csv") for name in names)

        inputs_csv = zf.read("inputs.csv").decode("utf-8")
        assert "variable,value,unit" in inputs_csv
        assert "note,test note,-" in inputs_csv
        assert "case,circular,-" in inputs_csv
        assert "kappa" in inputs_csv
        assert "Qsc" in inputs_csv
        assert "pa,101325.0,Pa" in inputs_csv

        csv_name = next(name for name in names if name.endswith("_results.csv"))
        csv_text = zf.read(csv_name).decode("utf-8")
        assert (
            "h_index,h_um,w_N,k_N_per_um,qs_L_per_min,qa_L_per_min,qc_L_per_min,p_min_MPa,p_max_MPa"
            in csv_text
        )
