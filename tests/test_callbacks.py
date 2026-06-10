import importlib

import numpy as np
import pytest

from openairbearing.app.ui_callbacks import (
    _build_solve_run_payload,
    _execute_solve_request,
    _recompute_from_app_state,
    _render_progress_state,
    _weighted_progress_percent,
)
from openairbearing.app.ui_config import get_bearing
from openairbearing.app.ui_state import bearing_to_form_values, form_to_bearing_kwargs
from openairbearing.bearings import (
    AnnularBearing,
    CircularBearing,
    InfiniteLinearBearing,
    RectangularBearing,
)


def test_get_bearing():
    assert get_bearing("circular") == CircularBearing
    assert get_bearing("annular") == AnnularBearing
    assert get_bearing("infinite") == InfiniteLinearBearing
    assert get_bearing("rectangular") == RectangularBearing
    with pytest.raises(TypeError, match="no default bearing defined"):
        get_bearing("unknown")


def test_recompute_from_app_state_returns_consistent_outputs():
    bearing = CircularBearing(nx=16, nh=6)
    form_values = bearing_to_form_values(bearing)
    app_state = {
        "case": "circular",
        "solvers": ["analytic"],
        "kappa": bearing.kappa,
        "Qsc": bearing.Qsc,
        "triggered_input": "case-select",
        "bearing_kwargs": form_to_bearing_kwargs({**form_values, "error_type": "none"}),
    }

    out_bearing, results, static_2d, moving_2d, shared_z_range, new_kappa, new_qsc = (
        _recompute_from_app_state(app_state)
    )

    assert out_bearing.case == "circular"
    assert len(results) == 1
    assert results[0].name == "analytic"
    assert static_2d is None
    assert moving_2d is None
    assert len(shared_z_range) == 2
    assert new_kappa == pytest.approx(out_bearing.kappa)
    assert new_qsc == pytest.approx(out_bearing.Qsc)


def test_build_solver_results_reuses_cached_solvers(monkeypatch):
    callbacks = importlib.import_module("openairbearing.app.ui_callbacks")

    callbacks._SOLVER_RESULTS_CACHE.clear()

    calls = {"analytic": 0, "numeric2d": 0}

    def fake_analytic(_bearing):
        calls["analytic"] += 1
        return {"solver": "analytic", "call": calls["analytic"]}

    def fake_numeric2d(_bearing):
        calls["numeric2d"] += 1
        return {"solver": "numeric2d", "call": calls["numeric2d"]}

    monkeypatch.setattr(callbacks, "solve_bearing_analytic", fake_analytic)
    monkeypatch.setattr(callbacks, "solve_bearing_fem_2d", fake_numeric2d)

    bearing_kwargs_a = {
        "pa": 1.0,
        "ps": 2.0,
        "u": np.array([0.0, 0.0]),
    }
    bearing_kwargs_b = {
        "pa": 1.1,
        "ps": 2.0,
        "u": np.array([0.0, 0.0]),
    }

    first = callbacks._build_solver_results(
        object(),
        ["analytic", "numeric2d"],
        case="circular",
        bearing_kwargs=bearing_kwargs_a,
    )
    second = callbacks._build_solver_results(
        object(),
        ["analytic"],
        case="circular",
        bearing_kwargs=bearing_kwargs_a,
    )
    third = callbacks._build_solver_results(
        object(),
        ["analytic"],
        case="circular",
        bearing_kwargs=bearing_kwargs_b,
    )

    assert calls["analytic"] == 2
    assert calls["numeric2d"] == 1
    assert second[0] is first[0]
    assert third[0] is not first[0]


def test_solver_cache_keeps_recent_bearings(monkeypatch):
    callbacks = importlib.import_module("openairbearing.app.ui_callbacks")

    callbacks._SOLVER_RESULTS_CACHE.clear()

    calls = {"analytic": 0}

    def fake_analytic(_bearing):
        calls["analytic"] += 1
        return calls["analytic"]

    monkeypatch.setattr(callbacks, "solve_bearing_analytic", fake_analytic)

    for idx in range(callbacks.MAX_CACHED_BEARINGS + 2):
        callbacks._build_solver_results(
            object(),
            ["analytic"],
            case="circular",
            bearing_kwargs={"pa": float(idx), "u": np.array([0.0, 0.0])},
        )

    assert len(callbacks._SOLVER_RESULTS_CACHE) == callbacks.MAX_CACHED_BEARINGS


def test_weighted_progress_percent_is_monotonic_and_front_light():
    values = [_weighted_progress_percent(i, 4) for i in range(5)]

    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(100.0)
    assert values == sorted(values)

    # Increment sizes should double: 1x, 2x, 4x, 8x (normalized to 100%).
    increments = [values[i + 1] - values[i] for i in range(4)]
    assert increments == pytest.approx(
        [100.0 / 15.0, 200.0 / 15.0, 400.0 / 15.0, 800.0 / 15.0]
    )


def test_build_solve_run_payload_orders_requested_solvers():
    app_state = {
        "case": "circular",
        "bearing_kwargs": {"pa": 1.0},
        "triggered_input": "solve-results-btn",
        "kappa": 1.23,
        "Qsc": 4.56,
    }

    payload = _build_solve_run_payload(
        app_state,
        ["numeric2d", "analytic", "numeric1d"],
    )

    assert payload is not None
    assert payload["status"] == "running"
    assert payload["solvers"] == ["analytic", "numeric1d", "numeric2d"]
    assert payload["completed"] == 0


def test_execute_solve_request_reports_progress_and_completes(monkeypatch):
    callbacks = importlib.import_module("openairbearing.app.ui_callbacks")
    callbacks._SOLVER_RESULTS_CACHE.clear()

    progress_updates = []
    calls = {"analytic": 0, "numeric1d": 0}

    def fake_analytic(_bearing):
        calls["analytic"] += 1
        return {"name": "analytic", "call": calls["analytic"]}

    def fake_numeric1d(_bearing):
        calls["numeric1d"] += 1
        return {"name": "numeric 1d", "call": calls["numeric1d"]}

    monkeypatch.setattr(callbacks, "solve_bearing_analytic", fake_analytic)
    monkeypatch.setattr(callbacks, "solve_bearing_fem_1d", fake_numeric1d)

    solve_request = {
        "run_id": 99,
        "case": "circular",
        "solvers": ["analytic", "numeric1d"],
        "bearing_kwargs": {"pa": 101325.0, "ps": 701325.0, "u": np.array([0.0, 0.0])},
        "triggered_input": "solve-results-btn",
        "kappa": 1e-12,
        "Qsc": 1.0,
        "total": 2,
        "completed": 0,
        "current_solver": None,
    }

    out = _execute_solve_request(solve_request, set_progress=progress_updates.append)

    assert calls == {"analytic": 1, "numeric1d": 1}
    assert len(progress_updates) == 4
    assert progress_updates[0]["completed"] == 0
    assert progress_updates[1]["completed"] == 1
    assert progress_updates[2]["completed"] == 1
    assert progress_updates[3]["completed"] == 2
    assert progress_updates[-1]["current_solver"] == "numeric1d"

    bearing, results, *_ = out
    assert bearing.case == "circular"
    assert len(results) == 2


def test_render_progress_state_handles_cancelled_status():
    style, text = _render_progress_state(
        {"status": "cancelled", "completed": 1, "total": 3, "current_solver": None}
    )

    assert style["backgroundColor"] == "#ef6c00"
    assert text == "Cancelled (1/3)"


# psutil multiprocess selenium "dash[testing]"
# def test_update_bearing(dash_duo):
#     app = import_app("openairbearing.app.app")
#     dash_duo.start_server(app)

#     # Simulate user selecting a bearing case
#     dash_duo.select_dcc_dropdown("#case-select", "circular")

#     # Simulate user input for parameters
#     dash_duo.find_element("#pa-input").send_keys("0.1")  # Ambient pressure in MPa
#     dash_duo.find_element("#ps-input").send_keys("0.6")  # Supply pressure in MPa

#     # Wait for the callback to update the outputs
#     dash_duo.wait_for_text_to_equal("#kappa-input", "calculated_value", timeout=5)
#     dash_duo.wait_for_text_to_equal("#Qsc-input", "calculated_value", timeout=5)

#     # Verify the output figures
#     assert dash_duo.find_element("#bearing-plots").is_displayed()
#     assert dash_duo.find_element("#bearing-shape").is_displayed()
