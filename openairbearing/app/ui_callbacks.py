"""Dash callback handlers for interactive bearing analysis UI.

Manages state synchronization, solver execution with caching, plotting updates,
and interactive features like slider-based pressure frame scrubbing and exports.
"""

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime

import dash
import numpy as np
from dash import dcc, html
from dash import exceptions as dash_exceptions
from dash.dependencies import Input, Output, State

from openairbearing.app.ui_config import (
    SOLVER_EXECUTION_ORDER,
    get_bearing,
    get_container_styles,
    get_default_error,
    get_default_solvers,
    get_error_options,
    get_solver_options,
)
from openairbearing.app.ui_export_utils import build_export_zip
from openairbearing.app.ui_plots import (
    empty_figure,
    get_pressure_2d_z_range,
    plot_bearing_shape,
    plot_key_results,
    plot_legend_only,
    plot_pressure_2d,
)
from openairbearing.app.ui_state import (
    FORM_FIELD_ORDER,
    bearing_to_form_values,
    form_to_bearing_kwargs,
)
from openairbearing.solution_analytic import solve_bearing_analytic
from openairbearing.solution_fem import (
    solve_bearing_fem_1d,
    solve_bearing_fem_2d,
    solve_bearing_fem_2d_nonlinear,
)
from openairbearing.utils import get_beta, get_kappa, get_Qsc

MAX_CACHED_BEARINGS = 5
_SOLVER_RESULTS_CACHE = OrderedDict()
_SOLVER_FUNCTIONS = {
    "analytic": "solve_bearing_analytic",
    "numeric1d": "solve_bearing_fem_1d",
    "numeric2d": "solve_bearing_fem_2d",
    "numeric2dfull": "solve_bearing_fem_2d_nonlinear",
}

# Keep explicit references so static analysis sees these imports as used.
_SOLVER_SYMBOLS = (
    solve_bearing_analytic,
    solve_bearing_fem_1d,
    solve_bearing_fem_2d,
    solve_bearing_fem_2d_nonlinear,
)


def _get_solver_function(solver_name):
    """Resolve solver function by name at call time."""
    function_name = _SOLVER_FUNCTIONS[solver_name]
    return globals()[function_name]


def _freeze(value):
    """Convert nested data structures to hashable tuples for cache keys."""
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        return ("ndarray", tuple(arr.shape), tuple(arr.ravel().tolist()))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return tuple((k, _freeze(v)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _bearing_cache_key(case, bearing_kwargs):
    """Build cache key for one bearing configuration.

    Args:
        case: Bearing case string (e.g., 'circular', 'rectangular').
        bearing_kwargs: Dictionary of bearing parameters.

    Returns:
        tuple: Hashable cache key combining case and frozen bearing_kwargs.
    """
    return case, _freeze(bearing_kwargs)


def _bearing_cache_key_id(case, bearing_kwargs):
    """Build a JSON-safe identifier for a bearing configuration."""
    # dcc.Store serializes JSON and turns tuples into lists; using a string
    # keeps this stable and avoids unhashable keys in Python caches.
    return repr(_bearing_cache_key(case, bearing_kwargs))


def _get_cached_solver_result(bearing_key, solver_name):
    """Get one cached solver result and mark bearing key as recently used.

    Args:
        bearing_key: Cache key from _bearing_cache_key().
        solver_name: Name of solver ('analytic', 'numeric1d', 'numeric2d', etc).

    Returns:
        Result: Cached solver result or None if not found.
    """
    cached_by_solver = _SOLVER_RESULTS_CACHE.get(bearing_key)
    if cached_by_solver is None:
        return None
    _SOLVER_RESULTS_CACHE.move_to_end(bearing_key)
    return cached_by_solver.get(solver_name)


def _set_cached_solver_result(bearing_key, solver_name, result):
    """Store one solver result and enforce LRU limit by bearing config.

    Args:
        bearing_key: Cache key from _bearing_cache_key().
        solver_name: Name of solver ('analytic', 'numeric1d', 'numeric2d', etc).
        result: Result object to cache.
    """
    cached_by_solver = _SOLVER_RESULTS_CACHE.get(bearing_key)
    if cached_by_solver is None:
        cached_by_solver = {}
        _SOLVER_RESULTS_CACHE[bearing_key] = cached_by_solver
    cached_by_solver[solver_name] = result
    _SOLVER_RESULTS_CACHE.move_to_end(bearing_key)

    while len(_SOLVER_RESULTS_CACHE) > MAX_CACHED_BEARINGS:
        _SOLVER_RESULTS_CACHE.popitem(last=False)


def _ordered_solvers(selected_solvers):
    """Return solvers in UI/caching execution order."""
    if selected_solvers is None:
        selected_list = []
    elif isinstance(selected_solvers, str):
        selected_list = [selected_solvers]
    elif isinstance(selected_solvers, (list, tuple, set)):
        selected_list = list(selected_solvers)
    else:
        # Fallback: best-effort conversion to a singleton.
        selected_list = [str(selected_solvers)]

    selected_set = {str(s) for s in selected_list}
    return [s for s in SOLVER_EXECUTION_ORDER if s in selected_set]


def _weighted_progress_percent(completed, total):
    """Return weighted progress in percent with increasing step sizes.

    The step weights are 1, 2, 4, ..., so each step has double the weight of
    the previous one. This strongly biases progress toward later solver steps.
    """
    total_i = int(total or 0)
    completed_i = int(completed or 0)
    if total_i <= 0:
        return 0.0

    completed_i = max(0, min(total_i, completed_i))
    weighted_done = float((1 << completed_i) - 1)
    weighted_total = float((1 << total_i) - 1)
    if weighted_total <= 0:
        return 0.0
    return 100.0 * (weighted_done / weighted_total)


def _build_solver_results(bearing, solvers, *, case, bearing_kwargs):
    """Run selected solvers and return result objects in consistent UI order.

    Solvers are executed in order and cached by bearing configuration. Cache
    hit returns stored result; cache miss computes and stores new result.

    Args:
        bearing: Bearing instance with geometry and properties.
        solvers: List of solver names to execute.
        case: Bearing case string for cache key.
        bearing_kwargs: Dictionary of bearing parameters for cache key.

    Returns:
        list: List of Result objects in order
            (analytic, numeric1d, numeric2d, numeric2dfull).
    """
    bearing_key = _bearing_cache_key(case, bearing_kwargs)
    results = []
    for solver_name in SOLVER_EXECUTION_ORDER:
        if solver_name not in solvers:
            continue

        cached = _get_cached_solver_result(bearing_key, solver_name)
        if cached is not None:
            results.append(cached)
            continue

        result = _get_solver_function(solver_name)(bearing)
        _set_cached_solver_result(bearing_key, solver_name, result)
        results.append(result)
    return results


def _resolve_flow_inputs(bearing, triggered_input, kappa, qsc):
    """Synchronize kappa/Qsc pair, computing missing value from the other.

    When one parameter is modified, recompute the other to maintain consistency.
    If triggered_input is neither 'kappa-input' nor 'Qsc-input', use provided
    values or fall back to bearing's current values.

    Args:
        bearing: Bearing instance being modified.
        triggered_input: ID of input control that triggered the callback.
        kappa: Current kappa value (m^2) or None.
        qsc: Current Qsc value (L/min) or None.

    Returns:
        tuple: (resolved_kappa, resolved_qsc) values.
    """
    if triggered_input == "kappa-input" and kappa is not None:
        bearing.kappa = kappa
        bearing.Qsc = get_Qsc(bearing)
        return kappa, bearing.Qsc

    if triggered_input == "Qsc-input" and qsc is not None:
        bearing.Qsc = qsc
        bearing.kappa = get_kappa(bearing)
        return bearing.kappa, qsc

    bearing.kappa = kappa if kappa is not None else get_kappa(bearing)
    bearing.Qsc = qsc if qsc is not None else get_Qsc(bearing)
    return bearing.kappa, bearing.Qsc


def _build_slider_state(bearing, results):
    """Build slider configuration for animated pressure frame selection.

    Constructs a slider with marks at selected film heights, initializing to
    the height with maximum stiffness. Returns empty config if no 2D results.

    Args:
        bearing: Bearing instance with film heights (ha).
        results: List of Result objects to search for 2D solution.

    Returns:
        tuple: (max_idx, initial_idx, marks_dict, disabled,
            row_style) for slider configuration.
    """
    res2d = next(
        (r for r in results if getattr(r, "name", "") == "numeric 2d nonlinear"),
        None,
    )
    if res2d is None:
        res2d = next((r for r in results if getattr(r, "p_2d", None) is not None), None)
    if res2d is None or len(res2d.p_2d) == 0:
        return 0, 0, {}, True, {"display": "none"}

    idx0 = int(np.argmax(res2d.k))

    ha_um = np.asarray(bearing.ha).ravel() * 1e6
    n_steps = len(ha_um)
    if n_steps == 0:
        return 0, 0, {}, True, {"display": "none"}

    max_ticks = 10
    if n_steps <= max_ticks:
        tick_idx = list(range(n_steps))
    else:
        tick_idx = np.linspace(0, n_steps - 1, num=max_ticks, dtype=int).tolist()
        tick_idx = list(dict.fromkeys(tick_idx))
    idx0 = int(np.clip(idx0, 0, n_steps - 1))

    w_vals = np.asarray(getattr(res2d, "w", [])).ravel()

    def _label_for_idx(i):
        ha_label = f"{ha_um[i]:.2f} um"
        if i < len(w_vals):
            return f"{ha_label}, {w_vals[i]:.0f} N"
        return ha_label

    marks = {
        i: {
            "label": _label_for_idx(i),
            "style": {
                "whiteSpace": "nowrap",
                "fontSize": "11px",
            },
        }
        for i in tick_idx
    }
    row_style = {"display": "flex", "alignItems": "center", "justifyContent": "center"}
    return n_steps - 1, idx0, marks, False, row_style


def _empty_update_response():
    """Return fallback callback payload with empty figures when update fails.

    Returns:
        tuple: Default outputs for update_bearing callback on error.
    """
    return (
        [],
        [],
        empty_figure(),
        empty_figure(),
        0,
        0,
        {},
        True,
        {"display": "none"},
        empty_figure(),
        dash.no_update,
        dash.no_update,
    )


def _build_plot_rows(figures, graph_height="400px", cell_width="calc(33% - 20px)"):
    """Create HTML rows with 3-column graph layout from figure list.

    Args:
        figures: List of Plotly figure objects.
        graph_height: CSS height string for each graph. Defaults to '400px'.
        cell_width: CSS width string for each cell. Defaults to 'calc(33% - 20px)'.

    Returns:
        list: List of html.Div elements, each containing up to 3 graphs.
    """
    rows = []
    for i in range(0, len(figures), 3):
        row_figures = figures[i : i + 3]
        row = html.Div(
            [
                html.Div(
                    dcc.Graph(
                        figure=figure,
                        config={"displayModeBar": False},
                        style={"height": graph_height},
                    ),
                    style={
                        "width": cell_width,
                        "margin": "10px",
                        "padding": "0px",
                    },
                )
                for figure in row_figures
            ],
            style={"display": "flex", "justifyContent": "space-between"},
        )
        rows.append(row)
    return rows


def _get_2d_results_pair(results):
    """Extract static and moving 2D result solutions from result list.

    Identifies 2D results by name or p_2d attribute. Returns first two
    distinct 2D results, with static 2D taking priority.

    Args:
        results: List of Result objects.

    Returns:
        tuple: (static_2d_result, moving_2d_result) or (None, None).
    """
    static_res = next(
        (r for r in results if getattr(r, "name", "") == "numeric 2d"), None
    )
    moving_res = next(
        (r for r in results if getattr(r, "name", "") == "numeric 2d nonlinear"),
        None,
    )

    if static_res is None:
        static_res = next(
            (r for r in results if getattr(r, "p_2d", None) is not None), None
        )
    if (
        moving_res is None
        and static_res is not None
        and getattr(static_res, "name", "") != "numeric 2d nonlinear"
    ):
        moving_res = next(
            (
                r
                for r in results
                if getattr(r, "p_2d", None) is not None and r is not static_res
            ),
            None,
        )
    return static_res, moving_res


def _apply_pressure_frame(fig_json, idx):
    """Apply slider frame index to pressure figure, swapping frame data.

    Args:
        fig_json: Plotly figure JSON dict with animated frames.
        idx: Frame index to display (clipped to available range).

    Returns:
        dict: Modified figure JSON with swapped frame data or dash.no_update.
    """
    if fig_json is None:
        return dash.no_update

    frames = fig_json.get("frames") or []
    if not frames:
        return dash.no_update

    idx = int(np.clip(idx, 0, len(frames) - 1))
    frame_data = (frames[idx] or {}).get("data") or []
    if not frame_data:
        return dash.no_update

    out = deepcopy(fig_json)
    out_data = out.get("data") or []
    if not out_data:
        out["data"] = deepcopy(frame_data)
    else:
        for t_i, tr in enumerate(frame_data):
            if t_i >= len(out_data):
                out_data.append({})
            for key in (
                "type",
                "x",
                "y",
                "z",
                "i",
                "j",
                "k",
                "intensity",
                "name",
                "colorscale",
                "showscale",
                "colorbar",
            ):
                if key in tr:
                    out_data[t_i][key] = deepcopy(tr[key])

    out.setdefault("layout", {})
    out["layout"]["uirevision"] = "pressure2d"
    out["layout"]["datarevision"] = idx
    return out


def _recompute_from_app_state(app_state):
    """Recompute bearing instance and all solver results from app state.

    Args:
        app_state: Dictionary with keys 'case', 'solvers', 'kappa', 'Qsc',
                   'triggered_input', and 'bearing_kwargs'.

    Returns:
        tuple: (bearing, results, static_2d, moving_2d, shared_z_range, kappa, qsc).
    """
    bearing_class = get_bearing(app_state["case"])
    bearing = bearing_class(**app_state["bearing_kwargs"])

    new_kappa, new_qsc = _resolve_flow_inputs(
        bearing,
        app_state.get("triggered_input"),
        app_state.get("kappa"),
        app_state.get("Qsc"),
    )

    bearing.beta = get_beta(bearing)
    results = _build_solver_results(
        bearing,
        app_state.get("solvers", []),
        case=app_state["case"],
        bearing_kwargs=app_state["bearing_kwargs"],
    )

    static_2d, moving_2d = _get_2d_results_pair(results)
    shared_z_range = get_pressure_2d_z_range(
        bearing, [r for r in [static_2d, moving_2d] if r is not None]
    )
    return bearing, results, static_2d, moving_2d, shared_z_range, new_kappa, new_qsc


def _recompute_from_solve_state(solve_state):
    """Recompute solver outputs from a solve-run state payload."""
    app_state = {
        "case": solve_state["case"],
        "solvers": solve_state.get("solvers", []),
        "kappa": solve_state.get("kappa"),
        "Qsc": solve_state.get("Qsc"),
        "triggered_input": solve_state.get("triggered_input"),
        "bearing_kwargs": solve_state.get("bearing_kwargs", {}),
    }
    return _recompute_from_app_state(app_state)


def _build_result_outputs(bearing, results, static_2d, moving_2d, shared_z_range):
    """Build the result-panel outputs from already computed solver results."""
    both_2d_visible = static_2d is not None and moving_2d is not None

    plot_figures = plot_key_results(bearing, results, legend=False)
    result_rows = _build_plot_rows(
        plot_figures, graph_height="400px", cell_width="calc(33% - 20px)"
    )

    pressure_2d_static_fig = (
        plot_pressure_2d(
            bearing,
            static_2d,
            slider=False,
            show_colorbar=not both_2d_visible,
            z_range_mpa=shared_z_range,
        )
        if static_2d is not None
        else empty_figure()
    )
    pressure_2d_moving_fig = (
        plot_pressure_2d(
            bearing,
            moving_2d,
            slider=False,
            show_colorbar=True,
            z_range_mpa=shared_z_range,
        )
        if moving_2d is not None
        else empty_figure()
    )

    s_max, s_val, s_marks, s_disabled, row_style = _build_slider_state(bearing, results)
    header_legend_fig = plot_legend_only(results)
    return (
        result_rows,
        pressure_2d_static_fig,
        pressure_2d_moving_fig,
        s_max,
        s_val,
        s_marks,
        s_disabled,
        row_style,
        header_legend_fig,
    )


def _empty_result_outputs():
    """Return empty result-panel outputs."""
    return (
        [],
        empty_figure(),
        empty_figure(),
        0,
        0,
        {},
        True,
        {"display": "none"},
        empty_figure(),
    )


def _result_outputs_no_update():
    """Return dash.no_update placeholders for result-panel outputs."""
    return tuple(dash.no_update for _ in range(9))


def _build_solve_run_payload(app_state, solvers):
    """Build a solve-run payload for one requested solve."""
    ordered = _ordered_solvers(solvers)
    if not ordered:
        return None

    run_id = int(datetime.now().timestamp() * 1000)
    return {
        "run_id": run_id,
        "status": "running",
        "case": app_state["case"],
        "solvers": ordered,
        "bearing_kwargs": app_state["bearing_kwargs"],
        "bearing_cache_key_id": _bearing_cache_key_id(
            app_state["case"], app_state["bearing_kwargs"]
        ),
        "triggered_input": app_state.get("triggered_input"),
        "kappa": app_state.get("kappa"),
        "Qsc": app_state.get("Qsc"),
        "total": len(ordered),
        "completed": 0,
        "current_solver": None,
        "error": None,
    }


def _build_progress_state(
    run_id,
    *,
    status,
    total,
    completed,
    current_solver=None,
    error=None,
):
    """Create the UI-facing solve progress payload."""
    return {
        "run_id": run_id,
        "status": status,
        "total": int(total or 0),
        "completed": int(completed or 0),
        "current_solver": current_solver,
        "error": error,
    }


def _render_progress_state(progress_state):
    """Render progress-bar style and status text from solve state."""
    if not progress_state:
        return (
            {"height": "10px", "width": "0%", "backgroundColor": "#4caf50"},
            "Idle (click solve! to compute results)",
        )

    status = progress_state.get("status")
    total = int(progress_state.get("total", 0) or 0)
    completed = int(progress_state.get("completed", 0) or 0)
    current_solver = progress_state.get("current_solver")

    if status == "completed":
        percent = 100.0
    elif total <= 0:
        percent = 0.0
    else:
        percent = _weighted_progress_percent(completed, total)

    percent = max(0.0, min(100.0, percent))
    width_pct = f"{percent:.0f}%"

    if status in {"queued", "running"}:
        text = f"Running ({completed}/{total})"
        if current_solver:
            text += f" - {current_solver}"
        color = "#4caf50"
    elif status == "completed":
        text = f"Done ({completed}/{total})"
        color = "#2e7d32"
    elif status == "cancelled":
        text = f"Cancelled ({completed}/{total})"
        color = "#ef6c00"
    elif status == "error":
        text = f"Error: {progress_state.get('error', 'solve failed')}"
        color = "#c62828"
    else:
        text = "Idle"
        color = "#4caf50"

    return (
        {"height": "10px", "width": width_pct, "backgroundColor": color},
        text,
    )


def _execute_solve_request(solve_request, *, set_progress=None):
    """Execute one solve request and return computed outputs and solve metadata."""
    app_state = {
        "case": solve_request["case"],
        "solvers": solve_request.get("solvers", []),
        "kappa": solve_request.get("kappa"),
        "Qsc": solve_request.get("Qsc"),
        "triggered_input": solve_request.get("triggered_input"),
        "bearing_kwargs": solve_request.get("bearing_kwargs", {}),
    }

    bearing_class = get_bearing(app_state["case"])
    bearing = bearing_class(**app_state["bearing_kwargs"])
    new_kappa, new_qsc = _resolve_flow_inputs(
        bearing,
        app_state.get("triggered_input"),
        app_state.get("kappa"),
        app_state.get("Qsc"),
    )
    bearing.beta = get_beta(bearing)

    ordered_solvers = solve_request.get("solvers", []) or []
    total = len(ordered_solvers)
    run_id = solve_request["run_id"]
    bearing_key = _bearing_cache_key(app_state["case"], app_state["bearing_kwargs"])
    results = []

    for index, solver_name in enumerate(ordered_solvers):
        if set_progress is not None:
            set_progress(
                _build_progress_state(
                    run_id,
                    status="running",
                    total=total,
                    completed=index,
                    current_solver=solver_name,
                )
            )

        cached = _get_cached_solver_result(bearing_key, solver_name)
        if cached is None:
            cached = _get_solver_function(solver_name)(bearing)
            _set_cached_solver_result(bearing_key, solver_name, cached)
        results.append(cached)

        if set_progress is not None:
            set_progress(
                _build_progress_state(
                    run_id,
                    status="running",
                    total=total,
                    completed=index + 1,
                    current_solver=solver_name,
                )
            )

    static_2d, moving_2d = _get_2d_results_pair(results)
    shared_z_range = get_pressure_2d_z_range(
        bearing, [r for r in [static_2d, moving_2d] if r is not None]
    )
    return (
        bearing,
        results,
        static_2d,
        moving_2d,
        shared_z_range,
        new_kappa,
        new_qsc,
    )


def _build_solve_success_response(solve_request, computed_outputs):
    """Build callback outputs for a successful solve."""
    (
        bearing,
        results,
        static_2d,
        moving_2d,
        shared_z_range,
        new_kappa,
        new_qsc,
    ) = computed_outputs
    solve_run = dict(solve_request)
    solve_run["status"] = "completed"
    solve_run["completed"] = len(solve_request.get("solvers", []))
    solve_run["current_solver"] = None
    solve_run["error"] = None
    return (
        *_build_result_outputs(bearing, results, static_2d, moving_2d, shared_z_range),
        solve_run,
        new_kappa,
        new_qsc,
    )


def _build_solve_error_response(solve_request, error):
    """Build callback outputs for a failed solve without clearing old results."""
    solve_run = dict(solve_request)
    solve_run["status"] = "error"
    solve_run["current_solver"] = None
    solve_run["error"] = str(error)
    return (*_result_outputs_no_update(), solve_run, dash.no_update, dash.no_update)


def register_callbacks(app):
    """Register all interactive callbacks for the Dash application.

    Callbacks include: state synchronization, bearing recomputation with caching,
    plot updates, export, container visibility, solver options update, input resets,
    and pressure frame animation control.

    Args:
        app: Dash application instance.
    """

    @app.callback(
        Output("app-state", "data"),
        [
            Input("case-select", "value"),
            Input("solver-select", "value"),
            Input("pa-input", "value"),
            Input("ps-input", "value"),
            Input("pc-input", "value"),
            Input("mu-input", "value"),
            Input("hp-input", "value"),
            Input("xa-input", "value"),
            Input("xc-input", "value"),
            Input("bore-diameter-input", "value"),
            Input("shaft-diameter-input", "value"),
            Input("nx-input", "value"),
            Input("ya-input", "value"),
            Input("ny-input", "value"),
            Input("ha-min-input", "value"),
            Input("ha-max-input", "value"),
            Input("nh-input", "value"),
            Input("divs-input", "value"),
            Input("kappa-input", "value"),
            Input("Qsc-input", "value"),
            Input("error-select", "value"),
            Input("error-input", "value"),
            Input("psi-input", "value"),
            Input("ux-input", "value"),
            Input("uy-input", "value"),
        ],
        prevent_initial_call=True,
    )
    def sync_app_state(
        case,
        solvers,
        pa_mpa,
        ps_mpa,
        pc_mpa,
        mu,
        hp_mm,
        xa_mm,
        xc_mm,
        bore_diameter_mm,
        shaft_diameter_mm,
        nx,
        ya_mm,
        ny,
        ha_min_um,
        ha_max_um,
        nh,
        divs,
        kappa,
        qsc,
        error_type,
        error_um,
        psi,
        ux,
        uy,
    ):
        """Normalize form inputs into a single callback state payload.

        Triggered by any form input change. Converts UI units to SI and
        packages bearing parameters for downstream solvers.
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        input_id = ctx.triggered[0]["prop_id"].split(".")[0]

        form_values = {
            "pa": pa_mpa,
            "ps": ps_mpa,
            "pc": pc_mpa,
            "mu": mu,
            "hp": hp_mm,
            "xa": xa_mm,
            "xc": xc_mm,
            "bore_diameter": bore_diameter_mm,
            "shaft_diameter": shaft_diameter_mm,
            "ya": ya_mm,
            "nx": nx,
            "ny": ny,
            "ha_min": ha_min_um,
            "ha_max": ha_max_um,
            "nh": nh,
            "divs": divs,
            "error_type": error_type,
            "error": error_um,
            "psi": psi,
            "ux": ux,
            "uy": uy,
        }

        return {
            "case": case,
            "solvers": solvers or [],
            "kappa": kappa,
            "Qsc": qsc,
            "triggered_input": input_id,
            "bearing_kwargs": form_to_bearing_kwargs(form_values, case=case),
        }

    @app.callback(
        [
            Output("shape-plots-container", "children"),
            Output("result-plots-container", "children"),
            Output("pressure-2d-plot-static", "figure"),
            Output("pressure-2d-plot-moving", "figure"),
            Output("pressure-2d-slider", "max"),
            Output("pressure-2d-slider", "value"),
            Output("pressure-2d-slider", "marks"),
            Output("pressure-2d-slider", "disabled"),
            Output("pressure-2d-row", "style"),
            Output("results-legend-only", "figure"),
            Output("kappa-input", "value", allow_duplicate=True),
            Output("Qsc-input", "value", allow_duplicate=True),
        ],
        Input("app-state", "data"),
        State("solve-run", "data"),
        prevent_initial_call=True,
    )
    def update_bearing(app_state, solve_run):
        """Update bearing geometry after input settles.

        Solver results are intentionally not recomputed on every input change.
        Results get computed via the explicit `solve!` flow.
        """
        if not app_state:
            raise dash.exceptions.PreventUpdate

        try:
            b = get_bearing(app_state["case"])(**app_state["bearing_kwargs"])
            new_kappa, new_qsc = _resolve_flow_inputs(
                b,
                app_state.get("triggered_input"),
                app_state.get("kappa"),
                app_state.get("Qsc"),
            )
            b.beta = get_beta(b)

            shape_figures = plot_bearing_shape(b)
            shape_rows = _build_plot_rows(
                shape_figures, graph_height="400px", cell_width="calc(33% - 20px)"
            )

            current_ordered_solvers = _ordered_solvers(app_state.get("solvers", []))
            keep_results = (
                solve_run
                and solve_run.get("status") in {"queued", "running", "completed"}
                and solve_run.get("bearing_cache_key_id")
                == _bearing_cache_key_id(app_state["case"], app_state["bearing_kwargs"])
                and solve_run.get("solvers") == current_ordered_solvers
            )

            if keep_results:
                return (
                    shape_rows,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    new_kappa,
                    new_qsc,
                )

            return (
                shape_rows,
                [],
                empty_figure(),
                empty_figure(),
                0,
                0,
                {},
                True,
                {"display": "none"},
                empty_figure(),
                new_kappa,
                new_qsc,
            )
        except Exception as e:
            print(f"Error: {e}")
            return _empty_update_response()

    @app.callback(
        [Output("solve-request", "data"), Output("solve-run", "data")],
        [
            Input("solve-results-btn", "n_clicks"),
            Input("app-state", "data"),
        ],
        [
            State("auto-solve-finish-typing", "value"),
            State("solve-run", "data"),
            State("solver-select", "value"),
        ],
        prevent_initial_call=True,
    )
    def initialize_solve_run(
        solve_clicks, app_state, auto_solve_value, solve_run, solvers
    ):
        """Initialize or reset solve progress based on user intent."""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        auto_enabled = "on" in (auto_solve_value or [])
        solve_run_running = bool(
            solve_run and solve_run.get("status") in {"queued", "running"}
        )

        if triggered_id == "app-state":
            # During a running solve job, ignore geometry changes.
            if solve_run_running:
                return dash.no_update, dash.no_update

            if not app_state:
                return dash.no_update, dash.no_update

            # Only auto-start when the user finishes typing in a numeric input.
            eligible_finish_typing = (app_state.get("triggered_input") or "").endswith(
                "-input"
            )
            if auto_enabled and eligible_finish_typing:
                solve_request = _build_solve_run_payload(app_state, solvers)
                if solve_request is None:
                    return dash.no_update, dash.no_update
                return solve_request, solve_request

            # Auto-solve disabled or not a finish-typing trigger.
            return dash.no_update, dash.no_update

        if triggered_id == "solve-results-btn":
            if not app_state:
                raise dash.exceptions.PreventUpdate
            solve_request = _build_solve_run_payload(app_state, solvers)
            if solve_request is None:
                return None, None
            return solve_request, solve_request

        raise dash.exceptions.PreventUpdate

    @app.callback(
        Output("solve-run", "data", allow_duplicate=True),
        Input("cancel-solve-btn", "n_clicks"),
        State("solve-run", "data"),
        prevent_initial_call=True,
    )
    def cancel_running_solve(_cancel_clicks, solve_run):
        """Mark a running solve as cancelled in the UI state."""
        if not solve_run or solve_run.get("status") not in {"queued", "running"}:
            raise dash.exceptions.PreventUpdate
        solve_run = dict(solve_run)
        solve_run["status"] = "cancelled"
        solve_run["current_solver"] = None
        return solve_run

    @app.callback(
        [
            Output("solve-progress-bar-fill", "style"),
            Output("solve-progress-text", "children"),
        ],
        [Input("solve-progress", "data"), Input("solve-run", "data")],
        prevent_initial_call=True,
    )
    def render_solve_progress(solve_progress, solve_run):
        """Render solve progress UI from live progress and final solve state."""
        active_progress = solve_progress
        if active_progress and active_progress.get("status") in {"queued", "running"}:
            return _render_progress_state(active_progress)
        return _render_progress_state(solve_run)

    @app.callback(
        Output("cancel-solve-btn", "disabled"),
        Input("solve-run", "data"),
        prevent_initial_call=False,
    )
    def toggle_cancel_button(solve_run):
        """Enable cancel only while a solve request is actively running."""
        if not solve_run:
            return True
        return solve_run.get("status") not in {"queued", "running"}

    solve_outputs = [
        Output("result-plots-container", "children", allow_duplicate=True),
        Output("pressure-2d-plot-static", "figure", allow_duplicate=True),
        Output("pressure-2d-plot-moving", "figure", allow_duplicate=True),
        Output("pressure-2d-slider", "max", allow_duplicate=True),
        Output("pressure-2d-slider", "value", allow_duplicate=True),
        Output("pressure-2d-slider", "marks", allow_duplicate=True),
        Output("pressure-2d-slider", "disabled", allow_duplicate=True),
        Output("pressure-2d-row", "style", allow_duplicate=True),
        Output("results-legend-only", "figure", allow_duplicate=True),
        Output("solve-run", "data", allow_duplicate=True),
        Output("kappa-input", "value", allow_duplicate=True),
        Output("Qsc-input", "value", allow_duplicate=True),
    ]

    if getattr(app, "_background_manager", None) is not None:

        @app.callback(
            solve_outputs,
            Input("solve-request", "data"),
            background=True,
            running=[
                (Output("solve-results-btn", "disabled"), True, False),
            ],
            cancel=[Input("cancel-solve-btn", "n_clicks")],
            progress=Output("solve-progress", "data"),
            progress_default=None,
            prevent_initial_call=True,
        )
        def run_solve_in_background(set_progress, solve_request):
            """Execute the selected solvers in Dash's background callback queue."""
            if not solve_request:
                raise dash.exceptions.PreventUpdate

            try:
                computed_outputs = _execute_solve_request(
                    solve_request, set_progress=set_progress
                )
                return _build_solve_success_response(solve_request, computed_outputs)
            except Exception as exc:
                return _build_solve_error_response(solve_request, exc)

    else:

        @app.callback(
            solve_outputs,
            Input("solve-request", "data"),
            running=[
                (Output("solve-results-btn", "disabled"), True, False),
            ],
            prevent_initial_call=True,
        )
        def run_solve_foreground(solve_request):
            """Fallback solve path when no background manager is configured."""
            if not solve_request:
                raise dash.exceptions.PreventUpdate

            try:
                computed_outputs = _execute_solve_request(solve_request)
                return _build_solve_success_response(solve_request, computed_outputs)
            except Exception as exc:
                return _build_solve_error_response(solve_request, exc)

    @app.callback(
        [Output("export-download", "data"), Output("export-status", "children")],
        Input("export-results-btn", "n_clicks"),
        [State("app-state", "data"), State("export-note-input", "value")],
        prevent_initial_call=True,
    )
    def export_results(n_clicks, app_state, note):
        """Export bearing analysis results as CSV data in zip archive.

        Generates CSV files for numerical results and metadata.
        """
        if not n_clicks or not app_state:
            raise dash.exceptions.PreventUpdate

        try:
            bearing, results, *_ = _recompute_from_app_state(app_state)
            zip_bytes = build_export_zip(
                app_state=app_state,
                note=note,
                bearing=bearing,
                results=results,
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"openairbearing_export_{timestamp}.zip"
            status = "Export ready."
            return dcc.send_bytes(zip_bytes, filename), status
        except Exception as exc:
            return dash.no_update, f"Export failed: {exc}"

    @app.callback(
        Output("pc-container", "style"),
        Output("xc-container", "style"),
        Output("ya-container", "style"),
        Output("ny-container", "style"),
        Output("divs-container", "style"),
        Output("journal-container", "style"),
        Input("case-select", "value"),
    )
    def toggle_containers(case):
        """Show/hide parameter input containers based on bearing case.

        Toggles visibility of case-specific parameters (chamber pressure,
        center radius, span dimension, angular resolution, journal diameter inputs).
        """
        return get_container_styles(case)

    @app.callback(
        Output("clearance-input", "value"),
        Input("bore-diameter-input", "value"),
        Input("shaft-diameter-input", "value"),
    )
    def sync_journal_clearance(bore_diameter_mm, shaft_diameter_mm):
        """Keep journal clearance display in sync with the diameters."""
        if bore_diameter_mm is None or shaft_diameter_mm is None:
            return None
        return 0.5 * (bore_diameter_mm - shaft_diameter_mm) * 1000.0

    @app.callback(
        [Output("solver-select", "options"), Output("solver-select", "value")],
        Input("case-select", "value"),
    )
    def update_solver_options(case):
        """Update available solvers and defaults based on the selected case.

        Some solvers (e.g., 2D FEM) only work for circular/annular geometries.
        """
        return get_solver_options(case), get_default_solvers(case)

    @app.callback(
        [Output("error-select", "options"), Output("error-select", "value")],
        Input("case-select", "value"),
    )
    def update_error_options(case):
        """Update geometry error choices to match the selected bearing case."""
        return get_error_options(case), get_default_error(case)

    @app.callback(
        [
            Output("mu-input", "value"),
            Output("hp-input", "value"),
            Output("xa-input", "value"),
            Output("xc-input", "value"),
            Output("ya-input", "value"),
            Output("bore-diameter-input", "value"),
            Output("shaft-diameter-input", "value"),
            Output("clearance-input", "value", allow_duplicate=True),
            Output("kappa-input", "value", allow_duplicate=True),
            Output("Qsc-input", "value", allow_duplicate=True),
            Output("pa-input", "value"),
            Output("pc-input", "value"),
            Output("ps-input", "value"),
            Output("ha-min-input", "value"),
            Output("ha-max-input", "value"),
            Output("nx-input", "value"),
            Output("ny-input", "value"),
            Output("nh-input", "value"),
            Output("divs-input", "value"),
            Output("error-input", "value"),
            Output("psi-input", "value"),
            Output("ux-input", "value"),
            Output("uy-input", "value"),
        ],
        [
            Input("reset-all", "n_clicks"),
            Input("mu-reset", "n_clicks"),
            Input("hp-reset", "n_clicks"),
            Input("xa-reset", "n_clicks"),
            Input("xc-reset", "n_clicks"),
            Input("bore-diameter-reset", "n_clicks"),
            Input("shaft-diameter-reset", "n_clicks"),
            Input("ya-reset", "n_clicks"),
            Input("kappa-reset", "n_clicks"),
            Input("Qsc-reset", "n_clicks"),
            Input("pa-reset", "n_clicks"),
            Input("pc-reset", "n_clicks"),
            Input("ps-reset", "n_clicks"),
            Input("ha-min-reset", "n_clicks"),
            Input("ha-max-reset", "n_clicks"),
            Input("nx-reset", "n_clicks"),
            Input("ny-reset", "n_clicks"),
            Input("nh-reset", "n_clicks"),
            Input("divs-reset", "n_clicks"),
            Input("error-reset", "n_clicks"),
            Input("psi-reset", "n_clicks"),
            Input("ux-reset", "n_clicks"),
            Input("uy-reset", "n_clicks"),
            Input("case-select", "value"),
        ],
        prevent_initial_call=True,
    )
    def reset_values(reset_all, *args):
        """Reset one or all input controls to case defaults.

        If 'reset-all' or case changed, reset all parameters. Otherwise,
        reset only the parameter corresponding to the clicked reset button.
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        case = args[-1]
        current_values = bearing_to_form_values(get_bearing(case)())

        if button_id in ["reset-all", "case-select"]:
            return [current_values[field] for field in FORM_FIELD_ORDER]

        param = button_id.replace("-reset", "")
        _alias = {
            "ha-min": "ha_min",
            "ha-max": "ha_max",
            "bore-diameter": "bore_diameter",
            "shaft-diameter": "shaft_diameter",
        }
        param = _alias.get(param, param)
        return [
            current_values[p] if p == param else dash.no_update
            for p in FORM_FIELD_ORDER
        ]

    @app.callback(
        [
            Output("pressure-2d-plot-static", "figure", allow_duplicate=True),
            Output("pressure-2d-plot-moving", "figure", allow_duplicate=True),
        ],
        Input("pressure-2d-slider", "value"),
        State("pressure-2d-plot-static", "figure"),
        State("pressure-2d-plot-moving", "figure"),
        prevent_initial_call=True,
    )
    def scrub_pressure_2d(idx, static_fig_json, moving_fig_json):
        """Swap visible 3D pressure frames on both static and moving figures.

        Applies slider frame index to both 2D pressure plots simultaneously.
        """
        if idx is None:
            raise dash_exceptions.PreventUpdate

        out_static = _apply_pressure_frame(static_fig_json, idx)
        out_moving = _apply_pressure_frame(moving_fig_json, idx)
        if out_static is dash.no_update and out_moving is dash.no_update:
            raise dash.exceptions.PreventUpdate
        return out_static, out_moving
