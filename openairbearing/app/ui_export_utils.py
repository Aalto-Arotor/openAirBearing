"""Utilities for exporting bearing analysis results as CSV zip archives.

Provides functions to generate CSV summaries and pressure distributions, and
package everything into zip archives.
"""

import csv
import io
import re
import zipfile
from datetime import UTC, datetime

import numpy as np

from openairbearing.app.ui_plots import (
    get_pressure_2d_z_range,
    plot_bearing_shape,
    plot_key_results,
    plot_legend_only,
    plot_pressure_2d,
)


def _slugify(value):
    """Convert string to URL-safe slug (lowercase, underscores, no spaces).

    Args:
        value: Any value to convert to slug.

    Returns:
        str: Slugified string or 'figure' if empty.
    """
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_")
    return value.lower() or "figure"


def _is_empty_figure(fig):
    """Check if Plotly figure contains no data traces.

    Args:
        fig: Plotly figure object.

    Returns:
        bool: True if figure has no traces.
    """
    return len(getattr(fig, "data", []) or []) == 0


def _export_title(fig, fallback):
    """Extract title from figure layout, falling back to default string.

    Args:
        fig: Plotly figure object.
        fallback: Default title if figure title is missing or empty.

    Returns:
        str: Figure title or fallback.
    """
    title = ((fig.layout or {}).title.text if fig.layout else None) or fallback
    return str(title).strip() or fallback


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


def _as_1d(value):
    """Convert value to 1D numpy array or None if empty.

    Args:
        value: Scalar, array, list, or None.

    Returns:
        np.ndarray or None: Flattened 1D array or None.
    """
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return np.ravel(arr)


def _solver_slug(result):
    """Extract solver name slug from result object.

    Args:
        result: Result object with optional 'name' attribute.

    Returns:
        str: Slugified solver name or 'solver'.
    """
    return _slugify(getattr(result, "name", "solver"))


INPUT_UNITS = {
    "case": "-",
    "solvers": "-",
    "triggered_input": "-",
    "kappa": "m^2",
    "Qsc": "L/min",
    "pa": "Pa",
    "ps": "Pa",
    "pc": "Pa",
    "mu": "Pa·s",
    "hp": "m",
    "xa": "m",
    "xc": "m",
    "ya": "m",
    "bore_diameter": "m",
    "shaft_diameter": "m",
    "clearance": "m",
    "nx": "-",
    "ny": "-",
    "nh": "-",
    "ha_min": "m",
    "ha_max": "m",
    "error_type": "-",
    "error": "m",
    "Psi": "-",
    "u_x": "m/s",
    "u_y": "m/s",
    "created_utc": "ISO-8601",
    "note": "-",
}


def _stringify_value(value):
    """Convert value to CSV-safe string representation.

    Args:
        value: Any value (scalar, array, list, dict, None).

    Returns:
        str: String representation suitable for CSV.
    """
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        arr = np.ravel(value)
        return ";".join(str(v) for v in arr)
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(v) for v in value)
    return str(value)


def _metadata_csv_bytes(*, app_state, note):
    """Generate CSV bytes with analysis metadata and input parameters.

    Args:
        app_state: Dictionary with 'case', 'solvers', bearing parameters, etc.
        note: Optional user note string.

    Returns:
        bytes: UTF-8 encoded CSV content.
    """
    rows = [("variable", "value", "unit")]
    state = app_state or {}

    rows.append(
        (
            "created_utc",
            datetime.now(UTC).isoformat(),
            INPUT_UNITS["created_utc"],
        )
    )
    rows.append(("note", _stringify_value(note or ""), INPUT_UNITS["note"]))

    top_level_keys = ("case", "solvers", "kappa", "Qsc")
    for key in top_level_keys:
        if key not in state:
            continue
        rows.append((key, _stringify_value(state[key]), INPUT_UNITS.get(key, "-")))

    bearing_kwargs = state.get("bearing_kwargs") or {}
    for key, value in bearing_kwargs.items():
        if key == "u":
            u = np.ravel(np.asarray(value, dtype=float))
            if u.size > 0:
                rows.append(("u_x", _stringify_value(u[0]), INPUT_UNITS["u_x"]))
            if u.size > 1:
                rows.append(("u_y", _stringify_value(u[1]), INPUT_UNITS["u_y"]))
            continue
        rows.append((key, _stringify_value(value), INPUT_UNITS.get(key, "-")))

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _result_summary_csv_bytes(bearing, result):
    """Generate CSV bytes with per-height-index results summary.

    Includes film height, load, stiffness, flows, moments, and shear forces
    across all height indices.

    Args:
        bearing: Bearing instance with geometry (ha, x, etc).
        result: Result object with computed fields.

    Returns:
        bytes: UTF-8 encoded CSV content.
    """
    ha = np.ravel(np.asarray(bearing.ha))
    nh = len(ha)

    columns = {
        "h_index": np.arange(nh, dtype=int),
        "h_um": ha * 1e6,
    }

    scalar_fields = (
        ("w_N", getattr(result, "w", None)),
        ("k_N_per_um", getattr(result, "k", None)),
        ("qs_L_per_min", getattr(result, "qs", None)),
        ("qa_L_per_min", getattr(result, "qa", None)),
        ("qc_L_per_min", getattr(result, "qc", None)),
    )
    for key, value in scalar_fields:
        arr = _as_1d(value)
        if arr is not None and len(arr) == nh:
            columns[key] = arr

    moment = np.asarray(getattr(result, "moment", []))
    if moment.ndim == 2 and len(moment) == nh and moment.shape[1] >= 2:
        columns["moment_x_Nm"] = moment[:, 0]
        columns["moment_y_Nm"] = moment[:, 1]
        columns["moment_mag_Nm"] = np.sqrt(moment[:, 0] ** 2 + moment[:, 1] ** 2)

    shear_force = np.asarray(getattr(result, "shear_force", []))
    if shear_force.ndim == 2 and len(shear_force) == nh and shear_force.shape[1] >= 2:
        columns["shear_x_N"] = shear_force[:, 0]
        columns["shear_y_N"] = shear_force[:, 1]
        columns["shear_mag_N"] = np.sqrt(
            shear_force[:, 0] ** 2 + shear_force[:, 1] ** 2
        )

    pressure_1d = np.asarray(getattr(result, "p", []))
    if pressure_1d.ndim == 2 and pressure_1d.shape[1] == nh:
        p_mpa = (pressure_1d - bearing.pa) * 1e-6
        columns["p_min_MPa"] = np.nanmin(p_mpa, axis=0)
        columns["p_max_MPa"] = np.nanmax(p_mpa, axis=0)

    header = list(columns.keys())
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    for row_idx in range(nh):
        writer.writerow([columns[key][row_idx] for key in header])
    return buffer.getvalue().encode("utf-8")


def _pressure_1d_csv_bytes(bearing, result):
    """Generate CSV bytes with 1D pressure distribution (pressure vs x, h).

    Args:
        bearing: Bearing instance with x grid.
        result: Result object with 1D pressure field.

    Returns:
        bytes or None: UTF-8 encoded CSV or None if incompatible.
    """
    pressure_1d = np.asarray(getattr(result, "p", []))
    if pressure_1d.ndim != 2:
        return None

    x = np.ravel(np.asarray(getattr(bearing, "x", [])))
    ha = np.ravel(np.asarray(getattr(bearing, "ha", [])))
    if pressure_1d.shape[0] != len(x) or pressure_1d.shape[1] != len(ha):
        return None

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["h_index", "h_um", "x_index", "x_mm", "p_MPa"])
    for h_idx, h in enumerate(ha):
        for x_idx, x_val in enumerate(x):
            p_mpa = (pressure_1d[x_idx, h_idx] - bearing.pa) * 1e-6
            writer.writerow([h_idx, h * 1e6, x_idx, x_val * 1e3, p_mpa])
    return buffer.getvalue().encode("utf-8")


def _pressure_2d_csv_bytes(bearing, result):
    """Generate CSV bytes with 2D pressure field at all nodes and heights.

    Args:
        bearing: Bearing instance with basis and doflocs.
        result: Result object with 2D pressure fields.

    Returns:
        bytes or None: UTF-8 encoded CSV or None if incompatible.
    """
    p_2d = getattr(result, "p_2d", None)
    basis = getattr(getattr(bearing, "fem_2d", None), "basis", None)
    if p_2d is None or basis is None:
        return None

    doflocs = np.asarray(getattr(basis, "doflocs", []))
    if doflocs.ndim != 2 or doflocs.shape[0] < 2:
        return None

    x_mm = np.ravel(doflocs[0]) * 1e3
    y_mm = np.ravel(doflocs[1]) * 1e3
    ha = np.ravel(np.asarray(getattr(bearing, "ha", [])))

    buffer = io.StringIO()
    buffer.write("h_index,h_um,node_index,x_mm,y_mm,p_MPa\n")
    node_index = np.arange(len(x_mm), dtype=int)

    for h_idx, p_field in enumerate(p_2d):
        p_vec = np.ravel(np.asarray(p_field))
        if len(p_vec) != len(x_mm):
            continue

        h_um = ha[h_idx] * 1e6 if h_idx < len(ha) else np.nan
        rows = np.column_stack(
            [
                np.full(len(x_mm), h_idx, dtype=int),
                np.full(len(x_mm), h_um, dtype=float),
                node_index,
                x_mm,
                y_mm,
                (p_vec - bearing.pa) * 1e-6,
            ]
        )
        np.savetxt(
            buffer,
            rows,
            delimiter=",",
            fmt=["%d", "%.10g", "%d", "%.10g", "%.10g", "%.10g"],
        )

    return buffer.getvalue().encode("utf-8")


def build_export_figures(bearing, results):
    """Build the complete export figure set from bearing geometry and results.

    Generates bearing shape plots, key results plots, 2D pressure visualizations,
    and a results legend. Applies consistent export styling.

    Args:
        bearing: Bearing instance with geometry.
        results: List of Result objects from solvers.

    Returns:
        list: List of (name, figure) tuples ready for export.
    """
    figures = []

    for idx, fig in enumerate(plot_bearing_shape(bearing), start=1):
        if _is_empty_figure(fig):
            continue
        figures.append((f"shape_{idx}_{_slugify(_export_title(fig, 'shape'))}", fig))

    for idx, fig in enumerate(plot_key_results(bearing, results, legend=True), start=1):
        if _is_empty_figure(fig):
            continue
        figures.append((f"results_{idx}_{_slugify(_export_title(fig, 'result'))}", fig))

    static_2d, moving_2d = _get_2d_results_pair(results)
    shared_z_range = get_pressure_2d_z_range(
        bearing, [r for r in [static_2d, moving_2d] if r is not None]
    )

    if static_2d is not None:
        figures.append(
            (
                "pressure_2d_static",
                plot_pressure_2d(
                    bearing,
                    static_2d,
                    slider=False,
                    include_frames=False,
                    show_colorbar=True,
                    z_range_mpa=shared_z_range,
                ),
            )
        )

    if moving_2d is not None:
        figures.append(
            (
                "pressure_2d_moving",
                plot_pressure_2d(
                    bearing,
                    moving_2d,
                    slider=False,
                    include_frames=False,
                    show_colorbar=True,
                    z_range_mpa=shared_z_range,
                ),
            )
        )

    legend_fig = plot_legend_only(results)
    if not _is_empty_figure(legend_fig):
        figures.append(("results_legend", legend_fig))

    return figures


def build_export_zip(*, app_state, note, bearing, results):
    """Build a zip archive containing CSV trace data and metadata."""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "inputs.csv",
            _metadata_csv_bytes(app_state=app_state, note=note),
            compress_type=zipfile.ZIP_DEFLATED,
        )

        for result_index, result in enumerate(results, start=1):
            solver_slug = _solver_slug(result)
            prefix = f"{result_index:02d}_{solver_slug}"

            zf.writestr(
                f"{prefix}_results.csv",
                _result_summary_csv_bytes(bearing, result),
                compress_type=zipfile.ZIP_DEFLATED,
            )

    return zip_buffer.getvalue()
