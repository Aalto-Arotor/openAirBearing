"""Plotting utilities for bearing analysis and visualization.

This module provides functions for plotting bearing performance metrics (load capacity,
stiffness, flow rates), pressure distributions (1D and 2D), geometric shapes, and
comprehensive result summaries. Functions support both analytical
and FEM-based solutions.
"""

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from skfem.visuals.matplotlib import plot as skfem_plot

AUTO_COLORS = [
    "blue",
    "red",
    "black",
    "magenta",
    "gray",
    "purple",
    "green",
    "yellow",
    "cyan",
    "orange",
]

AUTO_LINESTYLES = [
    "solid",
]

AUTO_MARKERS = [
    "none",
]

AUTO_HIGHLIGHT_MARKERS = [
    "o",
]


def _validate_style_cycle(values, *, name):
    """Validate style cycle values and return as a list."""
    if values is None:
        raise ValueError(f"{name} cannot be None")
    values = list(values)
    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one item")
    return values


def _apply_title(ax, title):
    """Apply title, allowing None to disable it."""
    if title is None:
        ax.set_title("")
    else:
        ax.set_title(title)


def _style_legend(legend):
    """Apply consistent legend frame styling."""
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("black")


def _apply_mm_axis_format(ax):
    """Format x/y axes in millimeters when mesh coordinates are in meters."""
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 1e3:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 1e3:g}"))


def _as_list(results):
    """Convert a single result or list of results into a list.

    Args:
        results: A Result object or list of Result objects.

    Returns:
        list: Results as a list for uniform processing.
    """
    return [results] if not isinstance(results, list) else results


def assign_result_styles(
    results, *, colors=AUTO_COLORS, styles=AUTO_LINESTYLES, markers=AUTO_MARKERS
):
    """Assign consistent colors and styles to each result in a sequence.

    Result attributes already set (``color``, ``linestyle``, ``marker``) are left
    untouched to preserve manual overrides. Call this once after collecting all
    results and before any plotting. Assigned styles persist on Result objects.

    Args:
        results: A Result object or list of Result objects.
        colors: List of color names to assign cyclically. Defaults to AUTO_COLORS.
        styles: List of line styles to assign cyclically. Defaults to AUTO_LINESTYLES.
        markers: List of marker styles to assign cyclically. Defaults to AUTO_MARKERS.

    Returns:
        list: Results list with assigned color/style/marker fields.
    """
    results = _as_list(results)
    colors = _validate_style_cycle(colors, name="colors")
    styles = _validate_style_cycle(styles, name="styles")
    markers = _validate_style_cycle(markers, name="markers")

    color_idx = 0
    style_idx = 0
    marker_idx = 0
    for result in results:
        if getattr(result, "color", None) is None:
            result.color = colors[color_idx % len(colors)]
            color_idx += 1
        if getattr(result, "linestyle", None) is None:
            result.linestyle = styles[style_idx % len(styles)]
            style_idx += 1
        if getattr(result, "marker", None) is None:
            result.marker = markers[marker_idx % len(markers)]
            marker_idx += 1
    return results


def _solver_color(result, *, index=None):
    """Return a color for the result with fallback priority.

    Color priority:
    1. ``result.color`` – set manually or by assign_result_styles().
    2. ``AUTO_COLORS[index]`` – per-plot fallback when no color is stored.
    3. 'gray' – final fallback.

    Args:
        result: Result object.
        index: Optional index for AUTO_COLORS fallback.

    Returns:
        str: Color name for plotting.
    """
    color = getattr(result, "color", None)
    if color is not None:
        return color
    if index is not None:
        return AUTO_COLORS[index % len(AUTO_COLORS)]
    return "gray"


def _solver_linestyle(result, *, index=None):
    """Return a linestyle for the result with fallback priority."""
    linestyle = getattr(result, "linestyle", None)
    if linestyle is not None:
        return linestyle
    if index is not None:
        return AUTO_LINESTYLES[index % len(AUTO_LINESTYLES)]
    return "solid"


def _solver_marker(result, *, index=None):
    """Return a marker for the result with fallback priority."""
    marker = getattr(result, "marker", None)
    if marker is not None:
        return marker
    if index is not None:
        return AUTO_MARKERS[index % len(AUTO_MARKERS)]
    return "o"


def _solver_highlight_marker(result, *, index=None):
    """Return a highlight marker for max-k points with fallback priority."""
    highlight_marker = getattr(result, "highlight_marker", None)
    if highlight_marker is not None:
        return highlight_marker
    if index is not None:
        return AUTO_HIGHLIGHT_MARKERS[index % len(AUTO_HIGHLIGHT_MARKERS)]
    return "o"


def _metric_magnitude_2d(array):
    """Compute magnitude of 2D vector array (e.g., moment or shear force).

    Args:
        array: 2D array where columns 0 and 1 are vector components.

    Returns:
        np.ndarray or None: Magnitude of vectors, or None if input is invalid.
    """
    arr = np.asarray(array)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)
    return None


def _ensure_axes(ax=None, *, projection=None, figsize=None):
    """Ensure axes are available, creating figure if needed.

    Args:
        ax: Existing matplotlib axes or None to create new.
        projection: Projection type ('3d' or None).
        figsize: Figure size tuple (width, height) in inches.

    Returns:
        tuple: (figure, axes, was_created) where was_created
            indicates if new axes were created.
    """
    if ax is not None:
        return ax.figure, ax, False

    if projection == "3d":
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        return fig, ax, True

    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, True


def _resolve_pressure_index(result, n_fields, pressure_index=None):
    """Resolve pressure index with max-stiffness default and bounds checks.

    Args:
        result: Result object with optional ``k`` array.
        n_fields: Number of available pressure fields.
        pressure_index: Optional explicit pressure index. Supports negative indexing.

    Returns:
        int: Valid pressure index.

    Raises:
        IndexError: If the resolved index is out of range.
    """
    if n_fields <= 0:
        raise IndexError("No pressure fields available")

    if pressure_index is None:
        k_values = np.asarray(getattr(result, "k", []))
        idx = int(np.argmax(k_values)) if k_values.size > 0 else 0
    else:
        idx = int(pressure_index)

    if idx < 0:
        idx += n_fields
    if idx < 0 or idx >= n_fields:
        raise IndexError(
            f"pressure_index {pressure_index} out of range for {n_fields} fields"
        )
    return idx


def _select_pressure_field(result, *, pressure_index=None):
    """Select pressure field from a result.

    Args:
        result: Result object with p_2d and optional k attributes.
        pressure_index: Optional pressure field index. Defaults to max stiffness index.

    Returns:
        np.ndarray or None: Selected pressure field, or None if unavailable.
    """
    if getattr(result, "p_2d", None) is None:
        return None
    idx = _resolve_pressure_index(
        result, len(result.p_2d), pressure_index=pressure_index
    )
    return result.p_2d[idx]


def _plot_metric(
    bearing, results, values_getter, *, ylabel, title, legend=True, ax=None
):
    """Plot a metric (load, stiffness, flow, etc.) vs film thickness.

    Args:
        bearing: Bearing object.
        results: Result object or list of Result objects.
        values_getter: Callable that extracts metric values
            from a Result (e.g., lambda r: r.k).
        ylabel: Label for y-axis.
        title: Plot title. Use None to disable the title.
        legend: Whether to show legend. Defaults to True.
        ax: Matplotlib axes or None to create new.

    Returns:
        matplotlib.figure.Figure: Figure containing the plot.
    """
    results = _as_list(results)
    fig, ax, created_new = _ensure_axes(ax=ax)
    xh = bearing.ha * 1e6

    for idx, result in enumerate(results):
        y = values_getter(result)
        if y is None:
            continue
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(xh):
            continue
        color = _solver_color(result, index=idx)
        linestyle = _solver_linestyle(result, index=idx)
        marker = _solver_marker(result, index=idx)
        highlight_marker = _solver_highlight_marker(result, index=idx)
        idx_k_max = int(np.argmax(result.k))
        ax.plot(
            xh, y, color=color, linestyle=linestyle, marker=marker, label=result.name
        )
        ax.scatter(
            [xh[idx_k_max]], [y[idx_k_max]], color=color, marker=highlight_marker, s=25
        )

    ax.set_xlabel("h (μm)")
    ax.set_ylabel(ylabel)
    _apply_title(ax, title)
    if legend:
        _style_legend(ax.legend())
    ax.grid(False)
    if created_new:
        fig.tight_layout()
    return fig


def plot_load_capacity(
    bearing, results, *, legend=True, title="Load capacity", ax=None
):
    """Plot load capacity vs film thickness.

    Args:
        bearing: Bearing object.
        results: Result object or list of Result objects.
        legend: Whether to show legend. Defaults to True.
        title: Plot title. Use None to disable the title.
        ax: Matplotlib axes or None to create new.

    Returns:
        matplotlib.figure.Figure: Figure containing the plot.
    """
    return _plot_metric(
        bearing,
        results,
        lambda r: getattr(r, "w", None),
        ylabel="w (N)",
        title=title,
        legend=legend,
        ax=ax,
    )


def plot_stiffness(bearing, results, *, legend=True, title="Static stiffness", ax=None):
    """Plot static stiffness vs film thickness.

    Args:
        bearing: Bearing object.
        results: Result object or list of Result objects.
        legend: Whether to show legend. Defaults to True.
        title: Plot title. Use None to disable the title.
        ax: Matplotlib axes or None to create new.

    Returns:
        matplotlib.figure.Figure: Figure containing the plot.
    """
    return _plot_metric(
        bearing,
        results,
        lambda r: getattr(r, "k", None),
        ylabel="k (N/μm)",
        title=title,
        legend=legend,
        ax=ax,
    )


def plot_supply_flow_rate(
    bearing, results, *, legend=True, title="Supply flow", ax=None
):
    """Plot supply flow rate vs film thickness.

    Args:
        bearing: Bearing object.
        results: Result object or list of Result objects.
        legend: Whether to show legend. Defaults to True.
        title: Plot title. Use None to disable the title.
        ax: Matplotlib axes or None to create new.

    Returns:
        matplotlib.figure.Figure: Figure containing the plot.
    """
    return _plot_metric(
        bearing,
        results,
        lambda r: getattr(r, "qs", None),
        ylabel="q_s (L/min)",
        title=title,
        legend=legend,
        ax=ax,
    )


def plot_chamber_flow_rate(
    bearing, results, *, legend=True, title="Chamber flow", ax=None
):
    """Plot chamber flow rate vs film thickness.

    Args:
        bearing: Bearing object.
        results: Result object or list of Result objects.
        legend: Whether to show legend. Defaults to True.
        title: Plot title. Use None to disable the title.
        ax: Matplotlib axes or None to create new.

    Returns:
        matplotlib.figure.Figure: Figure containing the plot.
    """
    return _plot_metric(
        bearing,
        results,
        lambda r: getattr(r, "qc", None),
        ylabel="q_c (L/min)",
        title=title,
        legend=legend,
        ax=ax,
    )


def plot_ambient_flow_rate(
    bearing, results, *, legend=True, title="Ambient flow", ax=None
):
    """Plot ambient flow rate vs film thickness.

    Args:
        bearing: Bearing object.
        results: Result object or list of Result objects.
        legend: Whether to show legend. Defaults to True.
        title: Plot title. Use None to disable the title.
        ax: Matplotlib axes or None to create new.

    Returns:
        matplotlib.figure.Figure: Figure containing the plot.
    """
    return _plot_metric(
        bearing,
        results,
        lambda r: getattr(r, "qa", None),
        ylabel="q_a (L/min)",
        title=title,
        legend=legend,
        ax=ax,
    )


def plot_moment(bearing, results, *, legend=True, title="Tilting moment", ax=None):
    """Plot tilting moment magnitude vs film thickness.

    Args:
        bearing: Bearing object.
        results: Result object or list of Result objects.
        legend: Whether to show legend. Defaults to True.
        title: Plot title. Use None to disable the title.
        ax: Matplotlib axes or None to create new.

    Returns:
        matplotlib.figure.Figure: Figure containing the plot.
    """
    return _plot_metric(
        bearing,
        results,
        lambda r: _metric_magnitude_2d(getattr(r, "moment", None)),
        ylabel="M (N·m)",
        title=title,
        legend=legend,
        ax=ax,
    )


def plot_shear_force(bearing, results, *, legend=True, title="Shear force", ax=None):
    """Plot shear force magnitude vs film thickness.

    Args:
        bearing: Bearing object.
        results: Result object or list of Result objects.
        legend: Whether to show legend. Defaults to True.
        title: Plot title. Use None to disable the title.
        ax: Matplotlib axes or None to create new.

    Returns:
        matplotlib.figure.Figure: Figure containing the plot.
    """
    return _plot_metric(
        bearing,
        results,
        lambda r: _metric_magnitude_2d(getattr(r, "shear_force", None)),
        ylabel="F (N)",
        title=title,
        legend=legend,
        ax=ax,
    )


def plot_pressure_1d(
    bearing,
    results,
    *,
    p_index=None,
    pressure_index=None,
    legend=True,
    title="Pressure distribution",
    ax=None,
):
    """Plot 1D pressure distribution.

    Supports both grid-based (analytic, result.p) and FEM DOF-based
    (result.p_1d) pressure data.

    Args:
        bearing: Bearing object with x coordinates.
        results: Result object or list of Result objects with p or p_1d.
        p_index: Optional pressure index. Defaults to max stiffness index.
        pressure_index: Alias for ``p_index``.
        legend: Whether to show legend. Defaults to True.
        title: Plot title. Use None to disable the title.
        ax: Matplotlib axes or None to create new.

    Returns:
        matplotlib.figure.Figure: Figure containing the plot.
    """
    results = _as_list(results)
    fig, ax, created_new = _ensure_axes(ax=ax)

    for idx, result in enumerate(results):
        p_grid = getattr(result, "p", None)
        p_1d = getattr(result, "p_1d", None)

        if p_grid is not None:
            x_coords = bearing.x
            p_profiles = np.asarray(p_grid)
        elif p_1d is not None:
            sort_idx = np.argsort(bearing.fem_1d.basis.doflocs[0])
            x_coords = bearing.fem_1d.basis.doflocs[0][sort_idx]
            p_profiles = np.asarray(p_1d)[:, sort_idx].T
        else:
            continue

        if p_index is not None and pressure_index is not None:
            raise ValueError("Use only one of p_index or pressure_index")

        selected_index = pressure_index if pressure_index is not None else p_index
        h_idx = _resolve_pressure_index(
            result,
            p_profiles.shape[1],
            pressure_index=selected_index,
        )

        pressures = (p_profiles[:, h_idx] - bearing.pa) * 1e-6
        ax.plot(
            x_coords * 1e3,
            pressures,
            label=result.name,
            color=_solver_color(result, index=idx),
            linestyle=_solver_linestyle(result, index=idx),
            marker=_solver_marker(result, index=idx),
        )

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("p (MPa)")
    _apply_title(ax, title)
    if legend:
        _style_legend(ax.legend())
    ax.grid(False)
    if created_new:
        fig.tight_layout()
    return fig


def get_pressure_2d_z_range(bearing, results):
    """Compute pressure range (min/max) across all 2D results for consistent colorbar.

    Args:
        bearing: Bearing object with ambient pressure reference ``pa``.
        results: Result object or list of Result objects with p_2d.

    Returns:
        list: [min_pressure_mpa, max_pressure_mpa] normalized by ambient pressure.
    """
    results = [r for r in _as_list(results) if getattr(r, "p_2d", None) is not None]
    if not results:
        return [0.0, 1.0]

    z_min_pa = np.inf
    z_max_pa = -np.inf
    for result in results:
        for pfield in result.p_2d:
            z = np.asarray(pfield).ravel()
            zr = np.nan_to_num(z - bearing.pa, nan=0.0, posinf=0.0, neginf=0.0)
            z_min_pa = min(z_min_pa, float(np.min(zr)))
            z_max_pa = max(z_max_pa, float(np.max(zr)))

    if not np.isfinite(z_min_pa):
        z_min_pa = 0.0
    if not np.isfinite(z_max_pa) or z_max_pa <= 0:
        z_max_pa = 1.0

    return [z_min_pa * 1e-6, z_max_pa * 1e-6]


def plot_pressure_2d(
    bearing,
    results,
    *,
    pressure_index=None,
    legend=True,
    slider=True,
    include_frames=True,
    show_colorbar=True,
    z_range_mpa=None,
    style="surface",
    title="auto",
    ax=None,
):
    """Plot 2D pressure distribution using scikit-fem visualization.

    Args:
        bearing: Bearing object with fem_2d.basis for mesh interpolation.
        results: Result object or list of Result objects with p_2d.
        pressure_index: Optional pressure field index. Defaults to max stiffness index.
        legend: Unused, kept for compatibility.
        slider: Unused, kept for compatibility.
        include_frames: Unused, kept for compatibility.
        show_colorbar: Whether to show colorbar. Defaults to True.
        z_range_mpa: Optional [min, max] pressure range in MPa for colorbar scaling.
        style: Unused, kept for compatibility with previous API.
        title: Plot title. Use "auto" for default dynamic title or None to disable.
        ax: Matplotlib axes or None to create new.

    Returns:
        matplotlib.figure.Figure: Figure containing the plot.
    """
    del legend, slider, include_frames, style

    results = _as_list(results)
    result = next((r for r in results if getattr(r, "p_2d", None) is not None), None)
    if result is None:
        return empty_figure()

    pfield = _select_pressure_field(result, pressure_index=pressure_index)
    z_mpa = (np.asarray(pfield).ravel() - bearing.pa) * 1e-6

    z_range = (
        z_range_mpa
        if z_range_mpa is not None
        else get_pressure_2d_z_range(bearing, [result])
    )

    fig, ax, created_new = _ensure_axes(ax=ax)
    skfem_plot(
        bearing.fem_2d.basis,
        z_mpa,
        ax=ax,
        cmap="jet",
        nrefs=1,
        vmin=z_range[0],
        vmax=z_range[1],
        colorbar=False,
    )
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    _apply_mm_axis_format(ax)
    title_text = (
        f"Pressure distribution ({getattr(result, 'name', 'result')})"
        if title == "auto"
        else title
    )
    _apply_title(ax, title_text)
    ax.set_aspect("equal")
    if show_colorbar:
        mappable = ax.collections[-1] if ax.collections else None
        if mappable is not None:
            fig.colorbar(mappable, ax=ax, label="p (MPa)")
    if created_new:
        fig.tight_layout()
    return fig


def plot_xy_shape(bearing, *, title="XY profile", ax=None):
    """Plot bearing footprint in XY plane.

    Args:
        bearing: Bearing object with geometry properties (case, xa, ya, xc).
        title: Plot title. Use None to disable the title.
        ax: Matplotlib axes or None to create new.

    Returns:
        matplotlib.figure.Figure: Figure containing the plot.
    """
    fig, ax, created_new = _ensure_axes(ax=ax)
    case = bearing.case

    if case in ("circular", "annular"):
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.fill(
            bearing.xa * np.cos(theta) * 1e3,
            bearing.xa * np.sin(theta) * 1e3,
            color="lightgray",
            edgecolor="black",
        )
        if case == "annular":
            ax.fill(
                bearing.xc * np.cos(theta) * 1e3,
                bearing.xc * np.sin(theta) * 1e3,
                color="white",
                edgecolor="black",
            )
        ax.set_aspect("equal")
    elif case == "rectangular":
        x = 0.5 * bearing.xa * 1e3
        y = 0.5 * bearing.ya * 1e3
        ax.fill([-x, -x, x, x], [-y, y, y, -y], color="lightgray", edgecolor="black")
        ax.set_aspect("equal")
    else:
        ax.plot([0, bearing.xa * 1e3], [0, 0], color="black")

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    _apply_mm_axis_format(ax)
    _apply_title(ax, title)
    if created_new:
        fig.tight_layout()
    return fig


def plot_xz_shape(bearing, *, title="XZ profile", ax=None):
    """Plot bearing geometry profile in XZ plane.

    Args:
        bearing: Bearing object with geometry properties (x, geom_1d).
        title: Plot title. Use None to disable the title.
        ax: Matplotlib axes or None to create new.

    Returns:
        matplotlib.figure.Figure: Figure containing the plot.
    """
    fig, ax, created_new = _ensure_axes(ax=ax)
    if getattr(bearing, "geom_1d", None) is None:
        return empty_figure()

    x = np.asarray(getattr(bearing, "x", [])) * 1e3
    z = np.asarray(bearing.geom_1d) * 1e6
    ax.plot(x, z, color="black")
    ax.fill_between(x, z, z.max() + 1, color="lightgray", alpha=0.6)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("Shape (μm)")
    _apply_title(ax, title)
    if created_new:
        fig.tight_layout()
    return fig


def plot_geometry_error(bearing, *, title="Mesh & Geometry", ax=None):
    """Plot 3D mesh with geometry error field.

    Args:
        bearing: Bearing object with fem_2d.basis and geom_2d (2D geometry field).
        title: Plot title. Use None to disable the title.
        ax: Matplotlib axes or None to create new.

    Returns:
        matplotlib.figure.Figure: Figure containing the plot.
    """
    basis_2d = getattr(getattr(bearing, "fem_2d", None), "basis", None)
    if basis_2d is None or getattr(bearing, "geom_2d", None) is None:
        return empty_figure()

    fig, ax, created_new = _ensure_axes(ax=ax)
    z_um = np.asarray(bearing.geom_2d).ravel() * 1e6

    if getattr(ax, "name", None) == "3d":
        x_mm = np.asarray(basis_2d.doflocs[0]).ravel() * 1e3
        y_mm = np.asarray(basis_2d.doflocs[1]).ravel() * 1e3
        triangulation = mtri.Triangulation(x_mm, y_mm)
        surface = ax.plot_trisurf(
            triangulation,
            z_um,
            cmap="jet",
            linewidth=0.0,
            antialiased=False,
        )
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("z (μm)")
        _apply_title(ax, title)
        fig.colorbar(surface, ax=ax, label="z (μm)")
        if created_new:
            fig.tight_layout()
        return fig

    skfem_plot(
        basis_2d,
        z_um,
        ax=ax,
        cmap="jet",
        nrefs=1,
        colorbar=False,
    )
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    _apply_title(ax, title)
    ax.set_aspect("equal")
    mappable = ax.collections[-1] if ax.collections else None
    if mappable is not None:
        fig.colorbar(mappable, ax=ax, label="z (μm)")
    if created_new:
        fig.tight_layout()
    return fig


def plot_bearing_shape(bearing):
    """Generate all bearing geometry plots (XY, XZ, 3D mesh).

    Args:
        bearing: Bearing object.

    Returns:
        list: List of matplotlib Figure objects.
    """
    figs = [plot_xy_shape(bearing), plot_xz_shape(bearing)]
    if getattr(getattr(bearing, "fem_2d", None), "basis", None) is not None:
        figs.append(plot_geometry_error(bearing))
    return figs


def plot_legend_only(results, *, ax=None):
    """Generate a standalone legend figure from results.

    Args:
        results: Result object or list of Result objects.
        ax: Matplotlib axes or None to create new.

    Returns:
        matplotlib.figure.Figure: Figure containing legend only.
    """
    results = _as_list(results)
    fig, ax, created_new = _ensure_axes(ax=ax, figsize=(5, 1.2))
    handles = []
    seen = {}
    for idx, result in enumerate(results):
        name = getattr(result, "name", "result")
        if name in seen:
            continue
        seen[name] = idx
        handles.append(
            Line2D(
                [0],
                [0],
                color=_solver_color(result, index=idx),
                linestyle=_solver_linestyle(result, index=idx),
                marker=_solver_marker(result, index=idx),
                lw=2,
                label=name,
            )
        )
    if handles:
        _style_legend(
            ax.legend(handles=handles, loc="center", ncol=max(1, len(handles)))
        )
    ax.axis("off")
    if created_new:
        fig.tight_layout()
    return fig


def plot_subplots(
    items,
    plot_func,
    *,
    n_cols=2,
    projection=None,
    figsize=None,
    figsize_per_subplot=(5.0, 4.0),
    title_getter=None,
    hide_unused=True,
    plot_kwargs=None,
):
    """Create a grid of subplots by applying a function to each item.

    Args:
        items: Iterable of objects to plot.
        plot_func: Callable(item, ax=ax, **plot_kwargs) that plots the item.
        n_cols: Number of subplot columns. Defaults to 2.
        projection: Axis projection type (e.g., '3d'). Defaults to None.
        figsize: Explicit figure size (width, height) in inches.
            If None, computed from items and figsize_per_subplot.
        figsize_per_subplot: Default size per subplot when figsize
            not provided. Defaults to (5.0, 4.0).
        title_getter: Optional callable(item) that returns a title for each subplot.
        hide_unused: Whether to hide unused subplot axes. Defaults to True.
        plot_kwargs: Dict of additional keyword arguments for plot_func.

    Returns:
        tuple: (figure, list_of_axes) for populated subplots.
    """
    items = list(items)
    if not items:
        return empty_figure(), []

    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(len(items) / n_cols))

    if figsize is None:
        figsize = (
            figsize_per_subplot[0] * n_cols,
            figsize_per_subplot[1] * n_rows,
        )

    fig = plt.figure(figsize=figsize)
    axes = [
        fig.add_subplot(n_rows, n_cols, index + 1, projection=projection)
        if projection
        else fig.add_subplot(n_rows, n_cols, index + 1)
        for index in range(n_rows * n_cols)
    ]

    call_kwargs_base = dict(plot_kwargs or {})
    for index, item in enumerate(items):
        ax = axes[index]
        call_kwargs = dict(call_kwargs_base)
        call_kwargs["ax"] = ax
        plot_func(item, **call_kwargs)
        if title_getter is not None:
            ax.set_title(title_getter(item))

    if hide_unused:
        for index in range(len(items), len(axes)):
            axes[index].axis("off")

    fig.tight_layout()
    return fig, axes[: len(items)]


def plot_key_results(bearing, results, *, legend=True):
    """Generate a set of key result plots for comprehensive bearing analysis.

    Creates plots for load capacity, stiffness, 1D pressure (if available),
    supply and ambient flow rates, and optionally chamber flow, moment, and shear force.

    Args:
        bearing: Bearing object.
        results: Result object or list of Result objects.
        legend: Whether to include legends in plots. Defaults to True.

    Returns:
        list: List of matplotlib Figure objects.
    """
    results = _as_list(results)
    figs = [
        plot_load_capacity(bearing, results, legend=legend),
        plot_stiffness(bearing, results, legend=legend),
    ]
    if any(
        getattr(result, "p", None) is not None
        or getattr(result, "p_1d", None) is not None
        for result in results
    ):
        figs.append(plot_pressure_1d(bearing, results, legend=legend))
    figs.extend(
        [
            plot_supply_flow_rate(bearing, results, legend=legend),
            plot_ambient_flow_rate(bearing, results, legend=legend),
        ]
    )
    if bearing.type == "seal":
        figs.append(plot_chamber_flow_rate(bearing, results, legend=legend))
    if any(getattr(result, "moment", None) is not None for result in results):
        figs.append(plot_moment(bearing, results, legend=legend))
    if any(getattr(result, "shear_force", None) is not None for result in results):
        figs.append(plot_shear_force(bearing, results, legend=legend))
    return figs


def empty_figure():
    """Create an empty figure with axes turned off.

    Useful as a placeholder when plotting is not possible due to missing data.

    Returns:
        matplotlib.figure.Figure: Empty figure.
    """
    fig, ax = plt.subplots()
    ax.axis("off")
    fig.tight_layout()
    return fig


def apply_export_layout(figure: Figure):
    """Apply export-ready layout formatting to a figure.

    Prepares figure for file export (PDF, PNG, etc.).
    Currently a pass-through placeholder for future layout adjustments.

    Args:
        figure: matplotlib Figure object.

    Returns:
        matplotlib.figure.Figure: The same figure object.
    """
    return figure
