"""Plotly-based plotting utilities for interactive bearing visualization.

This module provides Plotly-based functions for rendering bearing performance metrics,
pressure distributions, and geometric visualization in interactive web-based dashboards.
Functions support animations, sliders, and 3D rendering for comprehensive analysis.
"""

import numpy as np
import plotly.graph_objects as go

EXPORT_FIGURE_WIDTH = 8.0
EXPORT_FIGURE_HEIGHT = 6.0
EXPORT_DPI = 72

EXPORT_FIGURE_WIDTH_PX = EXPORT_FIGURE_WIDTH * EXPORT_DPI
EXPORT_FIGURE_HEIGHT_PX = EXPORT_FIGURE_HEIGHT * EXPORT_DPI

# Plot styling
PLOT_FONT = dict(
    family="Arial",
    size=12,
)

# Solution colors
SOLVER_COLORS = {
    "analytic": "black",
    "numeric 1d": "red",
    "numeric 2d": "blue",
    "numeric 2d nonlinear": "green",
}

# Common axis properties
AXIS_STYLE = dict(
    title_font=PLOT_FONT,
    tickfont=PLOT_FONT,
    showline=True,
    showgrid=False,
    linecolor="black",
    ticks="inside",
    mirror=True,
)

FIG_LAYOUT = dict(
    font=PLOT_FONT,
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.7),
    showlegend=True,
    margin=dict(l=10, r=10, t=80, b=10),
)

MESH_DULL_LIGHTING = dict(
    ambient=0.95,
    diffuse=0.55,
    specular=0.0,
    roughness=1.0,
    fresnel=0.0,
)

MESH_DULL_LIGHTPOSITION = dict(x=0.0, y=0.0, z=100.0)


def _layout_with_legend(title: str, legend: bool) -> dict:
    """Return base figure layout with optional legend visibility and title.

    Args:
        title: Title to display at top of figure.
        legend: Whether to show legend.

    Returns:
        dict: Layout dictionary for Plotly figure.
    """
    layout = dict(FIG_LAYOUT)
    layout["title"] = title
    layout["showlegend"] = bool(legend)
    return layout


def _solver_color(res) -> str:
    """Resolve consistent line color for a solver result object.

    Args:
        res: Result object with name attribute.

    Returns:
        str: Color name (from SOLVER_COLORS) or 'gray' as fallback.
    """
    return SOLVER_COLORS.get(getattr(res, "name", ""), "gray")


def plot_key_results(bearing, results, *, legend=True):
    """Generate comprehensive set of key result plots for bearing analysis.

    Creates plots for load capacity, stiffness, 1D pressure (if available),
    supply and ambient flow rates, and optionally chamber flow, moment, and shear force.
    Uses Plotly for interactive visualization.

    Args:
        bearing: Bearing object with properties (type, ha, xa, etc.).
        results: Result object or list of Result objects.
        legend: Whether to show legends in plots. Defaults to True.

    Returns:
        list: List of Plotly Figure objects.
    """
    results = [results] if not isinstance(results, list) else results
    figs = []
    b = bearing

    figs.append(plot_load_capacity(b, results, legend=legend))
    figs.append(plot_stiffness(b, results, legend=legend))
    if any(
        getattr(res, "p", None) is not None or getattr(res, "p_1d", None) is not None
        for res in results
    ):
        figs.append(plot_pressure_1d(b, results, legend=legend))
    figs.append(plot_supply_flow_rate(b, results, legend=legend))
    figs.append(plot_ambient_flow_rate(b, results, legend=legend))
    if b.type == "seal":
        figs.append(plot_chamber_flow_rate(b, results, legend=legend))
    if any(getattr(res, "moment", None) is not None for res in results):
        figs.append(plot_moment(b, results, legend=legend))
    if any(getattr(res, "shear_force", None) is not None for res in results):
        figs.append(plot_shear_force(b, results, legend=legend))
    return figs


def plot_bearing_shape(bearing):
    """Generate all bearing geometry plots (XY, XZ, 3D mesh).

    Args:
        bearing: Bearing object with basis and geometry properties.

    Returns:
        list: List of Plotly Figure objects
            (XY profile, XZ profile, and optional 3D geometry).
    """
    figs = []
    b = bearing

    figs.append(plot_xy_shape(b))
    figs.append(plot_xz_shape(b))
    if b.fem_2d.basis is not None:
        figs.append(plot_geom_error(b))

    return figs


def plot_legend_only(results):
    """Create a compact figure that displays only unique solver legend entries.

    Args:
        results: Result object or list of Result objects.

    Returns:
        plotly.graph_objects.Figure: Minimal figure with legend only.
    """
    results = [results] if not isinstance(results, list) else results
    fig = go.Figure()

    seen = set()
    for res in results:
        name = getattr(res, "name", "result")
        if name in seen:
            continue
        seen.add(name)

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines+markers",
                name=name,
                line=dict(color=_solver_color(res)),
                marker=dict(color=_solver_color(res), size=8, symbol="circle"),
                showlegend=True,
            )
        )

    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=0.5,
            yanchor="middle",
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=0, b=0),
        height=70,
    )
    return fig


def plot_load_capacity(bearing, results, *, legend=True):
    """Plot load capacity versus film height for one or more solver results.

    Args:
        bearing: Bearing object with film thickness and maximum values.
        results: Result object or list of Result objects with w attribute.
        legend: Whether to show legend. Defaults to True.

    Returns:
        plotly.graph_objects.Figure: Interactive line plot.
    """
    results = [results] if not isinstance(results, list) else results
    fig = go.Figure()
    b = bearing
    xh = b.ha.flatten() * 1e6

    for res in results:
        color = _solver_color(res)

        idx_k_max = int(np.argmax(res.k))
        yw = res.w

        fig.add_trace(
            go.Scatter(
                x=xh,
                y=yw,
                name=res.name,
                showlegend=True,
                mode="lines+markers",
                marker=dict(
                    color=color,
                    size=[8 if i == idx_k_max else 0 for i in range(b.nh)],
                    symbol="circle",
                ),
                line=dict(color=color),
                hovertemplate=("h: %{x:.1f} μm <br>w: %{y:.1f} N<extra></extra>"),
            )
        )

    fig.update_xaxes(title_text="h (μm)", range=[0, b.ha_max * 1e6], **AXIS_STYLE)
    fig.update_yaxes(title_text="w (N)", range=[None, None], **AXIS_STYLE)
    fig.update_layout(**_layout_with_legend("Load capacity", legend))
    return fig


def plot_stiffness(bearing, results, *, legend=True):
    """Plot static stiffness versus film height for one or more results.

    Args:
        bearing: Bearing object with film thickness and maximum values.
        results: Result object or list of Result objects with k attribute.
        legend: Whether to show legend. Defaults to True.

    Returns:
        plotly.graph_objects.Figure: Interactive line plot.
    """
    results = [results] if not isinstance(results, list) else results
    fig = go.Figure()
    b = bearing

    for res in results:
        color = _solver_color(res)
        xh = b.ha.flatten() * 1e6
        yk = res.k
        idx_k_max = int(np.argmax(res.k))

        fig.add_trace(
            go.Scatter(
                x=xh,
                y=yk,
                name=res.name,
                showlegend=True,
                mode="lines+markers",
                marker=dict(
                    color=color,
                    size=[8 if i == idx_k_max else 0 for i in range(b.nh)],
                    symbol="circle",
                ),
                line=dict(color=color),
                hovertemplate=("h: %{x:.1f} μm <br>k: %{y:.1f} N/μm<extra></extra>"),
            )
        )

    fig.update_xaxes(title_text="h (μm)", range=[0, b.ha_max * 1e6], **AXIS_STYLE)
    fig.update_yaxes(title_text="k (N/μm)", **AXIS_STYLE)
    fig.update_layout(**_layout_with_legend("Static stiffness", legend))
    return fig


def plot_pressure_1d(bearing, results, *, legend=True):
    """Plot selected 1D pressure profiles at representative film heights.

    Supports both grid-based (analytic, result.p) and FEM DOF-based
    (result.p_1d) pressure data.

    Args:
        bearing: Bearing object with x coordinates and properties.
        results: Result object or list of Result objects with p or p_1d.
        legend: Whether to show legend. Defaults to True.

    Returns:
        plotly.graph_objects.Figure: Interactive line plot of pressure profiles.
    """
    results = [results] if not isinstance(results, list) else results
    fig = go.Figure()
    b = bearing

    for result in results:
        p_grid = getattr(result, "p", None)
        p_1d = getattr(result, "p_1d", None)

        if p_grid is not None:
            x_coords = b.x
            p_profiles = p_grid
            n_points = len(x_coords)
        elif p_1d is not None:
            sort_idx = np.argsort(b.fem_1d.basis.doflocs[0])
            x_coords = b.fem_1d.basis.doflocs[0][sort_idx]
            p_profiles = p_1d[:, sort_idx].T
            n_points = len(x_coords)
        else:
            continue

        color = SOLVER_COLORS.get(getattr(result, "name", ""), "purple")
        idx_k_max = int(np.argmax(result.k))
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=color),
                name=result.name,
                showlegend=True,
                hovertemplate=("h: %{x:.1f} μm <br>p: %{y:.2f} MPa<extra></extra>"),
            )
        )
        n_plots = 3
        h_indices = [1, idx_k_max, b.nh - 1]
        h_plots = b.ha[h_indices]
        t_locations = np.round(np.linspace(n_points, 0, n_plots + 2)[1:-1]).astype(int)
        for h_plot, t_loc in zip(h_plots, t_locations, strict=True):
            in_h = np.abs(b.ha - h_plot).argmin()
            pressures = (p_profiles[:, in_h] - b.pa) * 1e-6
            fig.add_trace(
                go.Scatter(
                    x=x_coords * 1e3,
                    y=pressures,
                    mode="lines+text",
                    textposition="top center",
                    text=[
                        f"{h_plot * 1e6:.2f} μm" if i == t_loc else None
                        for i in range(n_points)
                    ],
                    textfont=dict(color=color),
                    name=f"{result.name} {h_plot * 1e6:.1f} μm",
                    line=dict(color=color),
                    showlegend=False,
                    hovertemplate=("h: %{x:.1f} μm <br>p: %{y:.2f} MPa<extra></extra>"),
                ),
            )
    fig.update_xaxes(title_text="r (mm)", range=[0, b.xa * 1e3], **AXIS_STYLE)
    fig.update_yaxes(title_text="p (MPa)", range=[None, None], **AXIS_STYLE)
    fig.update_layout(**_layout_with_legend("Pressure distribution", legend))
    return fig


def get_pressure_2d_z_range(bearing, results):
    """Return shared z-axis/color range [min, max] in MPa for 2D pressure plots.

    Args:
        bearing: Bearing object with fem_2d.basis and pressure reference
            properties (pa, ps, pc).
        results: Result object or list of Result objects with p_2d.

    Returns:
        list: [min_pressure_mpa, max_pressure_mpa] for consistent colorbar scaling.
    """
    b = bearing
    if getattr(getattr(b, "fem_2d", None), "basis", None) is None:
        return [0.0, 1.0]

    results = [results] if not isinstance(results, list) else results
    results = [
        r for r in results if r is not None and getattr(r, "p_2d", None) is not None
    ]
    if not results:
        return [0.0, 1.0]

    z_max_pa = -np.inf
    z_min_pa = np.inf

    for result in results:
        n_idx = len(result.p_2d)
        for idx in range(n_idx):
            _, _, p_vtx, _, _, _ = _mesh_xyz_tris(b.fem_2d.basis, result.p_2d[idx])
            z_pa = np.nan_to_num(p_vtx - b.pa, nan=0.0, posinf=0.0, neginf=0.0)
            z_max_pa = max(z_max_pa, float(np.max(z_pa)))
            z_min_pa = min(z_min_pa, float(np.min(z_pa)))

    if not np.isfinite(z_max_pa) or z_max_pa <= 0:
        z_max_pa = 1.0
    z_max_mpa = np.max([z_max_pa, b.ps, b.pa, b.pc]) * 1e-6

    z_min_mpa = z_min_pa * 1e-6
    if not np.isfinite(z_min_mpa) or z_min_mpa >= 0:
        z_min_mpa = 0.0

    return [z_min_mpa, z_max_mpa]


def plot_pressure_2d(
    bearing,
    results,
    *,
    legend=True,
    slider=True,
    include_frames=True,
    show_colorbar=True,
    z_range_mpa=None,
):
    """Render 2D pressure field on FEM mesh with optional animation slider."""
    b = bearing
    if getattr(getattr(b, "fem_2d", None), "basis", None) is None:
        return empty_figure()

    results = [results] if not isinstance(results, list) else results
    fig = go.Figure()
    z_min, z_max = z_range_mpa if z_range_mpa is not None else [0.0, 1.0]
    plot_title = "Pressure distribution (2d numeric)"

    for result in results:
        if getattr(result, "p_2d", None) is None:
            continue
        if len(results) == 1 and getattr(result, "name", ""):
            plot_title = f"Pressure distribution ({result.name})"
        idx_k_max = int(np.argmax(result.k))
        n_idx = len(result.p_2d)

        x, y, z, i, j, k = _mesh_xyz_tris(b.fem_2d.basis, result.p_2d[idx_k_max])

        def _z_for(result, idx):
            _, _, p_vtx, _, _, _ = _mesh_xyz_tris(b.fem_2d.basis, result.p_2d[idx])
            z = p_vtx - b.pa
            return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        z = _z_for(result, idx_k_max)

        if z_range_mpa is None:
            z_max = max(float(np.max(_z_for(result, idx))) for idx in range(n_idx))
            if not np.isfinite(z_max) or z_max <= 0:
                z_max = 1.0
            z_max = np.max([z_max, b.ps, b.pa, b.pc]) * 1e-6

            z_min = (
                min(float(np.min(_z_for(result, idx))) for idx in range(n_idx)) * 1e-6
            )
            if not np.isfinite(z_min) or z_min >= 0:
                z_min = 0.0

        fig.add_trace(
            go.Mesh3d(
                x=x * 1e3,
                y=y * 1e3,
                z=z * 1e-6,
                i=i,
                j=j,
                k=k,
                intensity=z * 1e-6,
                colorscale="jet",
                cmin=z_min,
                cmax=z_max,
                showscale=show_colorbar,
                colorbar=dict(title="p (MPa)", thickness=15),
                flatshading=True,
                lighting=MESH_DULL_LIGHTING,
                lightposition=MESH_DULL_LIGHTPOSITION,
                hovertemplate=(
                    "x: %{x:.1f} mm <br>y: %{y:.1f} mm"
                    "<br>z: %{z:.2f} MPa<extra></extra>"
                ),
                name=result.name,
            )
        )
        if include_frames:
            frames = []
            for idx in range(n_idx):
                frame_traces = []
                z = _z_for(result, idx)
                frame_traces.append(
                    go.Mesh3d(
                        x=x * 1e3,
                        y=y * 1e3,
                        z=z * 1e-6,
                        i=i,
                        j=j,
                        k=k,
                        intensity=z * 1e-6,
                        colorscale="jet",
                        cmin=z_min,
                        cmax=z_max,
                        showscale=show_colorbar,
                        colorbar=dict(title="p (MPa)", thickness=15),
                        flatshading=True,
                        lighting=MESH_DULL_LIGHTING,
                        lightposition=MESH_DULL_LIGHTPOSITION,
                        name=result.name,
                    )
                )
                frames.append(go.Frame(data=frame_traces, name=str(idx)))

            fig.frames = frames
        if slider and n_idx > 1:
            fig.update_layout(
                sliders=[
                    {
                        "active": idx_k_max,
                        "currentvalue": {
                            "prefix": "h: ",
                            "suffix": " μm",
                            "font": {"size": 12},
                        },
                        "pad": {"t": 30},
                        "steps": [
                            {
                                "label": f"{b.ha[idx] * 1e6:.2f}",
                                "method": "animate",
                                "args": [
                                    [str(idx)],
                                    {
                                        "mode": "immediate",
                                        "frame": {"duration": 0, "redraw": True},
                                        "transition": {"duration": 0},
                                    },
                                ],
                            }
                            for idx in range(n_idx)
                        ],
                    }
                ]
            )

    fig.update_layout(
        scene=_scene_3d_layout(b, z_title="p (MPa)", z_range=[z_min, z_max]),
        title=plot_title,
        **FIG_LAYOUT,
    )
    return fig


def plot_supply_flow_rate(bearing, results, *, legend=True):
    """Plot the supply flow rate versus film height.

    Args:
        bearing: Bearing object with film thickness and maximum values.
        results: Result object or list of Result objects with qs attribute.
        legend: Whether to show legend. Defaults to True.

    Returns:
        plotly.graph_objects.Figure: Interactive line plot.
    """
    results = [results] if not isinstance(results, list) else results

    fig = go.Figure()
    b = bearing

    for result in results:
        color = SOLVER_COLORS.get(result.name, "purple")
        idx_k_max = np.argmax(result.k)

        fig.add_trace(
            go.Scatter(
                x=b.ha.flatten() * 1e6,
                y=result.qs,
                name=result.name,
                mode="lines+markers",
                marker=dict(
                    color=color,
                    size=[8 if i == idx_k_max else 0 for i in range(b.nx)],
                    symbol="circle",
                ),
                line=dict(color=color),
                hovertemplate=(
                    "h: %{x:.1f} μm <br>q<sub>s</sub>: %{y:.2f} L/min<extra></extra>"
                ),
            )
        )

    fig.update_xaxes(title_text="h (μm)", range=[0, b.ha_max * 1e6], **AXIS_STYLE)
    fig.update_yaxes(
        title_text="q<sub>s</sub> (L/min)", range=[None, None], **AXIS_STYLE
    )
    fig.update_layout(**_layout_with_legend("Supply flow", legend))
    return fig


def plot_chamber_flow_rate(bearing, results, *, legend=True):
    """Plot the chamber flow rate versus film height.

    Args:
        bearing: Bearing object with film thickness and maximum values.
        results: Result object or list of Result objects with qc attribute.
        legend: Whether to show legend. Defaults to True.

    Returns:
        plotly.graph_objects.Figure: Interactive line plot.
    """
    results = [results] if not isinstance(results, list) else results
    fig = go.Figure()
    b = bearing

    for result in results:
        color = SOLVER_COLORS.get(result.name, "purple")
        idx_k_max = np.argmax(result.k)
        fig.add_trace(
            go.Scatter(
                x=b.ha.flatten() * 1e6,
                y=result.qc,
                name=result.name,
                mode="lines+markers",
                marker=dict(
                    color=color,
                    size=[8 if i == idx_k_max else 0 for i in range(b.nx)],
                    symbol="circle",
                ),
                line=dict(color=color),
                hovertemplate=(
                    "h: %{x:.1f} μm <br>q<sub>c</sub>: %{y:.2f} L/min<extra></extra>"
                ),
            )
        )

    fig.update_xaxes(title_text="h (μm)", range=[0, b.ha_max * 1e6], **AXIS_STYLE)
    fig.update_yaxes(
        title_text="q<sub>c</sub> (L/min)", range=[None, None], **AXIS_STYLE
    )
    fig.update_layout(**_layout_with_legend("Chamber flow", legend))
    return fig


def plot_ambient_flow_rate(bearing, results, *, legend=True):
    """Plot the ambient flow rate versus film height.

    Args:
        bearing: Bearing object with film thickness and maximum values.
        results: Result object or list of Result objects with qa attribute.
        legend: Whether to show legend. Defaults to True.

    Returns:
        plotly.graph_objects.Figure: Interactive line plot.
    """
    results = [results] if not isinstance(results, list) else results

    fig = go.Figure()
    b = bearing

    for result in results:
        color = SOLVER_COLORS.get(result.name, "purple")
        idx_k_max = np.argmax(result.k)

        fig.add_trace(
            go.Scatter(
                x=b.ha.flatten() * 1e6,
                y=result.qa,
                name=result.name,
                mode="lines+markers",
                marker=dict(
                    color=color,
                    size=[8 if i == idx_k_max else 0 for i in range(b.nx)],
                    symbol="circle",
                ),
                line=dict(color=color),
                hovertemplate=(
                    "h: %{x:.1f} μm <br>q<sub>a</sub>: %{y:.2f} L/min<extra></extra>"
                ),
            )
        )

    fig.update_xaxes(title_text="h (μm)", range=[0, b.ha_max * 1e6], **AXIS_STYLE)
    fig.update_yaxes(
        title_text="q<sub>a</sub> (L/min)", range=[None, None], **AXIS_STYLE
    )
    fig.update_layout(**_layout_with_legend("Ambient flow", legend))
    return fig


def plot_moment(bearing, results, *, legend=True):
    """Plot pressure moment magnitude versus film height.

    Args:
        bearing: Bearing object with film thickness and maximum values.
        results: Result object or list of Result objects
            with moment attribute (2D vectors).
        legend: Whether to show legend. Defaults to True.

    Returns:
        plotly.graph_objects.Figure: Interactive line plot of moment vector magnitudes.
    """
    results = [results] if not isinstance(results, list) else results

    fig = go.Figure()
    b = bearing
    xh = b.ha.flatten() * 1e6
    plotted_magnitudes = []

    for result in results:
        moment = getattr(result, "moment", None)
        if moment is None:
            continue

        moment = np.asarray(moment)
        if moment.ndim != 2 or moment.shape[1] != 2:
            continue

        color = _solver_color(result)
        idx_k_max = int(np.argmax(result.k))
        moment = (moment[:, 0] ** 2 + moment[:, 1] ** 2) ** 0.5
        plotted_magnitudes.append(moment)
        fig.add_trace(
            go.Scatter(
                x=xh,
                y=moment,
                name=f"M {result.name}",
                mode="lines+markers",
                marker=dict(
                    color=color,
                    size=[8 if i == idx_k_max else 0 for i in range(b.nh)],
                    symbol="circle",
                ),
                line=dict(color=color, dash="solid"),
                showlegend=True,
                hovertemplate=("h: %{x:.1f} μm <br>M: %{y:.2f} N·m<extra></extra>"),
            )
        )

    y_range = [None, None]
    if plotted_magnitudes:
        max_magnitude = float(
            np.nanmax(
                np.concatenate([np.asarray(m).ravel() for m in plotted_magnitudes])
            )
        )
        if not np.isfinite(max_magnitude) or max_magnitude < 0.1:
            y_range = [0.0, 0.1]

    fig.update_xaxes(title_text="h (μm)", range=[0, b.ha_max * 1e6], **AXIS_STYLE)
    fig.update_yaxes(title_text="M (N·m)", range=y_range, **AXIS_STYLE)
    fig.update_layout(**_layout_with_legend("Tilting moment", legend))
    return fig


def plot_shear_force(bearing, results, *, legend=True):
    """Plot shear-force magnitude versus film height.

    Args:
        bearing: Bearing object with film thickness and maximum values.
        results: Result object or list of Result objects
            with shear_force attribute (2D vectors).
        legend: Whether to show legend. Defaults to True.

    Returns:
        plotly.graph_objects.Figure: Interactive line plot of shear force magnitudes.
    """
    results = [results] if not isinstance(results, list) else results

    fig = go.Figure()
    b = bearing
    xh = b.ha.flatten() * 1e6
    plotted_magnitudes = []

    for result in results:
        shear_force = getattr(result, "shear_force", None)
        if shear_force is None:
            continue

        shear_force = np.asarray(shear_force)
        if shear_force.ndim != 2 or shear_force.shape[1] != 2:
            continue

        color = _solver_color(result)
        idx_k_max = int(np.argmax(result.k))
        shear_force = (shear_force[:, 0] ** 2 + shear_force[:, 1] ** 2) ** 0.5
        plotted_magnitudes.append(shear_force)
        fig.add_trace(
            go.Scatter(
                x=xh,
                y=shear_force,
                name=f"F {result.name}",
                mode="lines+markers",
                marker=dict(
                    color=color,
                    size=[8 if i == idx_k_max else 0 for i in range(b.nh)],
                    symbol="circle",
                ),
                line=dict(color=color),
                showlegend=True,
                hovertemplate=("h: %{x:.1f} μm <br>F: %{y:.2f} N<extra></extra>"),
            )
        )

    y_range = [None, None]
    if plotted_magnitudes:
        max_magnitude = float(
            np.nanmax(
                np.concatenate([np.asarray(f).ravel() for f in plotted_magnitudes])
            )
        )
        if not np.isfinite(max_magnitude) or max_magnitude < 0.1:
            y_range = [0.0, 0.1]

    fig.update_xaxes(title_text="h (μm)", range=[0, b.ha_max * 1e6], **AXIS_STYLE)
    fig.update_yaxes(title_text="F (N)", range=y_range, **AXIS_STYLE)
    fig.update_layout(**_layout_with_legend("Shear force", legend))
    return fig


def plot_xy_shape(bearing):
    """Plot XY footprint of the bearing geometry.

    Args:
        bearing: Bearing object with case, xa, ya, xc properties.

    Returns:
        plotly.graph_objects.Figure: Interactive 2D plot of bearing footprint.
    """
    fig = go.Figure()
    b = bearing
    # SHAPE XY
    match b.case:
        case "annular" | "circular":
            # outer circle
            theta = np.linspace(0, 2 * np.pi, 100)
            xa = b.xa * np.cos(theta) * 1e3
            ya = b.xa * np.sin(theta) * 1e3
            fig.add_trace(
                go.Scatter(
                    x=xa,
                    y=ya,
                    fill="toself",
                    fillcolor="lightgrey",
                    line=dict(color="black"),
                    name="Shape",
                    showlegend=True,
                ),
            )

            # inner hole
            if b.case == "annular":
                xc = b.xc * np.cos(theta) * 1e3
                yc = b.xc * np.sin(theta) * 1e3
                fig.add_trace(
                    go.Scatter(
                        x=xc,
                        y=yc,
                        fill="toself",
                        fillcolor="white",
                        line=dict(color="black"),
                        name="Shape",
                        showlegend=False,
                    ),
                )

            # symmetry lines
            fig.add_trace(
                go.Scatter(
                    x=[0, 0],
                    y=[-b.xa * 0.2e3, b.xa * 0.2e3],
                    mode="lines",
                    line=dict(color="gray", width=1, dash="dashdot"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=[-b.xa * 0.2e3, b.xa * 0.2e3],
                    y=[0, 0],
                    mode="lines",
                    line=dict(color="gray", width=1, dash="dashdot"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
            )

        case "infinite":
            # right edge
            fig.add_trace(
                go.Scatter(
                    x=np.array([1, 1]) * b.xa * 1e3,
                    y=np.array([0, 1000]),
                    mode="lines",
                    line=dict(color="black"),
                    name="Shape",
                    showlegend=False,
                ),
            )
            # left edge
            fig.add_trace(
                go.Scatter(
                    x=np.array([0, 0]),
                    y=np.array([0, 1000]),
                    fill="tonextx",
                    fillcolor="lightgrey",
                    mode="lines",
                    line=dict(color="black"),
                    name="Shape",
                    showlegend=False,
                ),
            )

            fig.update_yaxes(range=[0, 1000])

            fig.update_xaxes(
                range=np.array([-0.5, 1.5]) * b.xa * 1e3,
            )

        case "rectangular":
            fig.add_trace(
                go.Scatter(
                    x=np.array([-1, -1, 1, 1, -1]) * b.xa * 0.5e3,
                    y=np.array([-1, 1, 1, -1, -1]) * b.ya * 0.5e3,
                    fill="toself",
                    fillcolor="lightgrey",
                    mode="lines",
                    line=dict(color="black"),
                    name="Shape",
                    showlegend=False,
                ),
            )
            fig.update_yaxes(
                range=np.array([-0.6, 0.6]) * b.ya * 1e3,
                **AXIS_STYLE,
            )
            fig.update_xaxes(
                range=np.array([-0.6, 0.6]) * b.xa * 1e3,
                scaleanchor="y",
                scaleratio=1,
                **AXIS_STYLE,
            )

        case _:
            pass

    fig.update_xaxes(title="x (mm)", **AXIS_STYLE)
    fig.update_yaxes(
        title="y (mm)",
        scaleanchor=False if b.csys == "cartesian" else "x",
        scaleratio=1,
        **AXIS_STYLE,
    )
    fig.update_layout(title="XY profile", **FIG_LAYOUT)
    return fig


def plot_xz_shape(bearing):
    """Plot XZ cross-section profile of geometry error or pad shape.

    Args:
        bearing: Bearing object with case, x, and geom_1d properties.

    Returns:
        plotly.graph_objects.Figure: Interactive 2D plot of XZ profile.
    """
    b = bearing
    fig = go.Figure()

    # PROFILE XZ
    match b.case:
        case "circular":
            x = np.concatenate(([-b.x[-1]], -np.flip(b.x), b.x, [b.x[-1]])) * 1e3
            y = np.concatenate(([100], np.flip(b.geom_1d), b.geom_1d, [100])) * 1e6
            t = ["Bearing<br>" if i == b.nx else None for i in range(b.nx * 2)]
        case "annular":
            x = (
                np.concatenate(
                    ([-b.x[-1]], -np.flip(b.x), [b.x[1]], [b.x[1]], b.x, [b.x[-1]])
                )
                * 1e3
            )
            y = (
                np.concatenate(
                    ([100], np.flip(b.geom_1d), [100], [100], b.geom_1d, [100])
                )
                * 1e6
            )
            t = [
                ("Bearing<br>" if i == i in [b.nx // 2, 2 + b.nx + b.nx // 2] else None)
                for i in range(b.nx * 2)
            ]
        case "infinite":
            x = np.concatenate(([b.x[1]], b.x, [b.x[-1]])) * 1e3
            y = np.concatenate(([100], b.geom_1d, [100])) * 1e6
            t = ["Bearing<br>" if i == b.nx // 2 else None for i in range(b.nx)]
        case "rectangular":
            return empty_figure()
        case _:
            return empty_figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            fill="toself",
            fillcolor="lightgrey",
            mode="lines+text",
            textposition="top center",
            text=t,
            textfont=dict(size=14),
            line=dict(color="black"),
            name="Bearing",
            showlegend=True,
        ),
    )

    # Guide surface XZ
    fig.add_trace(
        go.Scatter(
            x=np.array(
                [
                    (x[-1] - x[-1] * 1.2 if x[1] == 0 else x[1] * 1.2),
                    (x[1] + x[-1]) / 2,
                    x[-1] * 1.2,
                ]
            ),  # Convert to mm
            y=np.ones(3) * -0.1,  # Convert to um
            mode="lines+text",
            textposition="bottom center",
            text=[None, "<br>Guide surface", None],
            textfont=dict(size=14),
            line=dict(color="gray"),
            name="Shape",
            showlegend=False,
        ),
    )

    # symmetry line
    if b.csys == "polar":
        fig.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[-100, 100],
                mode="lines",
                line=dict(color="gray", width=1, dash="dashdot"),
                showlegend=False,
                hoverinfo="skip",
            ),
        )

    fig.update_xaxes(
        title="x (mm)",
        range=[(x[-1] - x[-1] * 1.1 if x[1] == 0 else x[1] * 1.1), x[-1] * 1.1],
        **AXIS_STYLE,
    )
    fig.update_yaxes(
        title="Shape (μm)",
        range=[-0.5 - 0.3e6 * abs(b.error), 1 + np.max(b.geom_1d) * 1e6],
        **AXIS_STYLE,
    )
    fig.update_layout(title="XZ profile", **FIG_LAYOUT)
    return fig


def plot_geom_error(bearing):
    """Plot 3D mesh and geometry error surface for FEM-capable bearings.

    Args:
        bearing: Bearing object with basis and geom_2d (2D geometry field).

    Returns:
        plotly.graph_objects.Figure: Interactive 3D surface plot.
    """
    b = bearing
    if (
        getattr(getattr(b, "fem_2d", None), "basis", None) is None
        or getattr(b, "geom_2d", None) is None
    ):
        return empty_figure()

    fig = go.Figure()

    x, y, z, i, j, k = _mesh_xyz_tris(b.fem_2d.basis, b.geom_2d)
    edge_pairs = _mesh_edge_pairs(b.fem_2d.basis, b.geom_2d)

    x_mm = x * 1e3
    y_mm = y * 1e3
    z_um = z * 1e6

    _add_guide_surface_with_grid(fig, x_mm, y_mm)

    fig.add_trace(
        go.Mesh3d(
            x=x_mm,
            y=y_mm,
            z=z_um,
            i=i,
            j=j,
            k=k,
            intensity=z_um,
            colorscale="jet",
            showscale=True,
            colorbar=dict(title="z (μm)", thickness=15),
            flatshading=True,
            lighting=MESH_DULL_LIGHTING,
            lightposition=MESH_DULL_LIGHTPOSITION,
            hovertemplate=(
                "x: %{x:.1f} mm <br>y: %{y:.1f} mm<br>z: %{z:.2f} μm<extra></extra>"
            ),
            name="",
        )
    )
    _add_mesh_edge_trace(fig, x_mm, y_mm, z_um, edge_pairs, color="black", width=0.5)

    fig.update_layout(
        scene=_scene_3d_layout(b, z_title="z (μm)"),
        title="Mesh & Geometry",
        **FIG_LAYOUT,
    )
    return fig


def _add_guide_surface_with_grid(fig, x_mm, y_mm, *, pad_ratio=0.1, grid_step_mm=10.0):
    """Add dark reference plane at z=0 and rectangular guide grid lines to 3D plot.

    Args:
        fig: Plotly Figure object to add traces to.
        x_mm: X coordinates in millimeters.
        y_mm: Y coordinates in millimeters.
        pad_ratio: Padding around coordinate bounds (fraction of span). Defaults to 0.1.
        grid_step_mm: Grid spacing in millimeters. Defaults to 10.0.
    """
    x_min, x_max = float(np.min(x_mm)), float(np.max(x_mm))
    y_min, y_max = float(np.min(y_mm)), float(np.max(y_mm))
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)
    x_pad = pad_ratio * x_span
    y_pad = pad_ratio * y_span

    x_bounds = np.array([x_min - x_pad, x_max + x_pad], dtype=float)
    y_bounds = np.array([y_min - y_pad, y_max + y_pad], dtype=float)

    x_grid = np.arange(
        np.ceil(x_bounds[0] / grid_step_mm) * grid_step_mm,
        np.floor(x_bounds[1] / grid_step_mm) * grid_step_mm + 1e-9,
        grid_step_mm,
    )
    y_grid = np.arange(
        np.ceil(y_bounds[0] / grid_step_mm) * grid_step_mm,
        np.floor(y_bounds[1] / grid_step_mm) * grid_step_mm + 1e-9,
        grid_step_mm,
    )

    fig.add_trace(
        go.Surface(
            x=np.tile(x_bounds, (2, 1)),
            y=np.tile(y_bounds.reshape(2, 1), (1, 2)),
            z=np.zeros((2, 2), dtype=float),
            showscale=False,
            opacity=0.38,
            colorscale=[[0.0, "#8f8f8f"], [1.0, "#8f8f8f"]],
            hoverinfo="skip",
            name="Guide surface",
            showlegend=False,
        )
    )

    for xg in x_grid:
        fig.add_trace(
            go.Scatter3d(
                x=[xg, xg],
                y=[y_bounds[0], y_bounds[1]],
                z=[0.0, 0.0],
                mode="lines",
                line=dict(color="#5f5f5f", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    for yg in y_grid:
        fig.add_trace(
            go.Scatter3d(
                x=[x_bounds[0], x_bounds[1]],
                y=[yg, yg],
                z=[0.0, 0.0],
                mode="lines",
                line=dict(color="#5f5f5f", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )


def _add_mesh_edge_trace(fig, x, y, z, edge_pairs, *, color="black", width=1):
    """Add a 3D line trace showing true mesh-cell edges for a surface.

    Args:
        fig: Plotly Figure object to add trace to.
        x: X coordinates of mesh vertices.
        y: Y coordinates of mesh vertices.
        z: Z coordinates of mesh vertices.
        edge_pairs: Iterable of (node_a, node_b) mesh-edge index pairs.
        color: Line color. Defaults to "black".
        width: Line width. Defaults to 1.
    """
    xe, ye, ze = [], [], []
    for a, b in edge_pairs:
        xe += [x[a], x[b], None]
        ye += [y[a], y[b], None]
        ze += [z[a], z[b], None]

    fig.add_trace(
        go.Scatter3d(
            x=xe,
            y=ye,
            z=ze,
            mode="lines",
            line=dict(color=color, width=width),
            showlegend=False,
            hoverinfo="skip",
        )
    )


def _mesh_edge_pairs(basis, dof_values, *, nrefs=1):
    """Return unique edge node pairs from refined cell connectivity.

    This preserves true cell boundaries for both triangular and quadrilateral
    meshes and avoids plotting artificial diagonal edges introduced by
    triangulation for rendering.
    """
    m_ref, _ = basis.refinterp(np.asarray(dof_values).ravel(), nrefs=nrefs)
    cells = np.asarray(m_ref.t, dtype=int)
    if cells.ndim != 2:
        raise ValueError("Unexpected refined mesh connectivity shape")

    nverts = cells.shape[0]
    if nverts not in (3, 4):
        raise ValueError(
            f"Unsupported element arity {nverts} in refined mesh connectivity"
        )

    edge_set = set()
    for cell in cells.T:
        for idx in range(nverts):
            a = int(cell[idx])
            b = int(cell[(idx + 1) % nverts])
            edge = (a, b) if a < b else (b, a)
            edge_set.add(edge)

    return sorted(edge_set)


def _scene_3d_layout(bearing, *, z_title: str, z_range=None):
    """Build a consistent 3D scene configuration for bearing mesh figures.

    Args:
        bearing: Bearing object with xa, ya, case properties.
        z_title: Title for z-axis.
        z_range: Optional [min, max] range for z-axis. If None, auto-scales.

    Returns:
        dict: Plotly scene configuration dictionary.
    """
    zaxis = dict(
        showgrid=False,
        showbackground=False,
        showline=True,
        linecolor="black",
        mirror=True,
        ticks="inside",
    )
    if z_range is not None:
        zaxis.update(range=z_range, autorange=False)

    return dict(
        domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
        aspectmode="manual",
        aspectratio=dict(
            x=1,
            y=bearing.ya / bearing.xa if bearing.case == "rectangular" else 1,
            z=0.5,
        ),
        xaxis_title="x (mm)",
        yaxis_title="y (mm)",
        zaxis_title=z_title,
        xaxis=dict(
            showgrid=False,
            showbackground=False,
            showline=True,
            linecolor="black",
            mirror=True,
            ticks="inside",
        ),
        yaxis=dict(
            showgrid=False,
            showbackground=False,
            showline=True,
            linecolor="black",
            mirror=True,
            ticks="inside",
        ),
        zaxis=zaxis,
        camera=dict(
            projection=dict(type="orthographic"),
            eye=dict(x=0.0, y=0.0, z=10),
            up=dict(x=0.0, y=1.0, z=0.0),
            center=dict(x=0.0, y=0.0, z=0.0),
        ),
    )


def _mesh_xyz_tris(basis, dof_values, *, nrefs=1):
    """Interpolate FEM dofs to plotting vertices and return triangle arrays.

    Args:
        basis: Finite element basis with refinterp method.
        dof_values: Degree-of-freedom values to interpolate.
        nrefs: Number of refinement levels. Defaults to 1.

    Supports refined triangular and quadrilateral meshes by triangulating
    quad cells for Plotly Mesh3d.

    Returns:
        tuple: (x, y, z, i, j, k) where x/y/z are node coordinates and
               i/j/k are triangle vertex indices.
    """
    m_ref, z_ref = basis.refinterp(np.asarray(dof_values).ravel(), nrefs=nrefs)
    x = np.asarray(m_ref.p[0]).ravel()
    y = np.asarray(m_ref.p[1]).ravel()
    cells = np.asarray(m_ref.t, dtype=int)
    if cells.ndim != 2:
        raise ValueError("Unexpected refined mesh connectivity shape")

    if cells.shape[0] == 3:
        i, j, k = cells
    elif cells.shape[0] == 4:
        i = np.concatenate([cells[0], cells[0]])
        j = np.concatenate([cells[1], cells[2]])
        k = np.concatenate([cells[2], cells[3]])
    else:
        raise ValueError(
            f"Unsupported element arity {cells.shape[0]} in refined mesh connectivity"
        )

    z = np.asarray(z_ref).ravel()
    return x, y, z, i, j, k


def empty_figure():
    """Create an empty figure with default layout.

    Useful as a placeholder when plotting is not possible due to missing data.

    Returns:
        plotly.graph_objects.Figure: Empty figure.
    """
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="White",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


def apply_export_layout(figure):
    """Apply constant export layout profile for publication-ready figures.

    Prepares figure for file export (PNG, PDF, etc.) with standardized sizing
    and marginsfor high-quality output.

    Args:
        figure: Plotly Figure object.

    Returns:
        plotly.graph_objects.Figure: Modified figure with export layout applied.
    """
    fig = go.Figure(figure)
    fig.update_layout(
        width=EXPORT_FIGURE_WIDTH_PX,
        height=EXPORT_FIGURE_HEIGHT_PX,
        margin=dict(l=8, r=8, t=24, b=8),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="center",
            x=0.5,
            font=PLOT_FONT,
        ),
        title=dict(font=PLOT_FONT),
        font=PLOT_FONT,
    )
    return fig
