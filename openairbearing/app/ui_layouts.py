"""Dash UI layout components for bearing analysis dashboard.

Defines layout structures for input controls, results display, and export configuration.
"""

from dash import dcc, html

from openairbearing.app.ui_form_schema import (
    BEARING_PARAMETER_FIELDS,
    CHAMBER_PARAMETER_FIELDS,
    FEM_PARAMETER_FIELDS,
    FLOW_PARAMETER_FIELDS,
    FLUID_PARAMETER_FIELDS,
    INNER_RADIUS_FIELDS,
    JOURNAL_PARAMETER_FIELDS,
    LENGTH_FIELDS,
    LOAD_PARAMETER_FIELDS,
    MODEL_PARAMETER_FIELDS,
    NUMERICAL_CONTROL_FIELDS,
    NUMERICAL_PARAMETER_FIELDS,
    NY_PARAMETER_FIELDS,
    SETUP_CONTROL_FIELDS,
)
from openairbearing.app.ui_plots import (
    empty_figure,
    get_pressure_2d_z_range,
    plot_bearing_shape,
    plot_key_results,
    plot_legend_only,
    plot_pressure_2d,
)
from openairbearing.app.ui_state import bearing_to_form_values
from openairbearing.app.ui_styles import STYLES


def _reset_button(reset_id, title="Reset to default"):
    """Create a standard reset button used beside input controls."""
    return html.Button(
        "↺",
        id=reset_id,
        title=title,
        style=STYLES["reset_button"],
    )


def _resettable_input_triplet(
    label,
    input_id,
    reset_id,
    value,
    *,
    label_id=None,
    input_type="number",
    min_value=None,
    max_value=None,
    step=None,
    input_mode=None,
    placeholder=None,
    read_only=False,
    disabled=False,
):
    """Create one [label, input, reset-button] triplet for grid-based form rows."""
    input_props = {
        "id": input_id,
        "type": input_type,
        "value": value,
        "style": STYLES["input"],
        # Only update the Dash callback graph after the user finishes typing.
        # This prevents recomputing results on every keystroke.
        "debounce": True,
    }
    if min_value is not None:
        input_props["min"] = min_value
    if max_value is not None:
        input_props["max"] = max_value
    if step is not None:
        input_props["step"] = step
    if input_mode is not None:
        input_props["inputMode"] = input_mode
    if placeholder is not None:
        input_props["placeholder"] = placeholder
    if read_only:
        input_props["readOnly"] = True
    if disabled:
        input_props["disabled"] = True

    label_component = html.Label(label, id=label_id) if label_id else html.Label(label)

    return [
        label_component,
        dcc.Input(**input_props),
        _reset_button(reset_id) if reset_id else html.Div(),
    ]


def _render_resettable_fields(field_specs, defaults):
    """Render multiple resettable field triplets from declarative specs."""
    children = []
    for spec in field_specs:
        children.extend(
            _resettable_input_triplet(
                spec["label"],
                spec["input_id"],
                spec.get("reset_id"),
                defaults[spec["value_key"]],
                label_id=spec.get("label_id"),
                input_type=spec.get("input_type", "number"),
                min_value=spec.get("min_value"),
                max_value=spec.get("max_value"),
                step=spec.get("step"),
                input_mode=spec.get("input_mode"),
                placeholder=spec.get("placeholder"),
                read_only=spec.get("read_only", False),
                disabled=spec.get("disabled", False),
            )
        )
    return children


def _control_triplet(label, component, *, label_id=None, spacer=False):
    """Create one [label, component, spacer] triplet for grid form rows."""
    label_component = html.Label(label, id=label_id) if label_id else html.Label(label)
    spacer_component = html.Label("") if spacer else html.Div()
    return [label_component, component, spacer_component]


def _render_controls(control_specs):
    """Render non-reset controls from declarative control specifications."""
    children = []
    for spec in control_specs:
        comp_type = spec["component"]
        props = spec.get("props", {})
        if comp_type == "dropdown":
            component = dcc.Dropdown(**props)
        elif comp_type == "checklist":
            component = dcc.Checklist(**props)
        else:
            raise ValueError(f"Unsupported control component: {comp_type}")

        children.extend(
            _control_triplet(
                spec["label"],
                component,
                label_id=spec.get("label_id"),
                spacer=spec.get("spacer", True),
            )
        )
    return children


def create_layout(default_bearing, bearing, results):
    """Create the main app layout.

    Args:
        bearing: Bearing instance
        results: List of calculation results

    Returns:
        html.Div: Main application layout
    """
    return html.Div(
        [
            dcc.Store(id="app-state", storage_type="memory"),
            dcc.Store(id="solve-request", storage_type="memory"),
            dcc.Store(id="solve-run", storage_type="memory"),
            dcc.Store(id="solve-progress", storage_type="memory"),
            dcc.Download(id="export-download"),
            html.Div(
                [
                    html.Img(
                        src="/assets/favicon.ico",
                        style={
                            "height": "40px",
                            "margin": "10px 5px",
                            "verticalAlign": "middle",
                        },
                    ),
                    html.H1(
                        "Open Air Bearing",
                        style={
                            "textAlign": "center",
                            "display": "inline-block",
                            "margin": "10px 0",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                },
            ),
            html.Div(
                [
                    html.Div(
                        [
                            create_input_layout(default_bearing),
                            create_export_layout(),
                        ],
                        style=STYLES["left_column_stack"],
                    ),
                    create_results_layout(bearing, results),
                ],
                style=STYLES["container"],
            ),
        ]
    )


def create_input_layout(default_bearing):
    """Create the left-side input panel with controls and reset buttons."""
    defaults = bearing_to_form_values(default_bearing)
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Setup", style={"margin": "0"}),
                    html.Button(
                        "Reset All",
                        id="reset-all",
                        title="Reset all values to default",
                        style=STYLES["reset_all_button"],
                    ),
                ],
                style=STYLES["header_container"],
            ),
            html.Div(
                _render_controls(SETUP_CONTROL_FIELDS),
                style=STYLES["input_container"],
            ),
            # Geometry inputs
            html.H4("Bearing parameters"),
            html.Div(
                _render_resettable_fields(BEARING_PARAMETER_FIELDS, defaults),
                style=STYLES["input_container"],
            ),
            html.Div(
                _render_resettable_fields(JOURNAL_PARAMETER_FIELDS, defaults),
                id="journal-container",
                style=STYLES["toggle_container"],
            ),
            html.Div(
                _render_resettable_fields(INNER_RADIUS_FIELDS, defaults),
                id="xc-container",
                style=STYLES["toggle_container"],
            ),
            html.Div(
                _render_resettable_fields(LENGTH_FIELDS, defaults),
                id="ya-container",
                style=STYLES["toggle_container"],
            ),
            html.Div(
                _render_resettable_fields(FLOW_PARAMETER_FIELDS, defaults),
                style=STYLES["input_container"],
            ),
            # Geometry inputs
            html.H4("Numerical model specific:"),
            html.Div(
                [
                    *_render_controls(NUMERICAL_CONTROL_FIELDS),
                    *_render_resettable_fields(NUMERICAL_PARAMETER_FIELDS, defaults),
                ],
                style=STYLES["input_container"],
            ),
            html.H4("Load parameters"),
            html.Div(
                _render_resettable_fields(LOAD_PARAMETER_FIELDS, defaults),
                style=STYLES["input_container"],
            ),
            html.Div(
                _render_resettable_fields(CHAMBER_PARAMETER_FIELDS, defaults),
                id="pc-container",
                style=STYLES["toggle_container"],
            ),
            html.H4("Fluid properties"),
            html.Div(
                _render_resettable_fields(FLUID_PARAMETER_FIELDS, defaults),
                style=STYLES["input_container"],
            ),
            html.H4("Model parameters"),
            html.Div(
                _render_resettable_fields(MODEL_PARAMETER_FIELDS, defaults),
                style=STYLES["input_container"],
            ),
            html.Div(
                _render_resettable_fields(FEM_PARAMETER_FIELDS, defaults),
                id="divs-container",
                style=STYLES["toggle_container"],
            ),
            html.Div(
                _render_resettable_fields(NY_PARAMETER_FIELDS, defaults),
                id="ny-container",
                style=STYLES["toggle_container"],
            ),
        ],
        style=STYLES["input_column"],
    )


def create_export_layout():
    """Create export controls in a standalone box below the input panel."""
    return html.Div(
        [
            html.H4("Export", style={"marginTop": "0"}),
            html.Div(
                [
                    html.Label("Exports inputs and results as csv files."),
                    dcc.Textarea(
                        id="export-note-input",
                        value="",
                        placeholder="Optional note attached to export",
                        style=STYLES["export_note_input"],
                    ),
                    html.Button(
                        "Export Results",
                        id="export-results-btn",
                        style=STYLES["export_button"],
                    ),
                    html.Div(id="export-status", style=STYLES["export_status"]),
                ],
                style=STYLES["export_container"],
            ),
        ],
        style=STYLES["input_column"],
    )


def create_results_layout(bearing, results):
    """Create the results section layout.

    Args:
        bearing: Bearing instance
        results: List of calculation results

    Returns:
        html.Div: Results layout with plots arranged in a grid.
    """
    # Get the list of plots from plot_key_results
    shape_figures = plot_bearing_shape(bearing)
    plot_figures = plot_key_results(bearing, results, legend=False)
    results = [results] if not isinstance(results, list) else results

    # Pad the list to multiple of 3
    while len(shape_figures) % 3 != 0:
        shape_figures.append(empty_figure())
    while len(plot_figures) % 3 != 0:
        plot_figures.append(empty_figure())
    # Create rows of plots (3 plots per row)
    shape_rows = []
    for i in range(0, len(shape_figures), 3):
        row = html.Div(
            [
                html.Div(
                    dcc.Graph(
                        figure=shape_figures[j],
                        config={"displayModeBar": False},
                        style={"height": "400px"},
                    ),
                    style={
                        "width": "33%",
                        "margin": "0px",
                        "padding": "0px",
                    },
                )
                for j in range(
                    i, min(i + 3, len(shape_figures))
                )  # Add up to 3 plots per row
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between",
            },  # Flexbox for row layout
        )
        shape_rows.append(row)

    result_rows = []
    for i in range(0, len(plot_figures), 3):
        row = html.Div(
            [
                html.Div(
                    dcc.Graph(
                        figure=plot_figures[j],
                        config={"displayModeBar": False},
                        style={"height": "400px"},
                    ),
                    style={
                        "width": "calc(33% - 20px)",
                        "margin": "10px",
                        "padding": "0px",
                    },
                )
                for j in range(
                    i, min(i + 3, len(plot_figures))
                )  # Add up to 3 plots per row
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between",
            },  # Flexbox for row layout
        )
        result_rows.append(row)

    static_2d = next(
        (r for r in results if getattr(r, "name", "") == "numeric 2d"), None
    )
    moving_2d = next(
        (r for r in results if getattr(r, "name", "") == "numeric 2d nonlinear"),
        None,
    )
    both_2d_visible = static_2d is not None and moving_2d is not None
    has_2d_result = static_2d is not None or moving_2d is not None
    shared_z_range = get_pressure_2d_z_range(
        bearing, [r for r in [static_2d, moving_2d] if r is not None]
    )

    pressure_2d_static_fig = (
        plot_pressure_2d(
            bearing,
            static_2d,
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
            show_colorbar=True,
            z_range_mpa=shared_z_range,
        )
        if moving_2d is not None
        else empty_figure()
    )

    pressure_row_style = (
        {"display": "flex", "alignItems": "center", "justifyContent": "center"}
        if has_2d_result
        else {"display": "none"}
    )
    # if pressure_2d_fig is not None:
    pressure_2d_row = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Slider(
                                id="pressure-2d-slider",
                                min=0,
                                max=0,
                                step=1,
                                value=0,
                                vertical=True,
                                verticalHeight=420,
                                updatemode="drag",
                                disabled=True,
                                marks={},
                                tooltip={"always_visible": False, "placement": "left"},
                            ),
                        ],
                        style={
                            "width": "100px",
                            "flex": "0 0 100px",
                            "overflow": "visible",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Graph(
                                id="pressure-2d-plot-static",
                                figure=pressure_2d_static_fig,
                                config={"displayModeBar": False, "scrollZoom": False},
                                style={"height": "550px"},
                            ),
                        ],
                        style={"flex": "1 1 0", "minWidth": 0},
                    ),
                    html.Div(
                        [
                            dcc.Graph(
                                id="pressure-2d-plot-moving",
                                figure=pressure_2d_moving_fig,
                                config={"displayModeBar": False, "scrollZoom": False},
                                style={"height": "550px"},
                            ),
                        ],
                        style={"flex": "1 1 0", "minWidth": 0},
                    ),
                ],
                style={
                    "width": "100%",
                    "margin": "10px",
                    "padding": "0px",
                    "display": "flex",
                    "flexDirection": "row",
                    "alignItems": "flex-end",
                    "gap": "16px",
                },
            ),
        ],
        id="pressure-2d-row",
        style=pressure_row_style,
    )

    header_legend_fig = plot_legend_only(results)

    results_header = html.Div(
        [
            html.H3("Results", style={"margin": "0"}),
        ],
        style={"display": "flex", "flexDirection": "column", "gap": "4px"},
    )
    legend_row = html.Div(
        [
            dcc.Graph(
                id="results-legend-only",
                figure=header_legend_fig,
                config={"displayModeBar": False, "staticPlot": True},
                style={"height": "70px", "width": "100%", "marginTop": "4px"},
            ),
        ],
        style={"display": "flex", "flexDirection": "column", "gap": "4px"},
    )
    return html.Div(
        [
            html.Div(
                [
                    html.H3(
                        "Bearing shape",
                        style={"margin": "0 0 20px 0", "textAlign": "left"},
                    ),
                    html.Div(shape_rows, id="shape-plots-container"),
                ],
                style=STYLES["plot_box"],
            ),
            html.Div(style={"height": "20px"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.Button(
                                "solve",
                                id="solve-results-btn",
                                title="Solve selected results",
                                style={
                                    "padding": "8px 12px",
                                    "fontSize": "18px",
                                    "backgroundColor": "#f8f9fa",
                                    "border": "1px solid #ddd",
                                    "borderRadius": "4px",
                                    "cursor": "pointer",
                                    "height": "34px",
                                    "width": "80px",
                                },
                            ),
                            html.Button(
                                "cancel",
                                id="cancel-solve-btn",
                                title="Cancel running solve",
                                disabled=True,
                                style={
                                    "padding": "8px 12px",
                                    "fontSize": "16px",
                                    "backgroundColor": "#fff5f5",
                                    "border": "1px solid #f0b6b6",
                                    "borderRadius": "4px",
                                    "cursor": "pointer",
                                    "height": "34px",
                                    "width": "90px",
                                },
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        id="solve-progress-bar-fill",
                                        style={
                                            "height": "10px",
                                            "width": "0%",
                                            "backgroundColor": "#4caf50",
                                        },
                                    )
                                ],
                                style={"flex": "1 1 auto", "minWidth": "180px"},
                            ),
                            html.Div(
                                "Idle (click solve to compute results)",
                                id="solve-progress-text",
                                style={
                                    "fontSize": "12px",
                                    "color": "#333",
                                    "marginTop": "6px",
                                },
                            ),
                            dcc.Checklist(
                                id="auto-solve-finish-typing",
                                options=[{"label": "auto solve", "value": "on"}],
                                value=[],
                                style={"marginTop": "6px"},
                                inputStyle={"marginRight": "6px"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "flexDirection": "row",
                            "alignItems": "center",
                            "gap": "16px",
                        },
                    )
                ],
                style=STYLES["plot_box"],
            ),
            html.Div(style={"height": "20px"}),
            html.Div(
                [
                    results_header,
                    legend_row,
                    html.Div(result_rows, id="result-plots-container"),
                    pressure_2d_row,
                ],
                style=STYLES["plot_box"],
            ),
        ],
        style=STYLES["plot_column"],
    )
