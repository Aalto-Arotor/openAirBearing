"""Declarative form schema for the Dash input panel."""

from openairbearing.app.ui_config import (
    CASE_OPTIONS,
    get_default_solvers,
    get_solver_options,
)

SETUP_CONTROL_FIELDS = [
    {
        "label": "Simulated Case",
        "component": "dropdown",
        "props": {
            "id": "case-select",
            "options": CASE_OPTIONS,
            "value": "circular",
            "style": {"width": "150px"},
        },
        "spacer": True,
    },
    {
        "label": "Solution selection",
        "component": "checklist",
        "props": {
            "id": "solver-select",
            "options": get_solver_options("circular"),
            "value": get_default_solvers("circular"),
            "style": {"color": "black", "width": "150px"},
        },
        "spacer": False,
    },
]

NUMERICAL_CONTROL_FIELDS = [
    {
        "label": "Geometrical error type",
        "component": "dropdown",
        "props": {
            "id": "error-select",
            "options": [
                {"label": "Linear", "value": "linear"},
                {"label": "Quadratic", "value": "quadratic"},
                {"label": "Saddle", "value": "saddle"},
                {"label": "Tilt x", "value": "tiltx"},
                {"label": "Tilt y", "value": "tilty"},
            ],
            "value": "linear",
            "style": {"width": "150px"},
        },
        "spacer": True,
    }
]

BEARING_PARAMETER_FIELDS = [
    {
        "label": "Porous Layer Thickness (mm)",
        "input_id": "hp-input",
        "reset_id": "hp-reset",
        "value_key": "hp",
        "min_value": 0.01,
    },
    {
        "label": "Outer radius / x length (mm)",
        "input_id": "xa-input",
        "reset_id": "xa-reset",
        "value_key": "xa",
        "min_value": 0.01,
    },
]

INNER_RADIUS_FIELDS = [
    {
        "label": "Inner radius (mm)",
        "input_id": "xc-input",
        "reset_id": "xc-reset",
        "value_key": "xc",
        "min_value": 0.01,
    }
]

LENGTH_FIELDS = [
    {
        "label": "y length (mm)",
        "input_id": "ya-input",
        "reset_id": "ya-reset",
        "value_key": "ya",
        "min_value": 0.01,
    }
]

FLOW_PARAMETER_FIELDS = [
    {
        "label": "Permeability (m^2)",
        "input_id": "kappa-input",
        "reset_id": "kappa-reset",
        "value_key": "kappa",
        "min_value": 0,
        "step": 1e-17,
        "input_mode": "numeric",
    },
    {
        "label": "Free flow (l/min)",
        "input_id": "Qsc-input",
        "reset_id": "Qsc-reset",
        "value_key": "Qsc",
        "min_value": 0.01,
        "step": 0.01,
        "input_mode": "numeric",
    },
]

JOURNAL_PARAMETER_FIELDS = [
    {
        "label": "Bore diameter (mm)",
        "input_id": "bore-diameter-input",
        "reset_id": "bore-diameter-reset",
        "value_key": "bore_diameter",
        "min_value": 0.01,
        "step": 0.001,
    },
    {
        "label": "Shaft diameter (mm)",
        "input_id": "shaft-diameter-input",
        "reset_id": "shaft-diameter-reset",
        "value_key": "shaft_diameter",
        "min_value": 0.01,
        "step": 0.001,
    },
    {
        "label": "Clearance (μm)",
        "input_id": "clearance-input",
        "value_key": "clearance",
        "step": 0.1,
        "disabled": True,
    },
]

NUMERICAL_PARAMETER_FIELDS = [
    {
        "label": "Geometry error (μm)",
        "input_id": "error-input",
        "reset_id": "error-reset",
        "value_key": "error",
        "step": 0.5,
    },
    {
        "label": "Slip coefficient Φ",
        "input_id": "psi-input",
        "reset_id": "psi-reset",
        "value_key": "psi",
        "min_value": 0,
        "step": 0.01,
    },
    {
        "label": "x velocity (m/s)",
        "input_id": "ux-input",
        "reset_id": "ux-reset",
        "value_key": "ux",
        "min_value": 0,
        "step": 0.01,
    },
    {
        "label": "y velocity (m/s)",
        "input_id": "uy-input",
        "reset_id": "uy-reset",
        "value_key": "uy",
        "min_value": 0,
        "step": 0.01,
    },
]

LOAD_PARAMETER_FIELDS = [
    {
        "label": "Ambient Pressure (MPa)",
        "input_id": "pa-input",
        "reset_id": "pa-reset",
        "value_key": "pa",
        "min_value": 1e-6,
        "step": 0.01,
        "input_mode": "numeric",
    },
    {
        "label": "Supply Pressure (MPa)",
        "input_id": "ps-input",
        "reset_id": "ps-reset",
        "value_key": "ps",
        "min_value": 0.1,
        "step": 0.01,
        "input_mode": "numeric",
    },
]

CHAMBER_PARAMETER_FIELDS = [
    {
        "label": "Chamber Pressure (MPa)",
        "input_id": "pc-input",
        "reset_id": "pc-reset",
        "value_key": "pc",
        "min_value": 0,
        "step": 0.01,
        "input_mode": "numeric",
    }
]

FLUID_PARAMETER_FIELDS = [
    {
        "label": "Dynamic Viscosity (Pa·s)",
        "input_id": "mu-input",
        "reset_id": "mu-reset",
        "value_key": "mu",
    },
]

MODEL_PARAMETER_FIELDS = [
    {
        "label": "Minimum Height (μm)",
        "label_id": "ha-min-label",
        "input_id": "ha-min-input",
        "reset_id": "ha-min-reset",
        "value_key": "ha_min",
        "min_value": 0,
        "step": 0.5,
    },
    {
        "label": "Maximum Height (μm)",
        "label_id": "ha-max-label",
        "input_id": "ha-max-input",
        "reset_id": "ha-max-reset",
        "value_key": "ha_max",
    },
    {
        "label": "Height points",
        "label_id": "nh-label",
        "input_id": "nh-input",
        "reset_id": "nh-reset",
        "value_key": "nh",
        "min_value": 3,
        "max_value": None,
        "step": 1,
    },
    {
        "label": "x direction points",
        "input_id": "nx-input",
        "reset_id": "nx-reset",
        "value_key": "nx",
        "min_value": 3,
        "max_value": None,
        "step": 1,
    },
]

NY_PARAMETER_FIELDS = [
    {
        "label": "y direction points",
        "input_id": "ny-input",
        "reset_id": "ny-reset",
        "value_key": "ny",
        "min_value": 3,
        "max_value": None,
        "step": 1,
    }
]

FEM_PARAMETER_FIELDS = [
    {
        "label": "Mesh divisions",
        "input_id": "divs-input",
        "reset_id": "divs-reset",
        "value_key": "divs",
        "min_value": 1,
        "max_value": None,
        "step": 1,
    }
]
