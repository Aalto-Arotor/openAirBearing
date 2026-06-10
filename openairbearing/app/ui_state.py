"""State conversion utilities between Bearing objects and form field dictionaries.

Handles bidirectional mapping between UI form state and bearing configuration objects.
"""

import numpy as np

FORM_FIELD_ORDER = [
    "mu",
    "hp",
    "xa",
    "xc",
    "ya",
    "bore_diameter",
    "shaft_diameter",
    "clearance",
    "kappa",
    "Qsc",
    "pa",
    "pc",
    "ps",
    "ha_min",
    "ha_max",
    "nx",
    "ny",
    "nh",
    "divs",
    "error",
    "psi",
    "ux",
    "uy",
]


def bearing_to_form_values(bearing):
    """Convert bearing SI attributes to UI form units and field names."""
    bore_diameter = getattr(bearing, "bore_diameter", None)
    shaft_diameter = getattr(bearing, "shaft_diameter", None)
    clearance = getattr(bearing, "clearance", None)
    return {
        "mu": bearing.mu,
        "hp": bearing.hp * 1e3,
        "xa": bearing.xa * 1e3,
        "xc": bearing.xc * 1e3,
        "ya": bearing.ya * 1e3,
        "bore_diameter": bore_diameter * 1e3 if bore_diameter is not None else None,
        "shaft_diameter": shaft_diameter * 1e3 if shaft_diameter is not None else None,
        "clearance": clearance * 1e6 if clearance is not None else None,
        "kappa": bearing.kappa,
        "Qsc": bearing.Qsc,
        "pa": bearing.pa * 1e-6,
        "pc": bearing.pc * 1e-6,
        "ps": bearing.ps * 1e-6,
        "ha_min": bearing.ha_min * 1e6,
        "ha_max": bearing.ha_max * 1e6,
        "nx": bearing.nx,
        "ny": bearing.ny,
        "nh": bearing.nh,
        "divs": getattr(bearing, "divs", 3),
        "error": bearing.error * 1e6,
        "psi": bearing.Psi,
        "ux": float(np.asarray(bearing.u).ravel()[0])
        if np.asarray(bearing.u).size
        else 0.0,
        "uy": float(np.asarray(bearing.u).ravel()[1])
        if np.asarray(bearing.u).size > 1
        else 0.0,
    }


def form_to_bearing_kwargs(form_values, case=None):
    """Convert form values to bearing constructor kwargs in SI units."""
    kwargs = {
        "pa": form_values["pa"] * 1e6,
        "ps": form_values["ps"] * 1e6,
        "pc": form_values["pc"] * 1e6,
        "mu": form_values["mu"],
        "hp": form_values["hp"] * 1e-3,
        "xa": form_values["xa"] * 1e-3,
        "xc": form_values["xc"] * 1e-3,
        "ya": form_values["ya"] * 1e-3,
        "nx": int(form_values["nx"]),
        "ny": int(form_values["ny"]),
        "ha_min": form_values["ha_min"] * 1e-6,
        "ha_max": form_values["ha_max"] * 1e-6,
        "nh": int(form_values["nh"]),
        "error_type": form_values["error_type"],
        "error": form_values["error"] * 1e-6,
        "Psi": form_values["psi"],
        "u": np.array(
            [
                0.0 if form_values.get("ux") is None else form_values["ux"],
                0.0 if form_values.get("uy") is None else form_values["uy"],
            ],
            dtype=float,
        ),
    }
    if case == "journal":
        bore_diameter = form_values.get("bore_diameter")
        shaft_diameter = form_values.get("shaft_diameter")
        kwargs.update(
            {
                "bore_diameter": (
                    None if bore_diameter is None else bore_diameter * 1e-3
                ),
                "shaft_diameter": (
                    None if shaft_diameter is None else shaft_diameter * 1e-3
                ),
                "clearance": (
                    None
                    if bore_diameter is None or shaft_diameter is None
                    else 0.5 * (bore_diameter - shaft_diameter) * 1e-3
                ),
            }
        )
    if case == "circular":
        kwargs["divs"] = int(form_values.get("divs") or 3)
    return kwargs
