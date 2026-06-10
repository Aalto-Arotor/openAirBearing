"""Configuration and constants for bearing analysis UI.

Defines solver options, default settings, and styling configuration.
"""

from openairbearing.bearings import (
    AnnularBearing,
    CircularBearing,
    InfiniteLinearBearing,
    JournalBearing,
    RectangularBearing,
)

CASE_OPTIONS = [
    {"label": "Circular thrust", "value": "circular"},
    {"label": "Annular thrust", "value": "annular"},
    {"label": "Infinitely long", "value": "infinite"},
    {"label": "Rectangular", "value": "rectangular"},
    {"label": "Journal", "value": "journal"},
]

_CASE_SOLVER_META = {
    "analytic": {"label": "Analytic"},
    "numeric1d": {"label": "Numeric 1d"},
    "numeric2d": {"label": "Numeric 2d"},
    "numeric2dfull": {"label": "Numeric 2d full"},
}

SOLVER_EXECUTION_ORDER = ("analytic", "numeric1d", "numeric2d", "numeric2dfull")

_CASE_SOLVER_SUPPORT = {
    "circular": {
        "analytic": True,
        "numeric1d": True,
        "numeric2d": True,
        "numeric2dfull": True,
    },
    "annular": {
        "analytic": True,
        "numeric1d": True,
        "numeric2d": True,
        "numeric2dfull": True,
    },
    "infinite": {
        "analytic": True,
        "numeric1d": True,
        "numeric2d": True,
        "numeric2dfull": False,
    },
    "rectangular": {
        "analytic": False,
        "numeric1d": False,
        "numeric2d": True,
        "numeric2dfull": True,
    },
    "journal": {
        "analytic": False,
        "numeric1d": False,
        "numeric2d": True,
        "numeric2dfull": True,
    },
}

CASE_DEFAULT_SOLVERS = {
    "circular": ["analytic"],
    "annular": ["analytic"],
    "infinite": ["analytic"],
    "rectangular": ["numeric2d"],
    "journal": ["numeric2d"],
}

CASE_ERROR_OPTIONS = {
    "circular": [
        {"label": "Linear", "value": "linear"},
        {"label": "Quadratic", "value": "quadratic"},
        {"label": "Saddle", "value": "saddle"},
        {"label": "Tilt x", "value": "tiltx"},
        {"label": "Tilt y", "value": "tilty"},
    ],
    "annular": [
        {"label": "Linear", "value": "linear"},
        {"label": "Quadratic", "value": "quadratic"},
        {"label": "Saddle", "value": "saddle"},
        {"label": "Tilt x", "value": "tiltx"},
        {"label": "Tilt y", "value": "tilty"},
    ],
    "infinite": [
        {"label": "Linear", "value": "linear"},
        {"label": "Quadratic", "value": "quadratic"},
    ],
    "rectangular": [
        {"label": "Linear", "value": "linear"},
        {"label": "Quadratic", "value": "quadratic"},
        {"label": "Saddle", "value": "saddle"},
        {"label": "Tilt x", "value": "tiltx"},
        {"label": "Tilt y", "value": "tilty"},
    ],
    "journal": [
        {"label": "None", "value": "none"},
        {"label": "Conicity", "value": "conicity"},
        {"label": "Misalignment", "value": "misalignment"},
    ],
}

CASE_DEFAULT_ERROR = {
    "circular": "linear",
    "annular": "linear",
    "infinite": "linear",
    "rectangular": "linear",
    "journal": "none",
}


def get_bearing(case):
    """Return default bearing class for a given case.

    Args:
        case: Bearing case string
        ('circular', 'annular', 'infinite', 'rectangular', 'journal').

    Returns:
        type: Bearing class corresponding to the case.

    Raises:
        TypeError: If case is not recognized.
    """
    bearing_map = {
        "circular": CircularBearing,
        "annular": AnnularBearing,
        "infinite": InfiniteLinearBearing,
        "rectangular": RectangularBearing,
        "journal": JournalBearing,
    }
    try:
        return bearing_map[case]
    except KeyError as exc:
        raise TypeError("no default bearing defined") from exc


BASE_TOGGLE_STYLE = {
    "grid-template-columns": "200px 100px 20px",
    "gap": "20px",
    "marginTop": "20px",
    "marginBottom": "20px",
    "align-items": "center",
}

CASE_CONTAINER_VISIBILITY = {
    "circular": {"pc": True, "xc": False, "ya": False, "ny": False, "divs": True},
    "annular": {"pc": True, "xc": True, "ya": False, "ny": True, "divs": False},
    "infinite": {"pc": True, "xc": False, "ya": False, "ny": False, "divs": False},
    "rectangular": {"pc": True, "xc": False, "ya": True, "ny": True, "divs": False},
    "journal": {
        "pc": True,
        "xc": False,
        "ya": True,
        "ny": True,
        "divs": False,
        "journal": True,
    },
}


def get_solver_options(case):
    """Return checklist options for solvers, including per-case disabled flags."""
    support = _CASE_SOLVER_SUPPORT.get(case, {})
    options = []
    for solver_key, meta in _CASE_SOLVER_META.items():
        options.append(
            {
                "label": meta["label"],
                "value": solver_key,
                "disabled": not support.get(solver_key, False),
            }
        )
    return options


def get_default_solvers(case):
    """Return default solver values for a given case."""
    return CASE_DEFAULT_SOLVERS.get(case, [])


def get_error_options(case):
    """Return geometry error options for a given bearing case."""
    return CASE_ERROR_OPTIONS.get(case, CASE_ERROR_OPTIONS["rectangular"])


def get_default_error(case):
    """Return the default geometry error type for a given bearing case."""
    return CASE_DEFAULT_ERROR.get(case, "linear")


def get_container_styles(case):
    """Return visibility styles for optional input containers by case."""
    visibility = CASE_CONTAINER_VISIBILITY.get(
        case,
        {
            "pc": False,
            "xc": False,
            "ya": False,
            "ny": False,
            "divs": False,
            "journal": False,
        },
    )

    def _style(show):
        return {**BASE_TOGGLE_STYLE, "display": "grid" if show else "none"}

    return (
        _style(visibility["pc"]),
        _style(visibility["xc"]),
        _style(visibility["ya"]),
        _style(visibility["ny"]),
        _style(visibility["divs"]),
        _style(visibility.get("journal", False)),
    )
