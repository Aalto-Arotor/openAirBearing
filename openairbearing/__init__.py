"""OpenAir - Air Bearing Analysis Tool.

This package provides tools for analyzing and visualizing air bearing performance.
"""

# Version information
__version__ = "0.2.1"

# Import specific bearing classes
# Import app factory for direct package-level access
from .app.app import build_app
from .bearings import (
    AnnularBearing,
    CircularBearing,
    InfiniteLinearBearing,
    JournalBearing,
    RectangularBearing,
)
from .mesh import OABMeshQuad1, OABMeshTri1

# Import visualization functions
from .plots import (
    assign_result_styles,
    plot_ambient_flow_rate,
    plot_bearing_shape,
    plot_chamber_flow_rate,
    plot_geometry_error,
    plot_key_results,
    plot_legend_only,
    plot_load_capacity,
    plot_moment,
    plot_pressure_1d,
    plot_pressure_2d,
    plot_shear_force,
    plot_stiffness,
    plot_subplots,
    plot_supply_flow_rate,
    plot_xy_shape,
    plot_xz_shape,
)

# Import solver functions
from .solution_analytic import solve_bearing_analytic
from .solution_fem import (
    solve_bearing_fem_1d,
    solve_bearing_fem_2d,
    solve_bearing_fem_2d_nonlinear,
)

# Import utility functions
from .utils import (
    Result,
    get_area,
    get_beta,
    get_geom_1d,
    get_geom_2d,
    get_kappa,
    get_Qsc,
)

# Define what should be available when using 'from openairbearing import *'
__all__ = [
    # App
    "app",
    "build_app",
    # Version
    "__version__",
    # Bearing types
    "RectangularBearing",
    "CircularBearing",
    "AnnularBearing",
    "InfiniteLinearBearing",
    "JournalBearing",
    # Bearing parameters
    "get_kappa",
    "get_Qsc",
    "get_beta",
    "get_geom_1d",
    "get_geom_2d",
    "get_area",
    # Result type
    "Result",
    # Mesh
    "OABMeshQuad1",
    "OABMeshTri1",
    # Solver
    "solve_bearing_analytic",
    "solve_bearing_fem_1d",
    "solve_bearing_fem_2d",
    "solve_bearing_fem_2d_nonlinear",
    # Visualization
    "assign_result_styles",
    "plot_subplots",
    "plot_bearing_shape",
    "plot_key_results",
    "plot_legend_only",
    "plot_load_capacity",
    "plot_stiffness",
    "plot_moment",
    "plot_shear_force",
    "plot_pressure_1d",
    "plot_pressure_2d",
    "plot_geometry_error",
    "plot_supply_flow_rate",
    "plot_chamber_flow_rate",
    "plot_ambient_flow_rate",
    "plot_xy_shape",
    "plot_xz_shape",
]


def __getattr__(name):
    if name == "app":
        from .app.app import app as app_instance

        return app_instance
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
