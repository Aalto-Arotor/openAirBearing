from copy import deepcopy

import numpy as np
import pytest
from plotly.graph_objects import Figure
from skfem import Basis, MeshQuad
from skfem.element import ElementQuad1

from openairbearing.app.ui_plots import (
    plot_bearing_shape,
    plot_geom_error,
    plot_key_results,
    plot_legend_only,
    plot_moment,
    plot_pressure_2d,
    plot_shear_force,
)
from openairbearing.bearings import (
    AnnularBearing,
    CircularBearing,
    InfiniteLinearBearing,
    JournalBearing,
    RectangularBearing,
)
from openairbearing.solution_analytic import solve_bearing_analytic
from openairbearing.solution_fem import (
    solve_bearing_fem_2d,
    solve_bearing_fem_2d_nonlinear,
)


@pytest.mark.parametrize(
    "bearing_cls",
    [
        CircularBearing,
        AnnularBearing,
        InfiniteLinearBearing,
        RectangularBearing,
        JournalBearing,
    ],
)
def test_plot_bearing_shape_returns_figures(bearing_cls):
    bearing = bearing_cls(nx=16, ny=10, nh=6)
    figures = plot_bearing_shape(bearing)

    assert isinstance(figures, list)
    assert len(figures) >= 2
    assert all(isinstance(fig, Figure) for fig in figures)


@pytest.mark.parametrize(
    "bearing_cls,expected_count",
    [
        (CircularBearing, 5),
        (AnnularBearing, 6),
        (InfiniteLinearBearing, 6),
    ],
)
def test_plot_key_results_counts_by_bearing_type(bearing_cls, expected_count):
    bearing = bearing_cls(nx=16, nh=8)
    result = solve_bearing_analytic(bearing)
    figures = plot_key_results(bearing, [result], legend=False)

    assert isinstance(figures, list)
    assert len(figures) == expected_count
    assert all(isinstance(fig, Figure) for fig in figures)


def test_plot_key_results_rectangular_with_numeric_2d():
    bearing = RectangularBearing(nx=12, ny=8, nh=6)
    result = solve_bearing_fem_2d(bearing)
    figures = plot_key_results(bearing, [result], legend=False)

    assert len(figures) == 6
    assert all(isinstance(fig, Figure) for fig in figures)


def test_plot_key_results_rectangular_with_numeric_2d_full():
    bearing = RectangularBearing(nx=12, ny=8, nh=6, u=[8.0, 0.0])
    result = solve_bearing_fem_2d_nonlinear(
        bearing,
        max_iter=20,
        tol=1e-5,
        relaxation=0.7,
    )
    figures = plot_key_results(bearing, [result], legend=False)

    assert len(figures) == 6
    assert all(isinstance(fig, Figure) for fig in figures)


def test_plot_legend_only_collapses_duplicate_solver_names():
    bearing = CircularBearing(nx=14, nh=6)
    result = solve_bearing_analytic(bearing)

    fig = plot_legend_only([result, result])
    assert isinstance(fig, Figure)
    assert len(fig.data) == 1


def test_plot_moment_near_zero_uses_default_y_range():
    bearing = CircularBearing(nx=14, nh=6)
    result = solve_bearing_analytic(bearing)
    result = deepcopy(result)
    result.moment = np.zeros((bearing.nh, 2))

    fig = plot_moment(bearing, [result], legend=False)
    assert isinstance(fig, Figure)
    assert list(fig.layout.yaxis.range) == [0.0, 0.1]


def test_plot_shear_force_large_values_keeps_autorange():
    bearing = CircularBearing(nx=14, nh=6)
    result = solve_bearing_analytic(bearing)
    result = deepcopy(result)
    shear_mag = np.linspace(0.0, 1.0, bearing.nh)
    result.shear_force = np.column_stack([shear_mag, np.zeros_like(shear_mag)])

    fig = plot_shear_force(bearing, [result], legend=False)
    assert isinstance(fig, Figure)
    assert list(fig.layout.yaxis.range) == [None, None]


def _make_ui_bearing_2d_stub(*, basis2d, geom_2d=None):
    class Bearing2DStub:
        pass

    class Fem2DStub:
        pass

    bearing = Bearing2DStub()
    bearing.fem_2d = Fem2DStub()
    bearing.fem_2d.basis = basis2d
    bearing.geom_2d = geom_2d
    bearing.pa = 1.0e5
    bearing.ps = 1.3e5
    bearing.pc = 1.0e5
    bearing.ha = np.array([4e-6, 6e-6, 8e-6])
    bearing.case = "rectangular"
    bearing.xa = 1.0e-3
    bearing.ya = 1.0e-3
    return bearing


def _make_ui_result_2d_stub(p_2d):
    class Result2DStub:
        pass

    result = Result2DStub()
    result.name = "numeric 2d"
    result.p_2d = list(p_2d)
    result.k = np.array([1.0, 5.0, 2.0])
    return result


def test_plot_pressure_2d_supports_quad_mesh_basis_ui():
    mesh = MeshQuad.init_tensor(
        np.linspace(0.0, 1.0e-3, 4), np.linspace(0.0, 1.0e-3, 3)
    )
    basis = Basis(mesh, ElementQuad1())
    p0 = np.full(basis.N, 1.10e5)
    p1 = np.full(basis.N, 1.20e5)
    p2 = np.full(basis.N, 1.15e5)
    bearing = _make_ui_bearing_2d_stub(basis2d=basis)
    result = _make_ui_result_2d_stub([p0, p1, p2])

    fig = plot_pressure_2d(bearing, [result], slider=False, include_frames=False)

    assert isinstance(fig, Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "mesh3d"


def test_plot_geom_error_supports_quad_mesh_basis_ui():
    mesh = MeshQuad.init_tensor(
        np.linspace(0.0, 1.0e-3, 4), np.linspace(0.0, 1.0e-3, 3)
    )
    basis = Basis(mesh, ElementQuad1())
    geom = np.linspace(0.0, 1.0e-6, basis.N)
    bearing = _make_ui_bearing_2d_stub(basis2d=basis, geom_2d=geom)

    fig = plot_geom_error(bearing)

    assert isinstance(fig, Figure)
    assert any(trace.type == "mesh3d" for trace in fig.data)
