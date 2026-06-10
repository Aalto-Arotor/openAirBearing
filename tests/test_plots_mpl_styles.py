import matplotlib.pyplot as plt
import numpy as np
import pytest
from skfem import Basis, MeshQuad, MeshTri
from skfem.element import ElementQuad1, ElementTriP1

import openairbearing as ab
from openairbearing.plots import (
    _solver_highlight_marker,
    assign_result_styles,
    plot_geometry_error,
    plot_legend_only,
    plot_load_capacity,
    plot_pressure_1d,
    plot_pressure_2d,
)
from openairbearing.utils import Result


def _make_result(*, name="solver"):
    nh = 3
    nx = 4
    return Result(
        name=name,
        p=np.full((nx, nh), 1.1e5),
        w=np.array([1.0, 2.0, 3.0]),
        k=np.array([10.0, 20.0, 30.0]),
        qs=np.array([0.1, 0.2, 0.3]),
        qa=np.array([0.1, 0.2, 0.3]),
        qc=np.array([0.1, 0.2, 0.3]),
    )


def _make_bearing_stub():
    class BearingStub:
        pass

    bearing = BearingStub()
    bearing.ha = np.array([4e-6, 6e-6, 8e-6])
    bearing.x = np.linspace(0.0, 1.0e-3, 4)
    bearing.pa = 1.0e5
    return bearing


def _make_bearing_2d_stub(*, basis2d, geom_2d=None):
    class Bearing2DStub:
        pass

    class Fem2DStub:
        pass

    bearing = Bearing2DStub()
    bearing.fem_2d = Fem2DStub()
    bearing.fem_2d.basis = basis2d
    bearing.pa = 1.0e5
    bearing.geom_2d = geom_2d
    return bearing


def _make_result_2d_stub(p_2d):
    return Result(
        name="numeric 2d",
        p=None,
        p_1d=None,
        p_2d=list(p_2d),
        w=np.array([1.0, 2.0, 3.0]),
        k=np.array([1.0, 5.0, 2.0]),
        qs=np.array([0.1, 0.2, 0.3]),
        qa=np.array([0.1, 0.2, 0.3]),
        qc=np.array([0.0, 0.0, 0.0]),
    )


def test_package_exports_use_assign_result_styles_only():
    assert hasattr(ab, "assign_result_styles")
    assert not hasattr(ab, "assign_colors")


def test_assign_result_styles_assigns_and_preserves_fields():
    r1 = _make_result(name="a")
    r2 = _make_result(name="b")
    r2.color = "black"
    r2.linestyle = "--"
    r2.marker = "s"

    assign_result_styles(
        [r1, r2],
        colors=["red", "blue"],
        styles=[":", "-"],
        markers=["^", "o"],
    )

    assert r1.color == "red"
    assert r1.linestyle == ":"
    assert r1.marker == "^"
    assert r2.color == "black"
    assert r2.linestyle == "--"
    assert r2.marker == "s"


def test_plot_load_capacity_uses_linestyle_and_marker():
    bearing = _make_bearing_stub()
    result = _make_result()
    result.color = "magenta"
    result.linestyle = "--"
    result.marker = "x"

    fig = plot_load_capacity(bearing, [result], legend=False)
    line = fig.axes[0].lines[0]

    assert line.get_color() == "magenta"
    assert line.get_linestyle() == "--"
    assert line.get_marker() == "x"


def test_plot_load_capacity_title_default_custom_and_none():
    bearing = _make_bearing_stub()
    result = _make_result()

    fig_default = plot_load_capacity(bearing, [result], legend=False)
    assert fig_default.axes[0].get_title() == "Load capacity"

    fig_custom = plot_load_capacity(bearing, [result], legend=False, title="My title")
    assert fig_custom.axes[0].get_title() == "My title"

    fig_none = plot_load_capacity(bearing, [result], legend=False, title=None)
    assert fig_none.axes[0].get_title() == ""


def test_plot_pressure_1d_uses_linestyle_and_marker():
    bearing = _make_bearing_stub()
    result = _make_result()
    result.linestyle = ":"
    result.marker = "d"

    fig = plot_pressure_1d(bearing, [result], legend=False)
    line = fig.axes[0].lines[0]

    assert line.get_linestyle() == ":"
    assert line.get_marker() == "d"


def test_plot_pressure_1d_title_none():
    bearing = _make_bearing_stub()
    result = _make_result()

    fig = plot_pressure_1d(bearing, [result], legend=False, title=None)
    assert fig.axes[0].get_title() == ""


def test_plot_pressure_1d_defaults_to_max_stiffness_index():
    bearing = _make_bearing_stub()
    result = _make_result()
    result.k = np.array([1.0, 5.0, 2.0])
    result.p = np.array(
        [
            [1.10e5, 1.20e5, 1.30e5],
            [1.10e5, 1.20e5, 1.30e5],
            [1.10e5, 1.20e5, 1.30e5],
            [1.10e5, 1.20e5, 1.30e5],
        ]
    )

    fig = plot_pressure_1d(bearing, [result], legend=False)
    y = fig.axes[0].lines[0].get_ydata()

    assert np.allclose(y, 0.02)


def test_plot_pressure_1d_accepts_explicit_pressure_index():
    bearing = _make_bearing_stub()
    result = _make_result()
    result.k = np.array([1.0, 5.0, 2.0])
    result.p = np.array(
        [
            [1.10e5, 1.20e5, 1.30e5],
            [1.10e5, 1.20e5, 1.30e5],
            [1.10e5, 1.20e5, 1.30e5],
            [1.10e5, 1.20e5, 1.30e5],
        ]
    )

    fig = plot_pressure_1d(bearing, [result], pressure_index=0, legend=False)
    y = fig.axes[0].lines[0].get_ydata()
    assert np.allclose(y, 0.01)

    with pytest.raises(IndexError):
        plot_pressure_1d(bearing, [result], pressure_index=10, legend=False)


def test_plot_legend_only_uses_result_styles():
    result = _make_result()
    result.color = "green"
    result.linestyle = "-."
    result.marker = "s"

    fig = plot_legend_only([result])
    legend = fig.axes[0].get_legend()
    handle = legend.legend_handles[0]

    assert handle.get_color() == "green"
    assert handle.get_linestyle() == "-."
    assert handle.get_marker() == "s"


def test_solver_highlight_marker_uses_auto_and_override():
    result = _make_result()
    assert _solver_highlight_marker(result, index=0) == "o"

    result.highlight_marker = "x"
    assert _solver_highlight_marker(result, index=0) == "x"


def test_plot_pressure_2d_supports_tri_mesh_basis():
    mesh = MeshTri.init_tensor(np.linspace(0.0, 1.0e-3, 4), np.linspace(0.0, 1.0e-3, 3))
    basis = Basis(mesh, ElementTriP1())
    p0 = np.full(basis.N, 1.10e5)
    p1 = np.full(basis.N, 1.20e5)
    p2 = np.full(basis.N, 1.15e5)
    bearing = _make_bearing_2d_stub(basis2d=basis)
    result = _make_result_2d_stub([p0, p1, p2])

    fig = plot_pressure_2d(bearing, [result], show_colorbar=True)

    assert fig.axes[0].get_xlabel() == "x (mm)"
    assert fig.axes[0].get_ylabel() == "y (mm)"
    assert fig.axes[0].get_title() == "Pressure distribution (numeric 2d)"
    assert len(fig.axes) >= 2


def test_plot_pressure_2d_supports_quad_mesh_basis():
    mesh = MeshQuad.init_tensor(
        np.linspace(0.0, 1.0e-3, 4), np.linspace(0.0, 1.0e-3, 3)
    )
    basis = Basis(mesh, ElementQuad1())
    p0 = np.full(basis.N, 1.10e5)
    p1 = np.full(basis.N, 1.20e5)
    p2 = np.full(basis.N, 1.15e5)
    bearing = _make_bearing_2d_stub(basis2d=basis)
    result = _make_result_2d_stub([p0, p1, p2])

    fig = plot_pressure_2d(bearing, [result], show_colorbar=False)

    assert fig.axes[0].get_xlabel() == "x (mm)"
    assert fig.axes[0].get_ylabel() == "y (mm)"
    assert fig.axes[0].get_title() == "Pressure distribution (numeric 2d)"


def test_plot_geometry_error_uses_skfem_plotting():
    mesh = MeshQuad.init_tensor(
        np.linspace(0.0, 1.0e-3, 4), np.linspace(0.0, 1.0e-3, 3)
    )
    basis = Basis(mesh, ElementQuad1())
    geom = np.linspace(0.0, 1.0e-6, basis.N)
    bearing = _make_bearing_2d_stub(basis2d=basis, geom_2d=geom)

    fig = plot_geometry_error(bearing)

    assert fig.axes[0].get_xlabel() == "x (mm)"
    assert fig.axes[0].get_ylabel() == "y (mm)"
    assert fig.axes[0].get_title() == "Mesh & Geometry"


def test_plot_geometry_error_accepts_3d_axis():
    mesh = MeshQuad.init_tensor(
        np.linspace(0.0, 1.0e-3, 4), np.linspace(0.0, 1.0e-3, 3)
    )
    basis = Basis(mesh, ElementQuad1())
    geom = np.linspace(0.0, 1.0e-6, basis.N)
    bearing = _make_bearing_2d_stub(basis2d=basis, geom_2d=geom)

    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection="3d")
    plot_geometry_error(bearing, ax=ax3d)
    fig.canvas.draw()

    assert fig.axes[0].name == "3d"
    assert fig.axes[0].get_xlabel() == "x (mm)"
