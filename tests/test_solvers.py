import numpy as np
import pytest
from skfem import asm

from openairbearing.bearings import (
    AnnularBearing,
    CircularBearing,
    InfiniteLinearBearing,
    JournalBearing,
    RectangularBearing,
)
from openairbearing.fem_utils import load_form_2d_axial
from openairbearing.solution_analytic import (
    get_pressure_analytic_annular,
    get_pressure_analytic_circular,
    get_pressure_analytic_infinite,
    solve_bearing_analytic,
)
from openairbearing.solution_fem import (
    solve_bearing_fem_1d,
    solve_bearing_fem_2d,
    solve_bearing_fem_2d_nonlinear,
)


def test_get_pressure_analytic_circular():
    """Test the analytic pressure distribution for a circular bearing."""
    bearing = CircularBearing()
    p = get_pressure_analytic_circular(bearing)
    assert p.shape == (bearing.nx, bearing.nh)
    assert np.all(p >= bearing.pa - 1e-6)
    assert np.all(p <= bearing.ps + 1e-6)


def test_user_kappa_is_preserved_and_affects_analytic_flow():
    low_kappa = 2e-15
    high_kappa = 2e-14

    low = CircularBearing(xa=10e-3, kappa=low_kappa, nh=12)
    high = CircularBearing(xa=10e-3, kappa=high_kappa, nh=12)

    assert low.kappa == pytest.approx(low_kappa)
    assert high.kappa == pytest.approx(high_kappa)

    low_result = solve_bearing_analytic(low)
    high_result = solve_bearing_analytic(high)

    assert np.mean(high_result.qa) > np.mean(low_result.qa)
    assert np.mean(high_result.qs) > np.mean(low_result.qs)
    assert not np.allclose(high_result.qa, low_result.qa)


# def test_oab_mesh_quad1_annulus():
#     """Test OABMeshQuad1 annulus mesh creation and boundaries."""
#     m = OABMeshQuad1.init_annulus(25e-3, 58e-3, ntheta=12, nr=6)
#     assert m.p.shape[0] == 2
#     assert m.t.shape[0] == 4  # 4 nodes per quad
#     assert "inner" in m.boundaries
#     assert "outer" in m.boundaries


# def test_annular_bearing_quad_mesh_solve():
#     """Test AnnularBearing with quad mesh solves successfully."""
#     bearing = AnnularBearing(mesh_type="quad", nx=10, ny=10, nh=4)
#     result = solve_bearing_fem_2d(bearing)
#     assert result.p.shape == (bearing.nx, bearing.nh)
#     assert np.all(np.isfinite(result.w))


def test_get_pressure_analytic_annular():
    """Test the analytic pressure distribution for a annular bearing."""
    bearing = AnnularBearing()
    p = get_pressure_analytic_annular(bearing)
    assert p.shape == (bearing.nx, bearing.nh)
    assert p[0, :] == pytest.approx(bearing.pc, rel=1e-6, abs=1e-6)
    assert p[-1, :] == pytest.approx(bearing.pa, rel=1e-6, abs=1e-6)
    assert np.all(p >= min(bearing.pa, bearing.pc) - 1e-6)
    assert np.all(p <= bearing.ps + 1e-6)


def test_annular_bearing_initializes_1d_fem_state():
    bearing = AnnularBearing(nx=20, nh=8)

    assert bearing.fem_1d is not None
    assert bearing.fem_1d.mesh is not None
    assert bearing.fem_1d.basis is not None
    assert bearing.fem_1d.D.size > 0
    assert bearing.fem_1d.p.shape[0] == bearing.fem_1d.basis.N

    chamber_dofs = bearing.fem_1d.basis.get_dofs("chamber")
    ambient_dofs = bearing.fem_1d.basis.get_dofs("ambient")
    assert chamber_dofs.all().size > 0
    assert ambient_dofs.all().size > 0


@pytest.mark.parametrize(
    ("bearing_cls", "has_chamber"),
    [
        (CircularBearing, False),
        (AnnularBearing, True),
        (InfiniteLinearBearing, True),
    ],
)
def test_bearings_initialize_1d_boundary_sets_from_labels(bearing_cls, has_chamber):
    bearing = bearing_cls(nx=20, nh=8)
    boundary_sets = bearing.fem_1d.boundary_sets

    assert boundary_sets is not None
    assert boundary_sets["ambient"] is not None
    assert boundary_sets["all"].size > 0
    assert np.array_equal(np.unique(bearing.fem_1d.D), boundary_sets["all"])

    ambient_ix = boundary_sets["ambient"].all()
    assert ambient_ix.size > 0
    assert np.allclose(bearing.fem_1d.p[ambient_ix], bearing.ps**2 - bearing.pa**2)

    if has_chamber:
        assert boundary_sets["chamber"] is not None
        chamber_ix = boundary_sets["chamber"].all()
        assert chamber_ix.size > 0
        assert np.allclose(bearing.fem_1d.p[chamber_ix], bearing.ps**2 - bearing.pc**2)
    else:
        assert boundary_sets["chamber"] is None


def test_rectangular_bearing_initializes_2d_fem_state():
    bearing = RectangularBearing(nx=12, ny=8, nh=4)

    assert bearing.fem_2d is not None
    assert bearing.fem_2d.mesh is not None
    assert bearing.fem_2d.basis is not None
    assert bearing.fem_2d.eta is not None
    assert bearing.fem_2d.p is not None
    assert bearing.fem_2d.D is not None
    assert bearing.fem_2d.boundary_sets is not None
    assert bearing.fem_2d.D.size > 0


def test_get_pressure_analytic_infinite():
    """Test the analytic pressure distribution for a infinite linear bearing."""
    bearing = InfiniteLinearBearing()
    p = get_pressure_analytic_infinite(bearing)
    assert p.shape == (bearing.nx, bearing.nh)
    assert np.all(p >= bearing.pa - 1e-6)
    assert np.all(p <= bearing.ps + 1e-6)


@pytest.mark.parametrize(
    "bearing_cls",
    [CircularBearing, AnnularBearing, InfiniteLinearBearing],
)
def test_solve_bearing_analytic_and_fem_1d_shapes_and_ranges(bearing_cls):
    bearing = bearing_cls(nx=20, nh=12)

    analytic = solve_bearing_analytic(bearing)
    numeric_1d = solve_bearing_fem_1d(bearing)

    assert analytic.p.shape == (bearing.nx, bearing.nh)
    assert numeric_1d.p_1d.shape[0] == bearing.nh
    assert analytic.w.shape == (bearing.nh,)
    assert numeric_1d.w.shape == (bearing.nh,)

    assert np.all(np.isfinite(analytic.w))
    assert np.all(np.isfinite(numeric_1d.w))
    assert np.all(analytic.p >= min(bearing.pa, bearing.pc) - 1e-6)
    assert np.all(analytic.p <= bearing.ps + 1e-6)
    assert np.all(numeric_1d.p_1d >= min(bearing.pa, bearing.pc) - 1e-6)
    assert np.all(numeric_1d.p_1d <= bearing.ps + 1e-6)


def test_solve_bearing_fem_2d_rectangular_result_contract():
    bearing = RectangularBearing(nx=12, ny=8, nh=6)
    result = solve_bearing_fem_2d(bearing)

    assert result.name == "numeric 2d"
    assert result.p is None
    assert result.w.shape == (bearing.nh,)
    assert result.k.shape == (bearing.nh,)
    assert result.qs.shape == (bearing.nh,)
    assert result.qa.shape == (bearing.nh,)
    assert result.qc.shape == (bearing.nh,)
    assert result.moment.shape == (bearing.nh, 2)
    assert result.shear_force.shape == (bearing.nh, 2)
    assert isinstance(result.p_2d, list)
    assert len(result.p_2d) == bearing.nh
    assert np.all(np.isfinite(result.w))
    assert np.all(np.isfinite(result.k))
    assert np.all(np.isfinite(result.moment))
    assert np.all(np.isfinite(result.shear_force))
    shear_norm = np.linalg.norm(result.shear_force, axis=1)
    shear_tol = 1e-12 * max(np.max(np.abs(result.w)), 1.0)
    assert np.max(shear_norm) <= shear_tol


def test_solve_bearing_fem_2d_journal_result_contract():
    bearing = JournalBearing(nx=12, ny=8, nh=5, pc=0.25e6 + 101325)
    result = solve_bearing_fem_2d(bearing)

    assert result.name == "numeric 2d"
    assert result.p is None
    assert result.w.shape == (bearing.nh,)
    assert result.k.shape == (bearing.nh,)
    assert result.qs.shape == (bearing.nh,)
    assert result.qa.shape == (bearing.nh,)
    assert result.qc.shape == (bearing.nh,)
    assert result.moment.shape == (bearing.nh, 2)
    assert result.shear_force.shape == (bearing.nh, 2)
    assert isinstance(result.p_2d, list)
    assert len(result.p_2d) == bearing.nh
    assert np.all(np.isfinite(result.w))
    assert np.all(np.isfinite(result.k))
    assert np.all(np.isfinite(result.qa))
    assert np.all(np.isfinite(result.qc))


def test_solve_bearing_fem_2d_journal_centered_has_near_zero_projected_load():
    bearing = JournalBearing(
        nx=16,
        ny=24,
        nh=5,
        eccentricity=0.0,
        eccentricity_sweep=np.zeros(5),
        u=np.zeros(2),
    )
    result = solve_bearing_fem_2d(bearing)

    basis = bearing.fem_2d.basis
    axial_load = []
    for p_dofs in result.p_2d:
        pfield = basis.interpolate(p_dofs)
        axial_load.append(
            asm(load_form_2d_axial(), basis, p=pfield, pa=bearing.pa).sum()
        )

    axial_load = np.asarray(axial_load)
    projected_to_axial = np.abs(result.w) / np.maximum(np.abs(axial_load), 1e-12)
    assert np.max(projected_to_axial) < 1e-4


@pytest.mark.parametrize(
    "bearing",
    [
        CircularBearing(nx=10, nh=4, u=np.array([8.0, 0.0])),
        AnnularBearing(nx=10, ny=10, nh=4, u=np.array([8.0, 0.0])),
        RectangularBearing(nx=10, ny=8, nh=4, u=np.array([8.0, 0.0])),
        JournalBearing(nx=10, ny=8, nh=4, u=np.array([8.0, 0.0])),
    ],
)
def test_solve_bearing_fem_2d_full_smoke_all_geometries(bearing):
    result = solve_bearing_fem_2d_nonlinear(
        bearing, max_iter=20, tol=1e-5, relaxation=0.7
    )

    assert result.name == "numeric 2d nonlinear"
    assert result.p is None
    assert result.w.shape == (bearing.nh,)
    assert result.k.shape == (bearing.nh,)
    assert result.qs.shape == (bearing.nh,)
    assert result.qa.shape == (bearing.nh,)
    assert result.qc.shape == (bearing.nh,)
    assert result.moment.shape == (bearing.nh, 2)
    assert result.shear_force.shape == (bearing.nh, 2)
    assert isinstance(result.p_2d, list)
    assert len(result.p_2d) == bearing.nh
    assert np.all(np.isfinite(result.w))
    assert np.all(np.isfinite(result.k))
    assert np.all(np.isfinite(result.moment))
    assert np.all(np.isfinite(result.shear_force))
    for p_field in result.p_2d:
        assert np.all(np.isfinite(p_field))
        assert np.min(p_field) >= 0.0


@pytest.mark.parametrize(
    "bearing_cls",
    [CircularBearing, AnnularBearing, InfiniteLinearBearing],
)
def test_analytic_and_fem_1d_loads_are_close(bearing_cls):
    bearing = bearing_cls(nx=20, nh=12)
    analytic = solve_bearing_analytic(bearing)
    numeric_1d = solve_bearing_fem_1d(bearing)

    relative_error = np.mean(
        np.abs(analytic.w - numeric_1d.w) / np.maximum(np.abs(analytic.w), 1e-9)
    )
    assert relative_error < 0.05
