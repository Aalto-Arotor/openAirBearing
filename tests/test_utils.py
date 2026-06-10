import numpy as np
import pytest

from openairbearing.bearings import (
    BaseBearing,
    CircularBearing,
    JournalBearing,
    RectangularBearing,
)
from openairbearing.mesh import OABMeshQuad1
from openairbearing.utils import (
    get_area,
    get_beta,
    get_dA,
    get_geom_1d,
    get_geom_2d,
    get_kappa,
    get_load_capacity,
    get_Qsc,
    get_stiffness,
    get_supply_flow,
    get_volumetric_flow,
)


def test_get_area():
    """Test the get_area function for different bearing cases."""
    bearing = BaseBearing(case="circular", xa=10)
    assert get_area(bearing) == pytest.approx(np.pi * 10**2)

    bearing = BaseBearing(case="annular", xa=10, xc=5)
    assert get_area(bearing) == pytest.approx(np.pi * (10**2 - 5**2))

    bearing = BaseBearing(case="infinite", xa=10)
    assert get_area(bearing) == pytest.approx(10)

    bearing = BaseBearing(case="rectangular", xa=10, ya=5)
    assert get_area(bearing) == pytest.approx(10 * 5)

    bearing = BaseBearing(case="journal", xa=10, ya=5)
    assert get_area(bearing) == pytest.approx(10 * 5)

    with pytest.raises(ValueError, match="Unknown case: unknown"):
        bearing = BaseBearing(case="unknown")
        get_area(bearing)


def test_journal_bearing_unwrap_grid():
    """Journal grid should be initialized from circumferential unwrap."""
    bearing = JournalBearing(
        bore_diameter=80e-3,
        shaft_diameter=79.9e-3,
        xa=30e-3,
        nx=13,
        ny=17,
    )
    unwrap_diameter = 0.5 * (bearing.bore_diameter + bearing.shaft_diameter)
    expected_ya = np.pi * unwrap_diameter

    assert bearing.case == "journal"
    assert bearing.ya == pytest.approx(expected_ya)
    assert bearing.x.shape == (bearing.nx,)
    assert bearing.y.shape == (bearing.ny,)
    assert bearing.clearance == pytest.approx(0.5 * (80e-3 - 79.9e-3))


def test_journal_bearing_derive_missing_parameter():
    """Any one of bore_diameter, shaft_diameter, clearance can be derived."""
    b1 = JournalBearing(bore_diameter=80e-3, shaft_diameter=79.9e-3, clearance=None)
    assert b1.clearance == pytest.approx(0.05e-3)

    b2 = JournalBearing(bore_diameter=None, shaft_diameter=79.9e-3, clearance=0.05e-3)
    assert b2.bore_diameter == pytest.approx(80e-3)

    b3 = JournalBearing(bore_diameter=80e-3, shaft_diameter=None, clearance=0.05e-3)
    assert b3.shaft_diameter == pytest.approx(79.9e-3)

    with pytest.raises(ValueError, match="At most one"):
        JournalBearing(bore_diameter=None, shaft_diameter=None, clearance=0.05e-3)


def test_journal_bearing_default_h_func_tracks_min_gap_from_eccentricity_sweep():
    bearing = JournalBearing(
        bore_diameter=80e-3,
        shaft_diameter=79.9e-3,
        nx=10,
        ny=32,
        nh=7,
    )

    expected_min_gap = bearing.clearance - bearing.eccentricity_sweep
    assert np.allclose(bearing.h, expected_min_gap)
    assert np.allclose(bearing.ha, expected_min_gap)

    x = bearing.fem_2d.basis.doflocs[0]
    y = bearing.fem_2d.basis.doflocs[1]
    for i in range(bearing.nh):
        h_field = bearing.h_func_2d(x, y, i)
        assert np.min(h_field) == pytest.approx(
            expected_min_gap[i], rel=1e-6, abs=1e-12
        )


def test_journal_bearing_uses_height_bounds_for_eccentricity_sweep():
    bearing = JournalBearing(
        bore_diameter=80e-3,
        shaft_diameter=79.9e-3,
        ha_min=2e-6,
        ha_max=8e-6,
        nh=6,
    )

    assert np.min(bearing.ha) == pytest.approx(2e-6)
    assert np.max(bearing.ha) == pytest.approx(8e-6)
    assert np.min(bearing.eccentricity_sweep) == pytest.approx(bearing.clearance - 8e-6)
    assert np.max(bearing.eccentricity_sweep) == pytest.approx(bearing.clearance - 2e-6)


def test_get_geom_1d():
    """Test the get_geom_1d function for different error types."""
    nx = 10

    bearing = BaseBearing(
        case="circular", csys="polar", nx=nx, xa=10, ya=5, error_type="none"
    )
    geom = get_geom_1d(bearing)
    assert np.all(geom == 0)

    bearing.error_type = "linear"
    geom = get_geom_1d(bearing)
    assert geom.shape == (nx,)
    assert np.min(geom) == pytest.approx(0.0)

    bearing.error_type = "quadratic"
    geom = get_geom_1d(bearing)
    assert geom.shape == (nx,)
    assert np.min(geom) == pytest.approx(0.0)

    bearing.error_type = "invalid"
    geom = get_geom_1d(bearing)
    assert np.all(geom == 0)


def test_get_geom_2d():
    """Test the get_geom_2d function for representative bearing types."""
    circular = CircularBearing(nx=16, nh=6, error_type="saddle", error=2e-6)
    circular_geom = get_geom_2d(
        circular,
        x=circular.fem_2d.basis.doflocs[0],
        y=circular.fem_2d.basis.doflocs[1],
    )
    assert circular_geom.ndim == 1
    assert circular_geom.shape[0] == circular.fem_2d.basis.doflocs.shape[1]
    assert np.min(circular_geom) == pytest.approx(0.0)

    rectangular = RectangularBearing(nx=14, ny=10, nh=6, error_type="tiltx", error=1e-6)
    rectangular_geom = get_geom_2d(
        rectangular,
        x=rectangular.fem_2d.basis.doflocs[0],
        y=rectangular.fem_2d.basis.doflocs[1],
    )
    assert rectangular_geom.ndim == 1
    assert rectangular_geom.shape[0] == rectangular.fem_2d.basis.doflocs.shape[1]
    assert np.min(rectangular_geom) == pytest.approx(0.0)

    rectangular.error_type = "invalid"
    with pytest.raises(ValueError, match="Unknown error type"):
        get_geom_2d(
            rectangular,
            x=rectangular.fem_2d.basis.doflocs[0],
            y=rectangular.fem_2d.basis.doflocs[1],
        )


def test_get_geom_2d_journal_error_modes():
    """Journal bearing geometry errors should support conicity and misalignment."""
    bearing = JournalBearing(
        bore_diameter=80e-3,
        shaft_diameter=79.9e-3,
        nx=14,
        ny=18,
        nh=5,
        eccentricity=4e-6,
    )
    x = bearing.fem_2d.basis.doflocs[0]
    y = bearing.fem_2d.basis.doflocs[1]

    bearing.error_type = "conicity"
    bearing.error = 3e-6
    conicity = get_geom_2d(bearing, x=x, y=y)
    assert conicity.ndim == 1
    assert np.min(conicity) == pytest.approx(0.0)
    assert np.max(conicity) == pytest.approx(3e-6)

    bearing.error_type = "misalignment"
    bearing.error = 3e-6
    misalignment = get_geom_2d(bearing, x=x, y=y)
    assert misalignment.ndim == 1
    assert np.min(misalignment) < 0.0
    assert np.max(misalignment) > 0.0
    assert np.mean(misalignment) == pytest.approx(0.0, abs=1e-9)
    assert not np.allclose(conicity, misalignment)


def test_get_beta():
    """Test the get_beta function."""
    ha = np.linspace(1e-6, 20e-6, 10)
    hp = 4e-3
    kappa = 1e-15
    xa = 15
    bearing = BaseBearing(case="circular", xa=xa, hp=hp)
    bearing.kappa = kappa
    bearing.ha = ha
    beta = get_beta(bearing)
    assert beta.shape == ha.shape
    assert beta == pytest.approx(6 * kappa * xa**2 / (hp * ha**3))


def test_get_kappa():
    """Test the get_kappa function."""
    bearing = BaseBearing(
        case="circular", Qsc=3, mu=1.85e-5, hp=1e-3, pa=101325, psc=0.6e6 + 101325
    )
    bearing.A = get_area(bearing)
    kappa = get_kappa(bearing)
    assert kappa > 0


def test_get_Qsc():
    """Test the get_Qsc function."""
    bearing = BaseBearing(case="circular", psc=0.6e6)
    bearing.kappa = 1e-15
    Qsc = get_Qsc(bearing)
    assert Qsc > 0


def test_get_dA():
    """Test the get_dA function."""
    bearing = BaseBearing(case="circular", csys="polar", nx=10, ny=1, xa=10)
    dA = get_dA(bearing)
    assert dA.shape == (10, 1)
    assert np.all(dA > 0)

    bearing = BaseBearing(case="infinite", csys="cartesian", nx=10, ny=1, xa=10)
    dA = get_dA(bearing)
    assert dA.shape == (10, 1)
    assert np.all(dA > 0)


def test_get_load_capacity_shape():
    bearing = BaseBearing(case="infinite", csys="cartesian", nx=15, nh=7, xa=10)
    p = np.full((bearing.nx, bearing.nh), bearing.pa)
    w = get_load_capacity(bearing, p)
    assert w.shape == (bearing.nh,)
    assert np.allclose(w, 0.0)


# def test_get_load_capacity():
#     """Test the get_load_capacity function."""
#     bearing = BaseBearing(
#         case="circular", type="bearing", csys="polar",
#         nx=10, ny=1, xa=10, pa=101325,
#     )
#     p = np.ones(10) * 101325
#     w = get_load_capacity(bearing, p)
#     assert w.shape == p.shape ,  f"w {w.shape}, p {p.shape}"
#     # assert np.allclose(w, np.sum (p * bearing.dA, axis=0))


def test_get_stiffness():
    """Test the get_stiffness function."""
    nh = 40
    bearing = BaseBearing(case="circular", ha_min=1e-6, ha_max=20e-6, nh=40)
    w = 2 ** np.linspace(0, 100, nh)
    k = get_stiffness(bearing, w)
    expected = -np.gradient(w) / np.gradient(bearing.ha) / 1e6
    assert np.allclose(k, expected)


def test_flow_helpers_shapes_and_signals():
    bearing = CircularBearing(nx=20, nh=8)
    p = np.full((bearing.nx, bearing.nh), bearing.pa)

    qs_direct = get_supply_flow(bearing, p)
    qs, qa, qc = get_volumetric_flow(bearing, p)

    assert qs_direct.shape == (bearing.nh,)
    assert qs.shape == (bearing.nh,)
    assert qa.shape == (bearing.nh,)
    assert qc.shape == (bearing.nh,)
    assert np.allclose(qs, qs_direct)


def test_oab_mesh_quad1_circle_ratio_sets_corner_radius():
    """O-grid ratio should map to inner-square corner radius / outer radius."""
    diameter = 10.0
    nc = 6
    nr = 3
    ratio = 0.7
    mesh = OABMeshQuad1.init_circle(diameter=diameter, nc=nc, nr=nr, ratio=ratio)

    radius = 0.5 * diameter
    n_center = (nc + 1) ** 2
    center_nodes = mesh.p[:, :n_center]
    center_radii = np.sqrt(center_nodes[0] ** 2 + center_nodes[1] ** 2)

    assert np.max(center_radii) == pytest.approx(ratio * radius)


def test_oab_mesh_quad1_circle_has_bell_shaped_inner_sides():
    """Inner block side midpoints should bulge outward vs the baseline square."""
    diameter = 10.0
    nc = 6
    nr = 3
    ratio = 0.7
    mesh = OABMeshQuad1.init_circle(diameter=diameter, nc=nc, nr=nr, ratio=ratio)

    radius = 0.5 * diameter
    half_side = (ratio * radius) / np.sqrt(2.0)
    n_center = (nc + 1) ** 2
    center_nodes = mesh.p[:, :n_center]
    x = center_nodes[0]
    y = center_nodes[1]

    top_edge = np.isclose(y, np.max(y))
    top_mid = top_edge & np.isclose(x, 0.0)

    # Mid-side node should be pushed outward from the undeformed baseline.
    assert np.any(top_mid)
    assert np.max(y[top_mid]) > half_side

    # Corner node location should remain unchanged by bell deformation.
    top_right_corner = np.isclose(x, half_side) & np.isclose(y, half_side)
    assert np.any(top_right_corner)


def test_oab_mesh_quad1_circle_bell_shape_distributes_into_core():
    """Bell shaping should move interior core nodes, not boundary only."""
    diameter = 10.0
    nc = 6
    nr = 3
    ratio = 0.7
    mesh = OABMeshQuad1.init_circle(diameter=diameter, nc=nc, nr=nr, ratio=ratio)

    radius = 0.5 * diameter
    half_side = (ratio * radius) / np.sqrt(2.0)
    n_center = (nc + 1) ** 2
    center_nodes = mesh.p[:, :n_center]
    x = center_nodes[0]
    y = center_nodes[1]
    xs = np.linspace(-half_side, half_side, nc + 1)
    ys = np.linspace(-half_side, half_side, nc + 1)
    gx0, gy0 = np.meshgrid(xs, ys)
    x0 = gx0.ravel()
    y0 = gy0.ravel()

    # Interior nodes should move (not just edge nodes).
    interior = (~np.isclose(np.abs(x0), half_side)) & (
        ~np.isclose(np.abs(y0), half_side)
    )
    displacement = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    assert np.max(displacement[interior]) > 0.0

    # One representative interior node should move outward on the top half.
    probe = np.isclose(x0, 0.0) & np.isclose(y0, half_side / 3.0)
    assert np.any(probe)
    assert np.max(y[probe]) > np.max(y0[probe])
