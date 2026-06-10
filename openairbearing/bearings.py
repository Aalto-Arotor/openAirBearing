"""Bearing geometry and property classes for hydrodynamic and hydrostatic bearings.

Defines base and specialized bearing types (circular, annular,
rectangular, infinite linear) with geometry and fluid film properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skfem import Basis, MeshLine, MeshQuad, MeshTri
from skfem.element import ElementLineP2, ElementQuad2, ElementTriP2

from openairbearing.mesh import OABMeshQuad1
from openairbearing.utils import (
    get_area,
    get_beta,
    get_geom_1d,
    get_geom_2d,
    get_geom_2d_journal,
    get_kappa,
)


@dataclass
class FEM1DState:
    mesh: object | None = None
    basis: object | None = None
    D: np.ndarray | None = None
    p: np.ndarray | None = None
    boundary_sets: dict[str, object] | None = None


@dataclass
class FEM2DState:
    mesh: object | None = None
    basis: object | None = None
    eta: np.ndarray | None = None
    p: np.ndarray | None = None
    D: np.ndarray | None = None
    boundary_sets: dict[str, object] | None = None


@dataclass
class FEM3DState:
    mesh: object | None = None
    basis: object | None = None
    p: np.ndarray | None = None
    D: np.ndarray | None = None
    boundary_sets: dict[str, object] | None = None


@dataclass(frozen=True)
class FEMInitSpec1D:
    mesh_factory: callable
    basis_factory: callable


@dataclass(frozen=True)
class FEMInitSpec2D:
    mesh_factory: callable
    basis_factory: callable
    set_primary_basis: bool = False


@dataclass(frozen=True)
class FEMBoundarySpec1D:
    ambient_dofs: object
    chamber_dofs: object | None = None
    ambient_pressure: float | None = None
    chamber_pressure: float | None = None


@dataclass(frozen=True)
class FEMBoundaryLabels1D:
    ambient: str = "ambient"
    chamber: str | None = None
    ambient_pressure: float | None = None
    chamber_pressure: float | None = None


@dataclass(frozen=True)
class FEMBoundarySpec2D:
    ambient_dofs: object
    chamber_dofs: object | None = None
    ambient_pressure: float | None = None
    chamber_pressure: float | None = None


@dataclass
class BaseBearing:
    """Base class for all bearing types."""

    pa: float = 101325
    pc: float = pa
    ps: float = 0.6e6 + pa

    rho: float = 1.293
    mu: float = 1.85e-5

    hp: float = 4.5e-3

    ha_min: float = 1e-6
    ha_max: float = 10e-6

    xa: float = 37 / 2 * 1e-3
    xc: float = 0
    ya: float = 0
    nh: int = 20
    nx: int = 20
    ny: int = 1

    u: np.ndarray = field(default_factory=lambda: np.zeros(2))

    Psi: float = 0

    error_type: str = "none"
    error: float = 0e-6

    blocked: bool = False
    block_x: float = 25.2e-3 / 2
    block_w: float = 1e-3

    # short-circuit flow and pressure
    Qsc: float = 3  # L/min
    psc: float = 0.6e6 + pa

    x: np.ndarray | None = None
    dx: np.ndarray | None = None
    y: np.ndarray | None = None
    dy: np.ndarray | None = None

    mesh: object | None = None
    basis: object | None = None

    h: np.ndarray | None = None
    ha: np.ndarray | None = None
    A: float | None = None
    kappa: float | None = None
    beta: float | None = None
    # geom_func: callable | None = None
    geom_func_1d: callable | None = None
    geom_func_2d: callable | None = None
    geom_1d: np.ndarray | None = None
    geom_2d: np.ndarray | None = None
    h_func_1d: callable | None = None
    h_func_2d: callable | None = None

    fem_1d: FEM1DState = field(default_factory=FEM1DState)
    fem_2d: FEM2DState = field(default_factory=FEM2DState)
    fem_3d: FEM3DState = field(default_factory=FEM3DState)

    case: str = "base"
    type: str = "bearing"
    csys: str = "cartesian"

    def _coalesce(self, value, factory):
        return value if value is not None else factory()

    @staticmethod
    def _dofs_to_array(dofs):
        if dofs is None:
            return np.array([], dtype=int)
        if isinstance(dofs, np.ndarray):
            return np.asarray(dofs, dtype=int).ravel()
        if hasattr(dofs, "all"):
            return np.asarray(dofs.all(), dtype=int)
        return np.asarray(dofs, dtype=int)

    def _init_fem_1d(self, spec: FEMInitSpec1D):
        self.fem_1d.mesh = self._coalesce(self.fem_1d.mesh, spec.mesh_factory)
        self.fem_1d.basis = self._coalesce(
            self.fem_1d.basis,
            lambda: spec.basis_factory(self.fem_1d.mesh),
        )

    def _apply_boundary_spec_1d(self, spec: FEMBoundarySpec1D):
        ambient_pressure = (
            self.pa if spec.ambient_pressure is None else spec.ambient_pressure
        )
        chamber_pressure = (
            self.pc if spec.chamber_pressure is None else spec.chamber_pressure
        )

        self.fem_1d.p = self.fem_1d.basis.zeros() + self.ps**2 - ambient_pressure**2
        self.fem_1d.p[spec.ambient_dofs] = self.ps**2 - ambient_pressure**2

        if spec.chamber_dofs is not None:
            self.fem_1d.p[spec.chamber_dofs] = self.ps**2 - chamber_pressure**2

        all_boundary_dofs = np.unique(
            np.hstack(
                [
                    self._dofs_to_array(spec.ambient_dofs),
                    self._dofs_to_array(spec.chamber_dofs),
                ]
            )
        )
        self.fem_1d.boundary_sets = {
            "ambient": spec.ambient_dofs,
            "chamber": spec.chamber_dofs,
            "all": all_boundary_dofs,
        }
        self.fem_1d.D = self.fem_1d.boundary_sets["all"]

    def _init_1d_boundaries(self, labels: FEMBoundaryLabels1D):
        ambient_dofs = self.fem_1d.basis.get_dofs(labels.ambient)
        chamber_dofs = (
            self.fem_1d.basis.get_dofs(labels.chamber)
            if labels.chamber is not None
            else None
        )
        self._apply_boundary_spec_1d(
            FEMBoundarySpec1D(
                ambient_dofs=ambient_dofs,
                chamber_dofs=chamber_dofs,
                ambient_pressure=labels.ambient_pressure,
                chamber_pressure=labels.chamber_pressure,
            )
        )

    def _init_fem_2d(self, spec: FEMInitSpec2D):
        self.fem_2d.mesh = self._coalesce(self.fem_2d.mesh, spec.mesh_factory)
        self.fem_2d.basis = self._coalesce(
            self.fem_2d.basis,
            lambda: spec.basis_factory(self.fem_2d.mesh),
        )
        if spec.set_primary_basis:
            self.mesh = self.fem_2d.mesh
            self.basis = self.fem_2d.basis

    def _apply_boundary_spec_2d(self, spec: FEMBoundarySpec2D):
        ambient_pressure = (
            self.pa if spec.ambient_pressure is None else spec.ambient_pressure
        )
        chamber_pressure = (
            self.pc if spec.chamber_pressure is None else spec.chamber_pressure
        )

        self.fem_2d.eta = self.fem_2d.basis.zeros()
        self.fem_2d.p = self.fem_2d.basis.zeros()

        self.fem_2d.eta[spec.ambient_dofs] = self.ps**2 - ambient_pressure**2
        self.fem_2d.p[spec.ambient_dofs] = ambient_pressure

        if spec.chamber_dofs is not None:
            self.fem_2d.eta[spec.chamber_dofs] = self.ps**2 - chamber_pressure**2
            self.fem_2d.p[spec.chamber_dofs] = chamber_pressure

        all_boundary_dofs = np.unique(
            np.hstack(
                [
                    self._dofs_to_array(spec.ambient_dofs),
                    self._dofs_to_array(spec.chamber_dofs),
                ]
            )
        )
        self.fem_2d.boundary_sets = {
            "ambient": spec.ambient_dofs,
            "chamber": spec.chamber_dofs,
            "all": all_boundary_dofs,
        }
        self.fem_2d.D = self.fem_2d.boundary_sets["all"]

    def __post_init__(self):
        if self.fem_2d.mesh is None and self.mesh is not None:
            self.fem_2d.mesh = self.mesh
        if self.mesh is None and self.fem_2d.mesh is not None:
            self.mesh = self.fem_2d.mesh
        if self.fem_2d.basis is None and self.basis is not None:
            self.fem_2d.basis = self.basis
        if self.basis is None and self.fem_2d.basis is not None:
            self.basis = self.fem_2d.basis

        self.ha = self._coalesce(
            self.ha, lambda: np.linspace(self.ha_min, self.ha_max, self.nh).T
        )
        self.h = self._coalesce(self.h, lambda: self.ha.copy())
        self.h_iter = np.arange(self.nh)

        def _default_h_func_1d(x, i):
            return self.ha[i] + self.geom_func_1d(bearing=self, x=x)

        def _default_h_func_2d(x, y, i):
            return self.ha[i] + self.geom_func_2d(bearing=self, x=x, y=y)

        self.h_func_1d = self._coalesce(self.h_func_1d, lambda: _default_h_func_1d)
        self.h_func_2d = self._coalesce(self.h_func_2d, lambda: _default_h_func_2d)

        self.x = self._coalesce(self.x, lambda: np.linspace(self.xc, self.xa, self.nx))
        self.y = self._coalesce(self.y, lambda: np.linspace(0, self.ya, self.ny))
        self.dx = self._coalesce(self.dx, lambda: np.gradient(self.x))
        self.dy = self._coalesce(
            self.dy, lambda: 1 if self.ny == 1 else np.gradient(self.y)
        )
        self.A = get_area(self)
        self.kappa = get_kappa(self)
        self.beta = get_beta(self)


@dataclass
class CircularBearing(BaseBearing):
    """Base class for circular thrust bearing"""

    case: str = "circular"
    type: str = "bearing"
    csys: str = "polar"

    nx: int = 50
    xc: float = 1e-6
    xa: float = 37e-3 / 2
    Qsc: float = 2.8  # L/min
    divs: int = 3

    def __post_init__(self):
        super().__post_init__()
        self.psc = self._coalesce(self.psc, lambda: 0.6e6 + self.pa)
        self.kappa = self._coalesce(self.kappa, lambda: get_kappa(self))
        self.beta = self._coalesce(self.beta, lambda: get_beta(self))

        def _make_circle_mesh1d():
            return MeshLine(self.x).with_boundaries(
                {"ambient": lambda x: np.isclose(x[0], self.xa)}
            )

        self._init_fem_1d(
            FEMInitSpec1D(
                mesh_factory=_make_circle_mesh1d,
                basis_factory=lambda mesh: Basis(mesh, ElementLineP2()),
            )
        )
        self._init_1d_boundaries(FEMBoundaryLabels1D(ambient="ambient"))

        def _make_circle_mesh2d():
            # mesh = MeshTri.init_circle(self.divs, smoothed=False).scaled(scale)
            # return mesh.with_boundaries({"outer": mesh.boundary_facets()})
            nc = int(np.floor(0.2 * self.nx))
            nr = int(np.ceil(0.5 * nc))
            return OABMeshQuad1.init_circle(diameter=2.0 * self.xa, nc=nc, nr=nr)

        self._init_fem_2d(
            FEMInitSpec2D(
                mesh_factory=_make_circle_mesh2d,
                basis_factory=lambda mesh: Basis(mesh, ElementQuad2()),
                set_primary_basis=True,
            )
        )

        self.geom_func_1d = self._coalesce(self.geom_func_1d, lambda: get_geom_1d)
        self.geom_func_2d = self._coalesce(self.geom_func_2d, lambda: get_geom_2d)
        self.geom_1d = self._coalesce(self.geom_1d, lambda: get_geom_1d(self))
        self.geom_2d = self._coalesce(
            self.geom_2d,
            lambda: self.geom_func_2d(
                bearing=self,
                x=self.fem_2d.basis.doflocs[0],
                y=self.fem_2d.basis.doflocs[1],
            ),
        )

        ambient_dofs = self.fem_2d.basis.get_dofs("outer")
        self._apply_boundary_spec_2d(FEMBoundarySpec2D(ambient_dofs=ambient_dofs))


@dataclass
class AnnularBearing(BaseBearing):
    """Base class for annular bearing"""

    case: str = "annular"
    type: str = "seal"
    csys: str = "polar"

    xa: float = 58e-3 / 2
    xc: float = 25e-3 / 2

    ya: float = 2 * np.pi
    nx: int = 32
    ny: int = 24
    Qsc: float = 3  # L/min

    def __post_init__(self):
        super().__post_init__()
        self.psc = self._coalesce(self.psc, lambda: 0.6e6 + self.pa)
        self.kappa = self._coalesce(self.kappa, lambda: get_kappa(self))
        self.beta = self._coalesce(self.beta, lambda: get_beta(self))

        # Ensure radial grid actually spans [xc, xa]
        self.x = self._coalesce(self.x, lambda: np.linspace(self.xc, self.xa, self.nx))
        self.y = self._coalesce(self.y, lambda: np.linspace(0.0, self.ya, self.ny))

        def _make_annular_mesh1d():
            return MeshLine(self.x).with_boundaries(
                {
                    "chamber": lambda x: np.isclose(x[0], self.xc),
                    "ambient": lambda x: np.isclose(x[0], self.xa),
                }
            )

        def _make_annular_mesh2d():
            return OABMeshQuad1.init_annulus(
                inner_diameter=2.0 * self.xc,
                outer_diameter=2.0 * self.xa,
                nr=self.nx,
                ntheta=self.ny,
            )

        self._init_fem_1d(
            FEMInitSpec1D(
                mesh_factory=_make_annular_mesh1d,
                basis_factory=lambda mesh: Basis(mesh, ElementLineP2()),
            )
        )
        self._init_1d_boundaries(
            FEMBoundaryLabels1D(ambient="ambient", chamber="chamber")
        )

        self._init_fem_2d(
            FEMInitSpec2D(
                mesh_factory=_make_annular_mesh2d,
                basis_factory=lambda mesh: Basis(mesh, ElementQuad2()),
            )
        )

        self.geom_func_1d = self._coalesce(self.geom_func_1d, lambda: get_geom_1d)
        self.geom_func_2d = self._coalesce(self.geom_func_2d, lambda: get_geom_2d)
        self.geom_1d = self._coalesce(self.geom_1d, lambda: get_geom_1d(self))
        self.geom_2d = self._coalesce(
            self.geom_2d,
            lambda: self.geom_func_2d(
                bearing=self,
                x=self.fem_2d.basis.doflocs[0],
                y=self.fem_2d.basis.doflocs[1],
            ),
        )

        ambient_dofs = self.fem_2d.basis.get_dofs("outer")
        chamber_dofs = self.fem_2d.basis.get_dofs("inner")
        self._apply_boundary_spec_2d(
            FEMBoundarySpec2D(ambient_dofs=ambient_dofs, chamber_dofs=chamber_dofs)
        )


@dataclass
class InfiniteLinearBearing(BaseBearing):
    """Base class for Infinitely long linear bearing bearing"""

    case: str = "infinite"
    type: str = "seal"
    csys: str = "cartesian"

    ps: float = 0.41e6
    xa: float = 40e-3

    Qsc: float = 37  # L/min
    basis: None = None

    def __post_init__(self):
        super().__post_init__()
        self.psc = self._coalesce(self.psc, lambda: 0.41e6 + self.pa)
        self.kappa = self._coalesce(self.kappa, lambda: get_kappa(self))
        self.beta = self._coalesce(self.beta, lambda: get_beta(self))

        def _make_infinite_mesh1d():
            return MeshLine(self.x).with_boundaries(
                {
                    "left": lambda x: np.isclose(x[0], self.x[0]),
                    "chamber": lambda x: np.isclose(x[0], self.x[0]),
                    "ambient": lambda x: np.isclose(x[0], self.x[-1]),
                }
            )

        self._init_fem_1d(
            FEMInitSpec1D(
                mesh_factory=_make_infinite_mesh1d,
                basis_factory=lambda mesh: Basis(mesh, ElementLineP2()),
            )
        )
        self._init_1d_boundaries(
            FEMBoundaryLabels1D(ambient="ambient", chamber="chamber")
        )

        self.geom_func_1d = self._coalesce(self.geom_func_1d, lambda: get_geom_1d)
        self.geom_1d = self._coalesce(self.geom_1d, lambda: get_geom_1d(self))


@dataclass
class RectangularBearing(BaseBearing):
    """Base class for rectangular thrust bearing"""

    case: str = "rectangular"
    type: str = "bearing"
    csys: str = "cartesian"

    xa: float = 40e-3
    ya: float = 80e-3
    nx: int = 20
    ny: int = 40

    ps: float = 0.41e6

    Qsc: float = 2.94  # L/min

    def __post_init__(self):
        super().__post_init__()
        self.psc = self._coalesce(self.psc, lambda: 0.41e6 + self.pa)
        self.x = np.linspace(-self.xa / 2, self.xa / 2, self.nx)
        self.y = np.linspace(-self.ya / 2, self.ya / 2, self.ny)

        self.dx = self._coalesce(self.dx, lambda: self.xa / (self.nx + 1))
        self.dy = self._coalesce(self.dy, lambda: self.ya / (self.ny + 1))
        self.kappa = self._coalesce(self.kappa, lambda: get_kappa(self))
        self.beta = self._coalesce(self.beta, lambda: get_beta(self))

        def _make_rectangular_mesh2d():
            return MeshQuad.init_tensor(self.x, self.y).with_boundaries(
                {
                    "left": lambda x: np.isclose(x[0], self.x[0]),
                    "right": lambda x: np.isclose(x[0], self.x[-1]),
                    "bottom": lambda x: np.isclose(x[1], self.y[0]),
                    "top": lambda x: np.isclose(x[1], self.y[-1]),
                }
            )

        self._init_fem_2d(
            FEMInitSpec2D(
                mesh_factory=_make_rectangular_mesh2d,
                basis_factory=lambda mesh: Basis(mesh, ElementQuad2()),
            )
        )

        self.geom_func_2d = self._coalesce(self.geom_func_2d, lambda: get_geom_2d)
        self.geom_2d = self._coalesce(
            self.geom_2d,
            lambda: self.geom_func_2d(
                bearing=self,
                x=self.fem_2d.basis.doflocs[0],
                y=self.fem_2d.basis.doflocs[1],
            ),
        )

        d_left = self.fem_2d.basis.get_dofs("left")
        d_right = self.fem_2d.basis.get_dofs("right")
        d_bot = self.fem_2d.basis.get_dofs("bottom")
        d_top = self.fem_2d.basis.get_dofs("top")
        all_boundary_dofs = np.unique(
            np.hstack([d_left.all(), d_right.all(), d_bot.all(), d_top.all()])
        )
        self._apply_boundary_spec_2d(FEMBoundarySpec2D(ambient_dofs=all_boundary_dofs))


@dataclass
class JournalBearing(BaseBearing):
    """Base class for radial journal bearing."""

    case: str = "journal"
    type: str = "seal"
    csys: str = "cartesian"

    bore_diameter: float | None = 50.020e-3
    shaft_diameter: float | None = 49.990e-3
    eccentricity: float = 0.0
    clearance: float | None = None

    xa: float = 89.0e-3
    ya: float | None = None
    nx: int = 40
    ny: int = 20

    ps: float = 0.41e6

    Qsc: float = 11  # L/min
    eccentricity_sweep: np.ndarray | None = None

    def __post_init__(self):
        def _default_journal_h_func_2d(x, y, i):
            y_span = float(np.max(self.y) - np.min(self.y))
            if np.isclose(y_span, 0.0):
                h_nominal = np.full_like(y, self.clearance - self.eccentricity_sweep[i])
            else:
                theta = 2.0 * np.pi * (np.asarray(y) - float(np.min(self.y))) / y_span
                h_nominal = self.clearance + self.eccentricity_sweep[i] * np.cos(theta)

            h_error = self.geom_func_2d(bearing=self, x=x, y=y)
            return np.asarray(h_nominal + h_error, dtype=float)

        self.h_func_2d = self._coalesce(
            self.h_func_2d,
            lambda: _default_journal_h_func_2d,
        )

        nones = sum(
            v is None for v in (self.bore_diameter, self.shaft_diameter, self.clearance)
        )
        if nones > 1:
            raise ValueError(
                "At most one of bore_diameter, shaft_diameter, clearance may be None."
            )

        if self.clearance is None:
            self.clearance = 0.5 * (self.bore_diameter - self.shaft_diameter)
        elif self.bore_diameter is None:
            self.bore_diameter = self.shaft_diameter + 2.0 * self.clearance
        elif self.shaft_diameter is None:
            self.shaft_diameter = self.bore_diameter - 2.0 * self.clearance

        if self.clearance <= 0:
            raise ValueError("Journal clearance must be positive.")
        if abs(self.eccentricity) > self.clearance:
            raise ValueError("Journal eccentricity must satisfy |e| <= clearance.")

        self.ya = self._coalesce(
            self.ya,
            lambda: np.pi * 0.5 * (self.bore_diameter + self.shaft_diameter),
        )

        super().__post_init__()
        self.psc = self._coalesce(self.psc, lambda: 0.41e6 + self.pa)

        h_min_target = float(np.clip(self.ha_min, 0.0, self.clearance))
        h_max_target = float(np.clip(self.ha_max, 0.0, self.clearance))
        if h_min_target > h_max_target:
            h_min_target, h_max_target = h_max_target, h_min_target

        self.ha_min = h_min_target
        self.ha_max = h_max_target

        self.eccentricity_sweep = self._coalesce(
            self.eccentricity_sweep,
            lambda: self.clearance - np.linspace(self.ha_min, self.ha_max, self.nh),
        )

        if self.eccentricity_sweep.shape[0] != self.nh:
            raise ValueError("eccentricity_sweep length must equal nh.")

        self.h = self.clearance - self.eccentricity_sweep
        self.ha = self.h.copy()

        self.x = np.linspace(-self.xa / 2, self.xa / 2, self.nx)
        self.y = np.linspace(-self.ya / 2, self.ya / 2, self.ny)

        self.dx = self._coalesce(self.dx, lambda: np.gradient(self.x))
        self.dy = self._coalesce(self.dy, lambda: np.gradient(self.y))
        self.kappa = self._coalesce(self.kappa, lambda: get_kappa(self))
        self.beta = self._coalesce(self.beta, lambda: get_beta(self))

        def _make_journal_mesh2d():
            return MeshTri.init_tensor(self.x, self.y).with_boundaries(
                {
                    "left": lambda x: np.isclose(x[0], self.x[0]),
                    "right": lambda x: np.isclose(x[0], self.x[-1]),
                    "periodic_bottom": lambda x: np.isclose(x[1], self.y[0]),
                    "periodic_top": lambda x: np.isclose(x[1], self.y[-1]),
                }
            )

        self._init_fem_2d(
            FEMInitSpec2D(
                mesh_factory=_make_journal_mesh2d,
                basis_factory=lambda mesh: Basis(mesh, ElementTriP2()),
                set_primary_basis=True,
            )
        )

        self.geom_func_2d = self._coalesce(
            self.geom_func_2d, lambda: get_geom_2d_journal
        )

        def _default_geom_2d():
            return self.geom_func_2d(
                bearing=self,
                x=self.fem_2d.basis.doflocs[0],
                y=self.fem_2d.basis.doflocs[1],
            )

        self.geom_2d = self._coalesce(self.geom_2d, _default_geom_2d)

        d_left = self.fem_2d.basis.get_dofs("left")
        d_right = self.fem_2d.basis.get_dofs("right")
        self._apply_boundary_spec_2d(
            FEMBoundarySpec2D(ambient_dofs=d_right, chamber_dofs=d_left)
        )
