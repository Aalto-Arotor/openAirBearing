"""Weak-form builders for load and boundary flow terms used by FEM solvers.

Load forms integrate gauge pressure by default. Flow forms follow the p^2
formulation and return volumetric flow in L/min.
"""

import numpy as np
from skfem import LinearForm
from skfem.helpers import dot, grad

__all__ = [
    "load_form_circular_axial",
    "supply_flow_form_1d",
    "boundary_flow_form_circular_rim",
    "load_form_2d_axial",
    "load_form_infinite_axial_per_width",
    "load_form_journal_projected_x",
    "load_form_journal_projected_y",
    "boundary_flow_form_edge",
]


def _unit_weight(_w):
    return 1.0


def _make_load_component(
    area_weight, n_dot_dir=lambda w: 1.0, *, subtract_pa: bool = True
):
    """
    Generic surface-load component:
        W_dir = ∬ area_weight(w) * (n · dir)(w) * [p - pa (optional)] dΩ_param
    - area_weight(w): Jacobian from parameter domain to physical area
    - n_dot_dir(w):   projection of unit normal onto desired direction
    """

    @LinearForm
    def load_form(v, w):
        p = w.p
        pa = getattr(w, "pa", 0.0)
        pg = (p - pa) if subtract_pa else p
        return area_weight(w) * n_dot_dir(w) * pg * v

    return load_form


def _make_qfilm_p2(boundary_weight, *, mu: float, p_ref: float, h=None):
    """
    Unified boundary flow Q [L/min] via p^2-form:
        Q = ∮ 6e4 * W(w) * [ -(h^3/(24 μ p_ref)) * 2 p (∇p·n) ] ds
    If `h` is:
      - callable(w): evaluated at facet quadrature points (captured in closure)
      - scalar: captured constant
      - None: expect `h` to be provided at assemble(...)
        as scalar/array/DiscreteField
    """
    K = 6e4 / (24.0 * mu * p_ref)

    if callable(h):

        @LinearForm
        def qfilm(v, w):
            dpdn = dot(grad(w.p), w.n)
            qn = -K * (h(w) ** 3) * (2.0 * w.p * dpdn)
            return boundary_weight(w) * qn * v

        return qfilm

    if h is not None and np.isscalar(h):
        h_const = float(h)

        @LinearForm
        def qfilm(v, w):
            pval = w["p"]
            dpdn = dot(grad(pval), w.n)
            qn = -K * (h_const**3) * (2.0 * pval * dpdn)
            return boundary_weight(w) * qn * v

        return qfilm

    @LinearForm
    def qfilm(v, w):
        hval = w["h"]  # scalar/array/DiscreteField only
        pval = w["p"]
        dpdn = dot(grad(pval), w.n)
        qn = -K * (hval**3) * (2.0 * pval * dpdn)
        return boundary_weight(w) * qn * v

    return qfilm


def _make_supply_flow_form(b, area_weight):
    """
    Generic supply flow component [L/min]:
        Qs = ∬ area_weight(w) * qs_per_A(p) dΩ_param
    - area_weight(w): Jacobian from parameter domain to physical area
    """

    @LinearForm
    def flow_form(v, w):
        p = w.p
        qs_per_A = b.kappa * 6e4 * (b.ps**2 - p**2) / (2 * b.mu * b.hp * b.pa)
        # pa = getattr(w, "pa", 0.0)
        return area_weight(w) * qs_per_A * v

    return flow_form


# =========================
# Circular / Annular thrust
# (axisymmetric 1D in r)
# =========================


def _area_weight_circular(w):
    return 2.0 * np.pi * w.x[0]


def load_form_circular_axial():
    """Return linear form for axial load on circular/annular geometries."""
    return _make_load_component(_area_weight_circular, _unit_weight)


def supply_flow_form_1d(b):
    """Return supply-flow form for 1D radial FEM model."""
    return _make_supply_flow_form(b, _area_weight_circular)


def boundary_flow_form_circular_rim(*, mu: float, p_ref: float, h):
    """Return rim boundary-flow form for radial 1D circular/annular FEM."""
    return _make_qfilm_p2(_area_weight_circular, mu=mu, p_ref=p_ref, h=h)


# =========================
# Rectangular (planar 2D)
# & “Infinite” (per-unit width)
# =========================

_area_weight_rectangular = _unit_weight


def load_form_2d_axial():
    """Return axial load form for planar rectangular geometries."""
    return _make_load_component(
        _area_weight_rectangular,
        _unit_weight,
    )


def load_form_infinite_axial_per_width():
    """Return axial load form per unit width for infinite-line geometry."""
    return _make_load_component(
        _area_weight_rectangular,
        _unit_weight,
    )


def boundary_flow_form_edge(*, mu: float, p_ref: float, h):
    """Return boundary-flow form for planar edge boundaries."""
    return _make_qfilm_p2(_unit_weight, mu=mu, p_ref=p_ref, h=h)


# =========================
# Journal (cylinder surface, unwrapped)
# =========================


def load_form_journal_projected_x(*, radius: float):
    """Return projected journal load form in x-direction."""
    if radius <= 0.0:
        raise ValueError("Journal radius must be positive.")

    def _n_dot_x(w):
        return np.cos(w.x[1] / radius)

    return _make_load_component(_unit_weight, _n_dot_x)


def load_form_journal_projected_y(*, radius: float):
    """Return projected journal load form in y-direction."""
    if radius <= 0.0:
        raise ValueError("Journal radius must be positive.")

    def _n_dot_y(w):
        return np.sin(w.x[1] / radius)

    return _make_load_component(_unit_weight, _n_dot_y)
