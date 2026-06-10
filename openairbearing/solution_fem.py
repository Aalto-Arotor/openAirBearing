"""Finite element method solvers for bearing gap flow and pressure.

Provides 1D and 2D FEM-based solutions for hydrodynamic and hydrostatic
bearing pressure distributions using linear and nonlinear algorithms.
"""

import numpy as np
from skfem import (
    BilinearForm,
    LinearForm,
    asm,
    condense,
    solve,
)
from skfem.autodiff import NonlinearForm
from skfem.helpers import dot, grad

from openairbearing.fem_utils import (
    boundary_flow_form_circular_rim,
    boundary_flow_form_edge,
    load_form_2d_axial,
    load_form_circular_axial,
    load_form_infinite_axial_per_width,
    load_form_journal_projected_x,
    load_form_journal_projected_y,
    supply_flow_form_1d,
)
from openairbearing.utils import Result, get_stiffness


def _sweep_over_h(nh, solve_one_h):
    """Run a solver callback over all film-height points and stack outputs."""
    p_list, w_list, qa_list, qc_list, qs_list = [], [], [], [], []
    for i in range(nh):
        p_vec, w_i, qa_i, qc_i, qs_i = solve_one_h(i)
        p_list.append(p_vec)
        w_list.append(w_i)
        qa_list.append(qa_i)
        qc_list.append(qc_i)
        qs_list.append(qs_i)
    return (
        np.array(p_list),
        np.array(w_list),
        np.array(qa_list),
        np.array(qc_list),
        np.array(qs_list),
    )


def _assemble_eta_matrix_1d(b, basis, h_func, geometry_weight):
    """
    Assemble 1D eta-form Reynolds stiffness matrix.
    eta = ps^2 - p^2
    """

    @BilinearForm
    def reynolds(eta, v, w):
        h = h_func(w)
        return geometry_weight(w) * (
            h**3 / 12 * (1.0 + b.Psi) * grad(eta)[0] * grad(v)[0]
            + eta * v * b.kappa / b.hp
        )

    return reynolds.assemble(basis)


def _evaluate_1d_result_terms(b, basis, p_dofs, h_func, load_form, boundary_flow_form):
    """Evaluate load and flow terms for a solved 1D pressure field."""
    # pfield = p_dofsbasis.interpolate(p_dofs)
    load_value = asm(load_form, basis, p=p_dofs, pa=b.pa).sum()
    ambient_boundary = basis.boundary("ambient")
    if "chamber" in basis.mesh.boundaries:
        chamber_boundary = basis.boundary("chamber")
    else:
        chamber_boundary = None

    qa = asm(
        boundary_flow_form(mu=b.mu, p_ref=b.pa, h=h_func),
        ambient_boundary,
        p=p_dofs,
    ).sum()  # to ambient
    qc = (
        -asm(
            boundary_flow_form(mu=b.mu, p_ref=b.pa, h=h_func),
            chamber_boundary,
            p=p_dofs,
        ).sum()
        if chamber_boundary is not None
        else 0.0
    )  # to chamber

    qs = asm(supply_flow_form_1d(b), basis, p=p_dofs).sum()
    return load_value, qa, qc, qs


def solve_bearing_fem_1d(b):
    """Solve 1D FEM pressure/load/flow curves for one bearing configuration."""

    def _weight_polar(w):
        return w.x[0]

    def _weight_unit(w):
        return 1.0

    match b.csys:
        case "polar":
            geometry_weight = _weight_polar
            load_form = load_form_circular_axial()
            boundary_flow_form = boundary_flow_form_circular_rim
        case "cartesian":
            geometry_weight = _weight_unit
            load_form = load_form_infinite_axial_per_width()
            boundary_flow_form = boundary_flow_form_edge
        case _:
            geometry_weight = _weight_unit

    fem_1d = b.fem_1d
    basis = fem_1d.basis

    eta_dirichlet = fem_1d.p
    D = fem_1d.D

    def solve_one_h(i):
        def h_func(w):
            return b.h_func_1d(w.x[0], i)

        system_matrix = _assemble_eta_matrix_1d(
            b=b,
            basis=basis,
            h_func=h_func,
            geometry_weight=geometry_weight,
        )

        eta = solve(*condense(system_matrix, x=eta_dirichlet, D=D))
        p_dofs = np.sqrt(b.ps**2 - eta)

        load_value, qa, qc, qs = _evaluate_1d_result_terms(
            b=b,
            basis=basis,
            p_dofs=p_dofs,
            h_func=h_func,
            load_form=load_form,
            boundary_flow_form=boundary_flow_form,
        )

        return p_dofs, load_value, qa, qc, qs

    p, w, qa, qc, qs = _sweep_over_h(b.nh, solve_one_h)

    k = get_stiffness(bearing=b, w=w)

    name = "numeric 1d"
    return Result(name=name, p_1d=p, w=w, k=k, qs=qs, qa=qa, qc=qc)


def solve_bearing_fem_2d(b):
    """
    Solve 2D FEM pressure/load/flow curves for circular, annular, or rectangular cases.

    The formulation uses eta = ps^2 - p^2 and case-specific Dirichlet boundaries.

    Returns:
        Result: p_2d contains nodal pressure
        vectors per height index.
    """

    fem_2d = b.fem_2d
    eta_dirichlet = fem_2d.eta
    boundary_sets = fem_2d.boundary_sets

    def solve_one_h(i):
        def h_func(w):
            return b.h_func_2d(w.x[0], w.x[1], i)

        p_dofs, system_matrix = _solve_stationary_pressure_dofs_2d(
            b=b,
            h_func=h_func,
            eta_dirichlet=eta_dirichlet,
            boundary_dofs=fem_2d.D,
        )
        w_i, qa_i, qc_i, qs_i = _evaluate_2d_result_terms(
            b=b,
            p_dofs=p_dofs,
            system_matrix=system_matrix,
            boundary_sets=boundary_sets,
        )
        return p_dofs, w_i, qa_i, qc_i, qs_i

    p, w, qa, qc, qs = _sweep_over_h(b.nh, solve_one_h)
    k, moment, shear_force, p = _postprocess_2d_outputs(
        b,
        p,
        w,
        include_shear=True,
    )

    name = "numeric 2d"
    return Result(
        name=name,
        w=w,
        k=k,
        qs=qs,
        qa=qa,
        qc=qc,
        p_2d=p,
        moment=moment,
        shear_force=shear_force,
    )


def _assemble_eta_matrix_2d(b, h_func):
    @BilinearForm
    def reynolds(p, v, w):
        h = h_func(w)
        return (
            h**3 * (1.0 + b.Psi) * dot(grad(p), grad(v)) + 12.0 * p * v * b.kappa / b.hp
        )

    return reynolds.assemble(b.fem_2d.basis)


def _solve_stationary_pressure_dofs_2d(b, h_func, eta_dirichlet, boundary_dofs):
    system_matrix = _assemble_eta_matrix_2d(b, h_func)
    eta = solve(*condense(system_matrix, x=eta_dirichlet, D=boundary_dofs))
    p_dofs = np.sqrt(np.maximum(b.ps**2 - eta, 0.0))
    return p_dofs, system_matrix


def _evaluate_2d_result_terms(b, p_dofs, system_matrix, boundary_sets):
    basis = b.fem_2d.basis
    pfield = basis.interpolate(p_dofs)

    if b.case == "journal":
        journal_radius = 0.25 * (b.bore_diameter + b.shaft_diameter)
        wx = asm(
            load_form_journal_projected_x(radius=journal_radius),
            basis,
            p=pfield,
            pa=b.pa,
        ).sum()
        wy = asm(
            load_form_journal_projected_y(radius=journal_radius),
            basis,
            p=pfield,
            pa=b.pa,
        ).sum()
        w_i = np.hypot(wx, wy)
    else:
        w_i = asm(load_form_2d_axial(), basis, p=pfield, pa=b.pa).sum()

    eta = np.maximum(b.ps**2 - p_dofs**2, 0.0)
    boundary_residual = system_matrix @ eta
    flow_scale = 6e4 / (24.0 * b.mu * b.pa)

    qa_i = flow_scale * boundary_residual[boundary_sets["ambient"]].sum()
    if b.case in ("annular", "journal"):
        qc_i = -flow_scale * boundary_residual[boundary_sets["chamber"]].sum()
    else:
        qc_i = 0.0
    qs_i = qa_i - qc_i
    return w_i, qa_i, qc_i, qs_i


def _evaluate_pressure_moment_2d(b, p_dofs):
    """Return pressure moment components [Mx, My] in N·m."""
    basis = b.fem_2d.basis
    pfield = basis.interpolate(p_dofs)

    @LinearForm
    def moment_x(v, w):
        return w.x[0] * (w.p - w.pa) * v

    @LinearForm
    def moment_y(v, w):
        return -w.x[1] * (w.p - w.pa) * v

    mx = asm(moment_x, basis, p=pfield, pa=b.pa).sum()
    my = asm(moment_y, basis, p=pfield, pa=b.pa).sum()
    return np.array([mx, my], dtype=float)


def _evaluate_shear_force_2d(b, p_dofs, h_dofs):
    """Return net shear-force components [Fx, Fy] in N for moving solution."""
    basis = b.fem_2d.basis
    pfield = basis.interpolate(p_dofs)
    hfield = basis.interpolate(h_dofs)

    @LinearForm
    def shear_x(v, w):
        h = w.h
        return ((b.mu * b.u[0]) / h + 0.5 * h * grad(w.p)[0]) * v

    @LinearForm
    def shear_y(v, w):
        h = w.h
        return ((b.mu * b.u[1]) / h + 0.5 * h * grad(w.p)[1]) * v

    # fx = asm(shear_x, b.basis, p=pfield, h=hfield).sum()
    # fy = asm(shear_y, b.basis, p=pfield, h=hfield).sum()
    fx = shear_x.assemble(basis, p=pfield, h=hfield).sum()
    fy = shear_y.assemble(basis, p=pfield, h=hfield).sum()
    return np.array([fx, fy], dtype=float)


def _postprocess_2d_outputs(b, p, w, *, include_shear):
    """Compute common 2D postprocessed outputs from nodal pressure fields."""
    k = get_stiffness(bearing=b, w=w)
    moment = np.array([_evaluate_pressure_moment_2d(b, p[i]) for i in range(b.nh)])

    shear_force = None
    if include_shear:
        shear_force = np.array(
            [
                _evaluate_shear_force_2d(
                    b,
                    p[i],
                    b.h_func_2d(
                        b.fem_2d.basis.doflocs[0], b.fem_2d.basis.doflocs[1], i
                    ),
                )
                for i in range(b.nh)
            ]
        )

    p_list = list(p)
    return k, moment, shear_force, p_list


def solve_bearing_fem_2d_nonlinear(
    b,
    *,
    max_iter=10,
    tol=1.0,
    relaxation=0.9,
):
    """
    Solve 2D FEM pressure/load/flow curves for circular, annular, or rectangular cases
    using a nonlinear Newton solver.

    This solver correctly handles the sliding velocity terms (b.u), which are nonlinear
    in the pressure formulation.
    """
    fem_2d = b.fem_2d
    eta_dirichlet = fem_2d.eta
    p_dirichlet = fem_2d.p
    boundary_sets = fem_2d.boundary_sets

    def solve_one_h(i):
        def h_func(w):
            return b.h_func_2d(w.x[0], w.x[1], i)

        p_init, system_matrix = _solve_stationary_pressure_dofs_2d(
            b=b,
            h_func=h_func,
            eta_dirichlet=eta_dirichlet,
            boundary_dofs=fem_2d.D,
        )

        @NonlinearForm
        def reynolds(p, v, w):
            h = h_func(w)
            return (
                -1.0
                / (12.0 * b.mu)
                * p
                * h**3
                * (p.grad[0] * v.grad[0] + p.grad[1] * v.grad[1])
                + 0.5 * p * h * (b.u[0] * v.grad[0] + b.u[1] * v.grad[1])
                - b.kappa / (2.0 * b.mu * b.hp) * p**2 * v
                + b.kappa / (2.0 * b.mu * b.hp) * b.ps**2 * v
            )

        p = p_init.copy()

        D = fem_2d.D
        p[D] = p_dirichlet[D]

        for _ in range(max_iter):
            p_prev = p.copy()
            jacobian, rhs = reynolds.assemble(fem_2d.basis, x=p)
            p += relaxation * solve(*condense(jacobian, rhs, D=D))
            norm = np.linalg.norm(p - p_prev)
            if norm < tol:
                break

        w_i, qa_i, qc_i, qs_i = _evaluate_2d_result_terms(
            b=b,
            p_dofs=p,
            system_matrix=system_matrix,
            boundary_sets=boundary_sets,
        )
        return p, w_i, qa_i, qc_i, qs_i

    p, w, qa, qc, qs = _sweep_over_h(b.nh, solve_one_h)
    k, moment, shear_force, p = _postprocess_2d_outputs(
        b,
        p,
        w,
        include_shear=True,
    )

    name = "numeric 2d nonlinear"
    return Result(
        name=name,
        w=w,
        k=k,
        qs=qs,
        qa=qa,
        qc=qc,
        p_2d=p,
        moment=moment,
        shear_force=shear_force,
    )
