"""Analytical solutions for bearing pressure and performance.

Computes pressure distributions and bearing characteristics using closed-form equations.
"""

import numpy as np
from scipy.special import i0, k0

from openairbearing.utils import (
    Result,
    get_load_capacity,
    get_stiffness,
    get_volumetric_flow,
)


def solve_bearing_analytic(bearing) -> Result:
    """Solve bearing performance using closed-form pressure models by case."""
    match bearing.case:
        case "circular":
            p = get_pressure_analytic_circular(bearing)
        case "annular":
            p = get_pressure_analytic_annular(bearing)
        case "infinite":
            p = get_pressure_analytic_infinite(bearing)
        case _:
            return Result(
                name="none",
                p=np.array([]),
                w=np.array([]),
                k=np.array([]),
                qs=np.array([]),
                qa=np.array([]),
                qc=np.array([]),
            )
    w = get_load_capacity(bearing=bearing, p=p)
    k = get_stiffness(bearing=bearing, w=w)
    qs, qa, qc = get_volumetric_flow(bearing=bearing, p=p)

    name = "analytic"

    return Result(name=name, p=p, w=w, k=k, qs=qs, qa=qa, qc=qc)


def get_pressure_analytic_infinite(bearing):
    """Return analytic pressure field for infinitely long bearings/seals."""

    b = bearing

    f = (2 * b.beta) ** 0.5
    slip = (1 + b.Psi) ** 0.5

    # nondimensionals
    Pa = 1
    Ra = 1
    R = b.x / b.xa
    Ps = b.ps / b.pa
    Pc = b.pc / b.pa

    exp_f = np.exp((f * Ra) / slip)

    numer1 = -(Pc**2) + Ps**2 + exp_f * (Pa**2 - Ps**2)
    numer2 = exp_f * (-(Pa**2) + Ps**2 + exp_f * (Pc**2 - Ps**2))

    denom = -1 + np.exp((2 * f * Ra) / slip)

    C1 = numer1 / denom
    C2 = numer2 / denom

    p = (
        b.pa
        * (
            Ps**2
            + C1 * np.exp(np.outer(R, f) / slip)
            + C2 * np.exp(-np.outer(R, f) / slip)
        )
        ** 0.5
    )
    return p


def get_pressure_analytic_annular(bearing):
    """Return analytic pressure field for annular bearings using Bessel functions."""

    b = bearing

    f = (2 * b.beta) ** 0.5

    # nondimensionals
    Pa = 1
    Ra = 1
    R = b.x / b.xa
    Ps = b.ps / b.pa
    Pc = b.pc / b.pa
    Rc = b.xc / b.xa

    numer1 = (Pa**2 - Ps**2) * k0(f * Rc) + (Ps**2 - Pc**2) * k0(f * Ra)
    numer2 = (Pa**2 - Ps**2) * i0(f * Rc) + (Ps**2 - Pc**2) * i0(f * Ra)

    denom = i0(f * Rc) * k0(f * Ra) - i0(f * Ra) * k0(f * Rc)

    C1 = numer1 / denom
    C2 = numer2 / denom

    p = b.pa * (Ps**2 - C1 * i0(np.outer(R, f)) + C2 * k0(np.outer(R, f))) ** 0.5
    return p


def get_pressure_analytic_circular(bearing):
    """Return analytic pressure field for circular thrust bearings."""
    b = bearing
    p = (
        b.ps
        * (
            1
            - (1 - b.pa**2 / b.ps**2)
            * i0(np.outer(b.x / b.xa, (2 * b.beta) ** 0.5))
            / i0((2 * b.beta) ** 0.5)
        )
        ** 0.5
    )
    return p
