"""Utility functions for bearing calculations and data containers.

Provides bearing property calculations (area, geometry, stiffness, flow rates)
and Result dataclass for storing solver outputs.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Result:
    """Container for solver outputs."""

    name: str
    w: np.ndarray
    k: np.ndarray
    qs: np.ndarray
    qa: np.ndarray
    qc: np.ndarray
    p: np.ndarray = None
    p_1d: list = None
    p_2d: list = None
    p_3d: list = None
    moment: np.ndarray = None
    shear_force: np.ndarray = None
    color: str = None


def get_area(bearing) -> float:
    """
    Calculate the bearing area.
    """
    b = bearing
    match b.case:
        case "circular":
            A = np.pi * b.xa**2
        case "annular":
            A = np.pi * (b.xa**2 - b.xc**2)
        case "infinite":
            A = b.xa
        case "rectangular":
            A = b.xa * b.ya
        case "journal":
            A = b.xa * b.ya
        case _:
            raise ValueError(f"Unknown case: {b.case}")
    return A


def get_geom_1d(bearing, *, x=None):
    """
    Calculate the geometry error height at a given position x,
    normalized so that min(geom) = 0 over the bearing domain.
    """
    b = bearing
    err = float(b.error)
    xa = float(b.xa)
    x = b.x if x is None else x

    def calc_raw(x):
        match b.error_type:
            case "none":
                return np.zeros_like(x)
            case "linear":
                return err * (1.0 - x / xa)
            case "quadratic":
                return err * (1.0 - (x / xa) ** 2)
            case _:
                return np.zeros_like(x)

    xs = np.array([np.min(b.x), np.max(b.x)])
    if xs[0] < 0 < xs[1]:
        xs.append(0)
    min_val = np.min(calc_raw(xs))

    return calc_raw(x) - min_val


def get_geom_2d(bearing, *, x, y):
    """
    Calculate geometry error height at position (x, y),
    normalized so that min(geom) = 0 over the bearing domain.

    Args:
        bearing: Bearing object with properties (error, xa, ya, csys, etc.)
        x: x-coordinate (or r for polar)
        y: y-coordinate (or theta for polar, default 0)

    Returns:
        np.ndarray or float: Geometry error height at (x, y).
    """
    b = bearing
    err = float(b.error)
    xa = float(b.xa)
    ya = b.ya if b.ya != 0 else b.xa

    def calc_raw(x, y):
        if b.csys == "polar":
            r = (x**2 + y**2) ** 0.5
            match b.error_type:
                case "none":
                    return np.zeros_like(x)
                case "linear":
                    return err * (1.0 - r / xa)
                case "quadratic":
                    return err * (1.0 - (r / xa) ** 2)
                case "saddle":
                    return err * 0.5 * (1.0 - (x / xa) ** 2 + (y / ya) ** 2)
                case "tiltx":
                    return err * 0.5 * (x / xa)
                case "tilty":
                    return err * 0.5 * (y / ya)
                case _:
                    raise ValueError(f"Unknown error type: {b.error_type}")

        elif b.csys == "cartesian":
            if b.case == "journal" and b.error_type in {"conicity", "misalignment"}:
                return get_geom_2d_journal(b, x=x, y=y)
            match b.error_type:
                case "none":
                    return np.zeros_like(x)
                case "linear":
                    return err * 2.0 * np.maximum((np.abs(x) / xa), (np.abs(y) / ya))
                case "quadratic":
                    return err * 4.0 * np.maximum((x / xa) ** 2, (y / ya) ** 2)
                case "saddle":
                    return err * 0.5 * (1.0 - (x / xa) ** 2 + (y / ya) ** 2)
                case "tiltx":
                    return err * 0.5 * (x / xa)
                case "tilty":
                    return err * 0.5 * (y / ya)
                case _:
                    raise ValueError(f"Unknown error type: {b.error_type}")
        else:
            raise ValueError(f"Unknown coordinate system: {b.csys}")

    geom_raw = calc_raw(x, y)
    if b.case == "journal" and b.error_type == "misalignment":
        return np.asarray(geom_raw, dtype=float)

    min_val = np.min(calc_raw(b.fem_2d.basis.doflocs[0], b.fem_2d.basis.doflocs[1]))
    return geom_raw - min_val


def get_geom_2d_journal(bearing, *, x, y):
    """Return journal-specific geometry error fields on an unwrapped surface.

    Supported journal error modes:
    - ``none``: no geometry error
        - ``conicity``: axial taper, where one end is narrower than the other
        - ``misalignment``: shaft-axis tilt about the journal centerline, producing
            opposite circumferential offsets at opposite axial ends
    """
    b = bearing
    err = float(b.error)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_span = float(np.max(b.x) - np.min(b.x))
    if np.isclose(x_span, 0.0):
        axial = np.zeros_like(x)
    else:
        x_mid = 0.5 * (float(np.max(b.x)) + float(np.min(b.x)))
        axial = (x - x_mid) / x_span

    y_span = float(np.max(b.y) - np.min(b.y))
    if np.isclose(y_span, 0.0):
        theta = np.zeros_like(y)
    else:
        theta = 2.0 * np.pi * (y - float(np.min(b.y))) / y_span

    match b.error_type:
        case "none":
            geom = np.zeros_like(x)
        case "conicity":
            geom = err * (axial + 0.5)
        case "misalignment":
            geom = err * axial * np.cos(theta)
        case _:
            geom = np.zeros_like(x)

    return np.asarray(geom, dtype=float)


def get_beta(bearing):
    """
    Calculate the porous feeding parameter.

    Returns:
        float: The porous feeding parameter, beta.
    """
    b = bearing
    beta = 6 * b.kappa * b.xa**2 / (b.hp * b.ha**3)
    return beta


def get_kappa(bearing):
    """
    Calculate the permeability.

    Returns:
        float: Permeability, kappa (m^2).
    """
    b = bearing

    if getattr(b, "blocked", False):
        kappa = (
            2 * b.Qsc / 6e4 * b.mu * b.hp * b.pa / (b.block_A * (b.psc**2 - b.pa**2))
        )
    else:
        kappa = 2 * b.Qsc / 6e4 * b.mu * b.hp * b.pa / (b.A * (b.psc**2 - b.pa**2))
    return round_to_sig_dig(kappa, 3)


def get_Qsc(bearing):
    """
    Calculate free-flow rate at supply pressure.

    Returns:
        float: Free flow rate Qsc (L/min).
    """
    b = bearing

    if b.blocked:
        Qsc = (
            b.kappa * 6e4 * b.block_A * (b.psc**2 - b.pa**2) / (2 * b.mu * b.hp * b.pa)
        )
    else:
        Qsc = b.kappa * 6e4 * b.A * (b.psc**2 - b.pa**2) / (2 * b.mu * b.hp * b.pa)
    return round_to_sig_dig(Qsc, 3)


def round_to_sig_dig(number, digits):
    """Round a scalar to the requested number of significant digits."""
    return np.round(number, -int(np.floor(np.log10(np.abs(number)))) + (digits - 1))


def get_dA(bearing) -> np.ndarray:
    """Return per-node integration area weights for the bearing coordinate system."""
    b = bearing
    match b.csys:
        case "polar":
            dA = np.pi * np.gradient(b.x**2)
            dA[[0, -1]] = dA[[0, -1]] / 2
        case "cartesian":
            dA = b.dx.copy()
            dA[[0, -1]] = dA[[0, -1]] / 2
        case _:
            raise ValueError("Error: invalid csys in dA calculation")
    return dA[:, None]


def get_load_capacity(bearing, p: np.ndarray) -> np.ndarray:
    """
    Calculate the load capacity of the bearing.

    Args:
        p (np.ndarray): Pressure distribution.

    Returns:
        np.ndarray: Load capacity versus film height.
    """
    b = bearing
    p_rel = p - b.pa
    dA = get_dA(b)
    w = np.sum(p_rel * dA, axis=(0))
    return w


def get_stiffness(bearing, w: np.ndarray) -> np.ndarray:
    """
    Calculate the stiffness.
    Args:
        bearing: Bearing instance containing geometry and properties
        w (numpy.ndarray): The load capacity array.

    Returns:
        np.ndarray: Stiffness curve computed as -dW/dh (N/µm).
    """
    b = bearing
    k = -np.gradient(w) / np.gradient(b.h) * 1e-6  # N per µm
    return k


def get_supply_flow(bearing, p: np.ndarray) -> np.ndarray:
    """Calculate supply flow rate into the bearing.

    Args:
        bearing: Bearing instance containing geometry and properties
        p (np.ndarray): Pressure distribution array

    Returns:
        np.ndarray: Supply flow rate (L/min)
    """
    b = bearing
    dA = get_dA(b)
    qs_per_A = b.kappa * 6e4 * (b.ps**2 - p**2) / (2 * b.mu * b.hp * b.pa)
    qs = np.sum(qs_per_A * dA, axis=(0))
    return qs


def get_volumetric_flow(bearing, p: np.ndarray) -> tuple:
    """
    Calculate volumetric flow rates through the bearing.

        Args:
            bearing: Bearing instance containing geometry and properties
            p (np.ndarray): Pressure distribution array

    Returns:
        tuple: (qs, qa, qc) where:
            - qs (np.ndarray): Supply flow rate (L/min)
            - qa (np.ndarray): Ambient flow rate (L/min)
            - qc (np.ndarray): Chamber flow rate (L/min)
    """
    b = bearing

    h = b.ha

    match b.csys:
        case "polar":
            q = (
                -6e4
                * h**3
                * np.gradient(p**2, axis=0)
                * np.pi
                * b.x[:, None]
                / (12 * b.mu * b.pa * b.dx[:, None])
            )
        case "cartesian":
            q = (
                -6e4
                * h**3
                * np.gradient(p**2, axis=0)
                / (24 * b.mu * b.pa * b.dx[:, None])
            )
        case _:
            raise ValueError("Invalid csys")

    qa = q[-1, :]
    qc = q[1, :]
    qs = get_supply_flow(b, p)
    return qs, qa, qc
