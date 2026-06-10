"""Example 06 -- Circular bearing tilt sweep with custom gap height function.

Demonstrates the h_func_2d mechanism: instead of sweeping air-gap height,
we keep a constant 5 um gap and sweep the tilt angle of the bearing surface.

    h(x, y, i) = h0 + tilt_angles[i] * x + geom(x, y)

The tilt is about the y-axis so the gap varies linearly with the
x-coordinate of the circular mesh.
"""

import matplotlib.pyplot as plt
import numpy as np

import openairbearing as ab


def urad_to_deg(x):
    return np.degrees(x) * 1e-6


def deg_to_urad(x):
    return np.radians(x) * 1e6


def tilt_sweep_example():
    h0 = 5e-6  # constant mean gap height
    n_tilt = 200
    xa = 37e-3 / 2
    alpha_max = np.asin(4e-6 / xa)
    tilt_angles = np.linspace(-alpha_max, alpha_max, n_tilt)

    bearing = ab.CircularBearing(
        xa=xa,
        nh=n_tilt,
        h=tilt_angles,
        ha=h0,
        error_type="none",
        error=0,
    )

    # attach tilt schedule to the bearing instance
    bearing.h0 = h0
    bearing.tilt_angles = tilt_angles

    # custom gap height: h = h0 + tilt * x + geom
    def h_func_tilt(x, y, i):
        h_base = bearing.ha
        h_tilt = np.sin(bearing.h[i]) * x
        h_geom = bearing.geom_func_2d(bearing=bearing, x=x, y=y)
        return h_base + h_tilt + h_geom

    bearing.h_func_2d = h_func_tilt

    result = ab.solve_bearing_fem_2d(bearing)
    gaps = []
    for i in bearing.h_iter:
        gaps.append(
            bearing.h_func_2d(
                x=bearing.fem_2d.basis.doflocs[0],
                y=bearing.fem_2d.basis.doflocs[1],
                i=i,
            )
        )

    # -- plotting ---------------------------------------------------------
    tilt_urad = tilt_angles * 1e6  # micro-radians for display

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # load capacity
    axes[0, 0].plot(tilt_urad, result.w, "r")
    axes[0, 0].set_xlabel("Tilt angle (μrad)")
    axes[0, 0].set_ylabel("Load capacity (N)")

    # stiffness (dW / d_alpha)
    dMda = (
        -np.gradient(
            result.moment[:, 0],
        )
        / np.gradient(bearing.h)
        * 1e-6
    )
    axes[0, 1].plot(tilt_urad, dMda, "r")
    axes[0, 1].set_xlabel("Tilt angle (μrad)")
    axes[0, 1].set_ylabel("Tilt stiffness (N/μrad)")

    # ambient flow
    axes[1, 0].plot(tilt_urad, result.qa, "r")
    axes[1, 0].set_xlabel("Tilt angle (μrad)")
    axes[1, 0].set_ylabel("Ambient flow (L/min)")

    # moment components
    axes[1, 1].plot(tilt_urad, result.moment[:, 0], "b", label="Mx")
    axes[1, 1].plot(tilt_urad, result.moment[:, 1], "r", label="My")
    axes[1, 1].set_xlabel("Tilt angle (μrad)")
    axes[1, 1].set_ylabel("Moment (N m)")
    axes[1, 1].legend()

    for ax in axes.ravel():
        ax_deg = ax.secondary_xaxis("top", functions=(urad_to_deg, deg_to_urad))
        ax_deg.set_xlabel("Tilt angle (deg)")

    fig.tight_layout()

    # pressure contours: zero tilt vs max tilt
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 5))
    surf = axes2[0, 0].tricontourf(
        bearing.fem_2d.basis.doflocs[0] * 1e3,
        bearing.fem_2d.basis.doflocs[1] * 1e3,
        result.p_2d[0] * 1e-6,
        levels=30,
        cmap="jet",
    )
    axes2[0, 0].set_aspect("equal")
    axes2[0, 0].set_xlabel("x (mm)")
    axes2[0, 0].set_ylabel("y (mm)")
    fig2.colorbar(surf, ax=axes2[0, 0], pad=0.1, shrink=0.7, label="p (MPa)")

    surf = axes2[0, 1].tricontourf(
        bearing.fem_2d.basis.doflocs[0] * 1e3,
        bearing.fem_2d.basis.doflocs[1] * 1e3,
        result.p_2d[-1] * 1e-6,
        levels=30,
        cmap="jet",
    )
    axes2[0, 1].set_aspect("equal")
    axes2[0, 1].set_xlabel("x (mm)")
    axes2[0, 1].set_ylabel("y (mm)")
    fig2.colorbar(surf, ax=axes2[0, 1], pad=0.1, shrink=0.7, label="p (MPa)")

    surf = axes2[1, 0].tricontourf(
        bearing.fem_2d.basis.doflocs[0] * 1e3,
        bearing.fem_2d.basis.doflocs[1] * 1e3,
        gaps[0] * 1e6,
        levels=30,
        cmap="jet",
    )
    axes2[1, 0].set_aspect("equal")
    axes2[1, 0].set_xlabel("x (mm)")
    axes2[1, 0].set_ylabel("y (mm)")
    fig2.colorbar(surf, ax=axes2[1, 0], pad=0.1, shrink=0.7, label="h (μm)")

    surf = axes2[1, 1].tricontourf(
        bearing.fem_2d.basis.doflocs[0] * 1e3,
        bearing.fem_2d.basis.doflocs[1] * 1e3,
        gaps[-1] * 1e6,
        levels=30,
        cmap="jet",
    )
    axes2[1, 1].set_aspect("equal")
    axes2[1, 1].set_xlabel("x (mm)")
    axes2[1, 1].set_ylabel("y (mm)")
    fig2.colorbar(surf, ax=axes2[1, 1], pad=0.1, shrink=0.7, label="h (μm)")

    for ax in axes2.ravel():
        ax_deg = ax.secondary_xaxis("top", functions=(urad_to_deg, deg_to_urad))
        ax_deg.set_xlabel("Tilt angle (deg)")
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    tilt_sweep_example()
