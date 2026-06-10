import matplotlib.pyplot as plt
import numpy as np

import openairbearing as ab


def solve_velocity_case(vx_mps):
    bearing = ab.RectangularBearing(
        nx=24,
        ny=16,
        nh=20,
        error_type="none",
        error=0,
        u=np.array([vx_mps, 0.0]),
    )
    result = ab.solve_bearing_fem_2d_nonlinear(
        bearing,
        max_iter=10,
        tol=1e-1,
        relaxation=0.7,
    )
    result.name = f"numeric 2d nonlinear ({vx_mps:.0f} m/s)"
    return bearing, result


def summarize(cases):
    print("Rectangular velocity comparison:")
    for velocity, _, result in cases:
        max_shear = np.linalg.norm(result.shear_force, axis=1).max()
        print(
            f"u={velocity:>4.1f} m/s | "
            f"max load={result.w.max():.3f} N | "
            f"max shear={max_shear:.3f} N"
        )


def plot_bearing_shapes_subplots(cases):
    def _plot_xy(case, *, ax):
        _, bearing, _ = case
        ab.plot_xy_shape(bearing, ax=ax)

    def _plot_xz(case, *, ax):
        _, bearing, _ = case
        ab.plot_xz_shape(bearing, ax=ax)

    def _plot_geom(case, *, ax):
        _, bearing, _ = case
        ab.plot_geometry_error(bearing, ax=ax)

    def title(case):
        return f"{case[0]:.1f} m/s"

    n_cols = 2
    figs = []
    fig, _ = ab.plot_subplots(cases, _plot_xy, n_cols=n_cols, title_getter=title)
    figs.append(fig)
    fig, _ = ab.plot_subplots(cases, _plot_xz, n_cols=n_cols, title_getter=title)
    figs.append(fig)
    fig, _ = ab.plot_subplots(
        cases,
        _plot_geom,
        n_cols=n_cols,
        title_getter=title,
        figsize_per_subplot=(5.0, 4.5),
    )
    figs.append(fig)
    return figs


def rectangular_velocity_compare():
    velocities = [0.0, 5.0, 10.0]
    cases = []
    for v in velocities:
        bearing, result = solve_velocity_case(v)
        cases.append((v, bearing, result))

    summarize(cases)

    reference_bearing = cases[0][1]
    results = [result for _, _, result in cases]
    ab.assign_result_styles(results)

    figures = []

    # Performance metrics and pressure contours grid (3x2)
    fig_perf, axes = plt.subplots(3, 2, figsize=(10, 12))
    ab.plot_load_capacity(reference_bearing, results, ax=axes[0, 0])
    ab.plot_stiffness(reference_bearing, results, ax=axes[0, 1])
    ab.plot_supply_flow_rate(reference_bearing, results, ax=axes[1, 0])
    ab.plot_moment(reference_bearing, results, ax=axes[1, 1])
    ab.plot_pressure_2d(
        reference_bearing,
        [cases[0][2]],
        title="Pressure distribution (0 m/s)",
        ax=axes[2, 0],
    )
    ab.plot_pressure_2d(
        reference_bearing,
        [cases[-1][2]],
        title=f"Pressure distribution ({velocities[-1]:.0f} m/s)",
        ax=axes[2, 1],
    )
    fig_perf.tight_layout()
    figures.append(fig_perf)

    plt.show()


if __name__ == "__main__":
    rectangular_velocity_compare()
