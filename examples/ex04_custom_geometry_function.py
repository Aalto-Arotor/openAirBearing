import matplotlib.pyplot as plt
import numpy as np

import openairbearing as ab


def custom_geom_2d(*, bearing, x, y):
    xa = bearing.xa
    ya = bearing.ya
    crown = 0.5e-6 * ((x / (0.5 * xa)) ** 2 + (y / (0.5 * ya)) ** 2)
    wavelength = 20e-3  # 5 mm
    ripple = (
        0.4e-6 * np.sin(2 * np.pi * x / wavelength) * np.cos(2 * np.pi * y / wavelength)
    )
    raw = crown + ripple
    return raw - np.min(raw)


def solve_default_case():
    bearing = ab.RectangularBearing(nx=28, ny=20, nh=22, error_type="none", error=0)
    result = ab.solve_bearing_fem_2d(bearing)
    result.name = "numeric 2d (flat geometry)"
    return bearing, result


def solve_custom_case():
    bearing = ab.RectangularBearing(
        nx=28, ny=20, nh=22, error_type="none", error=0, geom_func_2d=custom_geom_2d
    )
    result = ab.solve_bearing_fem_2d(bearing)
    result.name = "numeric 2d (custom geometry)"
    return bearing, result


def plot_bearing_shapes_subplots(cases):
    def _plot_xy(case, *, ax):
        bearing, _ = case
        ab.plot_xy_shape(bearing, ax=ax)

    def _plot_xz(case, *, ax):
        bearing, _ = case
        ab.plot_xz_shape(bearing, ax=ax)

    def _plot_geom(case, *, ax):
        bearing, _ = case
        ab.plot_geometry_error(bearing, ax=ax)

    def title(case):
        return case[1].name

    n_cols = len(cases)
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


def custom_geometry_example():
    default_bearing, default_result = solve_default_case()
    custom_bearing, custom_result = solve_custom_case()

    results = [default_result, custom_result]
    ab.assign_result_styles(results)

    figures = []

    fig_grid, axes = plt.subplots(2, 2, figsize=(10, 8))
    ab.plot_load_capacity(default_bearing, results, ax=axes[0, 0])
    ab.plot_stiffness(default_bearing, results, ax=axes[0, 1], legend=False)
    ab.plot_supply_flow_rate(default_bearing, results, ax=axes[1, 0], legend=False)
    ab.plot_ambient_flow_rate(default_bearing, results, ax=axes[1, 1], legend=False)
    fig_grid.tight_layout()
    figures.append(fig_grid)

    fig_panel2 = plt.figure(figsize=(12, 9))
    ax11 = fig_panel2.add_subplot(2, 2, 1)
    ax12 = fig_panel2.add_subplot(2, 2, 2)
    ax21 = fig_panel2.add_subplot(2, 2, 3)
    ax22 = fig_panel2.add_subplot(2, 2, 4)

    ab.plot_geometry_error(default_bearing, title="Default geometry", ax=ax11)
    ab.plot_geometry_error(custom_bearing, title="Custom geometry", ax=ax12)
    ab.plot_pressure_2d(
        default_bearing,
        [default_result],
        title="Default pressure contour",
        ax=ax21,
    )
    ab.plot_pressure_2d(
        custom_bearing,
        [custom_result],
        title="Custom pressure contour",
        ax=ax22,
    )
    fig_panel2.tight_layout()
    figures.append(fig_panel2)

    plt.show()


if __name__ == "__main__":
    custom_geometry_example()
