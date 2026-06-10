import matplotlib.pyplot as plt

import openairbearing as ab


def circular_baseline_example():
    bearing = ab.CircularBearing(
        xa=18e-3,
        Qsc=5,
        nx=100,
        nh=60,
        error_type="none",
        error=0,
    )

    results = [
        ab.solve_bearing_analytic(bearing),
        ab.solve_bearing_fem_1d(bearing),
        ab.solve_bearing_fem_2d(bearing),
    ]

    figures = []

    fig_grid, axes = plt.subplots(2, 2, figsize=(10, 8))
    ab.plot_load_capacity(bearing, results, ax=axes[0, 0])
    ab.plot_stiffness(bearing, results, ax=axes[0, 1])
    ab.plot_ambient_flow_rate(bearing, results, ax=axes[1, 0])
    ab.plot_pressure_1d(bearing, results, ax=axes[1, 1])

    fig_grid.tight_layout()
    figures.append(fig_grid)
    figures.append(ab.plot_pressure_2d(bearing, results))

    plt.show()


if __name__ == "__main__":
    circular_baseline_example()
