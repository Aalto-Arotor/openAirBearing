import matplotlib.pyplot as plt

import openairbearing as ab


def solve_case(error_type, error):
    bearing = ab.CircularBearing(
        xa=18e-3,
        Qsc=5,
        nx=40,
        nh=50,
        error_type=error_type,
        error=error,
    )
    result = ab.solve_bearing_fem_2d(bearing)
    result.name = f"numeric 2d ({error_type})"
    return bearing, result


def plot_geometry_error_subplots(cases):
    def _plot_case(case, *, ax):
        _, bearing, _ = case
        ab.plot_geometry_error(bearing, ax=ax)

    fig, _ = ab.plot_subplots(
        cases,
        _plot_case,
        n_cols=2,
        title_getter=lambda case: f"Geometry error: {case[0]}",
        figsize_per_subplot=(5.0, 4.5),
    )
    return fig


def circular_geometry_error_compare():
    error = 2.0e-6

    specs = [
        ("none", 0.0),
        ("quadratic", error),
        ("saddle", error),
        ("tiltx", error),
    ]

    cases = []

    for error_type, error in specs:
        bearing, result = solve_case(error_type=error_type, error=error)
        label = f"{error_type}"
        cases.append((label, bearing, result))
    results_list = [result for _, _, result in cases]
    ab.assign_result_styles(results_list)

    baseline_bearing = cases[0][1]

    figures = []
    figures.append(plot_geometry_error_subplots(cases))
    fig_perf, axes = plt.subplots(2, 2, figsize=(10, 8))

    ab.plot_load_capacity(
        baseline_bearing,
        results_list,
        ax=axes[0, 0],
    )
    ab.plot_stiffness(
        baseline_bearing,
        results_list,
        ax=axes[0, 1],
    )
    ab.plot_supply_flow_rate(
        baseline_bearing,
        results_list,
        ax=axes[1, 0],
    )
    ab.plot_moment(
        baseline_bearing,
        results_list,
        ax=axes[1, 1],
    )
    fig_perf.tight_layout()
    figures.append(fig_perf)

    plt.show()


if __name__ == "__main__":
    circular_geometry_error_compare()
