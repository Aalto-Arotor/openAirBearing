import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import openairbearing as ab

"""
Example comparing experimental results to model results.
This example reproduces Figures 5 and 6 of referenece [1]. 
Experimental data is included as csv files.

[1] M. Miettinen, V. Vainio, R. Theska, R. Viitala, On the static performance of
    aerostatic elements, Precision Engineering 89 (2024) 1-10. 
    doi:10.1016/j.precisioneng.2024.05.017.
"""

EXPERIMENT_DIR = Path(__file__).with_name("ex05_experiment_data")
SERIES_COLORS = {0.6: "black", 0.4: "red", 0.2: "blue"}
CSV_FILES = {
    "w": ("bearing_w_*.csv", "h_um", "w_n"),
    "k": ("bearing_k_*.csv", "h_um", "k_n_per_um"),
    "qa": ("bearing_q_*.csv", "h_um", "qa_lpm"),
    "p": ("bearing_p_*.csv", "x_mm", "p_mpa"),
}


def _series_label(ps_mpa, *, prefix):
    return f"{prefix} ps={ps_mpa:.1f} MPa"


def _parse_ps_from_path(path):
    ps_text = path.stem.split("_")[-1].removesuffix("MPa")
    return float(ps_text)


def _load_series_csv(pattern, x_key, y_key):
    series = {}
    for path in sorted(EXPERIMENT_DIR.glob(pattern), reverse=True):
        x_values = []
        y_values = []
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                x_values.append(float(row[x_key]))
                y_values.append(float(row[y_key]))

        ps_mpa = _parse_ps_from_path(path)
        series[ps_mpa] = {
            x_key: np.asarray(x_values, dtype=float),
            y_key: np.asarray(y_values, dtype=float),
            "label": _series_label(ps_mpa, prefix="experiment"),
        }

    return series


def load_experimental_data():
    """Return experimental datasets grouped by supply pressure."""
    return {
        name: _load_series_csv(pattern, x_key, y_key)
        for name, (pattern, x_key, y_key) in CSV_FILES.items()
    }


def _overlay_series(ax, series_by_ps, x_key, y_key):
    for ps_mpa, series in series_by_ps.items():
        x = np.asarray(series.get(x_key, []), dtype=float)
        y = np.asarray(series.get(y_key, []), dtype=float)
        if x.size == 0 or y.size == 0 or x.size != y.size:
            continue
        ax.scatter(
            x,
            y,
            color=SERIES_COLORS.get(ps_mpa, "black"),
            marker="o",
            s=25,
            label=series.get("label", _series_label(ps_mpa, prefix="experiment")),
        )


def experimental_comparison_example():
    supply_pressures_mpa = (0.6, 0.4, 0.2)
    bearing_kwargs = dict(
        xa=36.83e-3 / 2,
        kappa=1.44e-15,
        nx=50,
        nh=57,
        ha_min=1e-6,
        ha_max=15e-6,
        error_type="none",
    )
    bearings = [
        ab.CircularBearing(ps=101325 + ps_mpa * 1e6, **bearing_kwargs)
        for ps_mpa in supply_pressures_mpa
    ]
    bearing = bearings[0]

    results = []
    for ps_mpa, series_bearing in zip(supply_pressures_mpa, bearings, strict=True):
        result = ab.solve_bearing_analytic(series_bearing)
        result.name = _series_label(ps_mpa, prefix="analytic")
        result.color = SERIES_COLORS[ps_mpa]
        result.linestyle = "solid"
        result.marker = "none"
        result.highlight_marker = "none"
        results.append(result)

    exp = load_experimental_data()
    pressure_plot_h = 6.0e-6
    p_index = int(np.argmin(np.abs(bearing.ha - pressure_plot_h)))
    fig_grid, axes = plt.subplots(2, 2, figsize=(10, 8))
    _overlay_series(axes[0, 0], exp["w"], "h_um", "w_n")
    ab.plot_load_capacity(bearing, results, ax=axes[0, 0])

    _overlay_series(axes[0, 1], exp["k"], "h_um", "k_n_per_um")
    ab.plot_stiffness(bearing, results, ax=axes[0, 1], legend=False)

    _overlay_series(axes[1, 0], exp["qa"], "h_um", "qa_lpm")
    ab.plot_ambient_flow_rate(bearing, results, ax=axes[1, 0], legend=False)

    _overlay_series(axes[1, 1], exp["p"], "x_mm", "p_mpa")
    ab.plot_pressure_1d(
        bearing,
        results,
        ax=axes[1, 1],
        legend=False,
        p_index=p_index,
    )

    fig_grid.tight_layout()

    plt.show()


if __name__ == "__main__":
    experimental_comparison_example()
