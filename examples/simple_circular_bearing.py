import openairbearing as ab

if __name__ == "__main__":

    # initialize circular bearing with parameters
    # 2 micrometer error with concave quadratic profile
    bearing = ab.CircularBearing(
        xa=40, Qsc=5, nx=50, nh=60, error_type="quadratic", error=-2e-6
    )

    # solve with analytic and numeric methods
    result = [
        ab.solve_bearing(bearing, "analytic"),
        ab.solve_bearing(bearing, "numeric"),
    ]

    # # plot bearing shape
    # figure = ab.plot_bearing_shape(bearing)
    # figure.show()

    # # plot results
    # figure = ab.plot_key_results(bearing, result)
    # figure.show()

    ab.plot_load_capacity(bearing, result).show()
    ab.plot_stiffness(bearing, result).show()
    ab.plot_pressure_distribution(bearing, result).show()
    ab.plot_supply_flow_rate(bearing, result).show()
    ab.plot_chamber_flow_rate(bearing, result).show()
    ab.plot_ambient_flow_rate(bearing, result).show()
