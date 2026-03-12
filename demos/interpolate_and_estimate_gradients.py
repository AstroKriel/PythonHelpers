## { SCRIPT

##
## === DEPENDENCIES
##

import numpy
from jormi.ww_plots import plot_manager
from jormi.ww_data import interpolate_data

##
## === HELPER FUNCTIONS
##


def generate_data(num_points):
    x_values = numpy.linspace(-5, 15, num=num_points, endpoint=True)
    y_exact = numpy.cos(-numpy.square(x_values) / 9.0)
    dydx_exact = -(2.0 / 9.0) * x_values * numpy.sin(numpy.square(x_values) / 9.0)
    d2ydx2_exact = - (2.0 / 9.0) * numpy.sin(numpy.square(x_values) / 9.0) \
                   - numpy.square((2.0 / 9.0) * x_values) * numpy.cos(numpy.square(x_values) / 9.0)
    return x_values, y_exact, dydx_exact, d2ydx2_exact


##
## === DEMO INTERPOLATING DATA
##


def main():
    x_values, y_values, dydx_exact, d2ydx2_exact = generate_data(25)
    x_interp = numpy.linspace(x_values.min(), x_values.max(), 50)
    fig, axs_grid = plot_manager.create_figure(num_rows=3, share_x=True, axis_shape=(5, 8))
    axs_list = list(numpy.squeeze(axs_grid))
    plot_style_approx = {
        "color": "black",
        "ls": "",
        "marker": "o",
        "ms": 10,
        "zorder": 5,
        "label": "raw data",
    }
    axs_list[0].plot(x_values, y_values, **plot_style_approx)
    axs_list[1].plot(x_values, dydx_exact, **plot_style_approx)
    axs_list[2].plot(x_values, d2ydx2_exact, **plot_style_approx)
    for (interp_method, color) in [
        ("linear", "red"),
        ("quadratic", "green"),
        ("cubic", "blue"),
    ]:
        _, y_interp = interpolate_data.interpolate_1d(
            x_values,
            y_values,
            x_interp,
            kind=interp_method,
        )
        dydx_interp = numpy.gradient(y_interp, x_interp)
        d2ydx2_interp = numpy.gradient(dydx_interp, x_interp)
        plot_style_approx = {
            "color": color,
            "ls": "-",
            "lw": 1.5,
            "marker": "o",
            "ms": 5,
            "zorder": 3,
            "label": f"interp1d ({interp_method})",
        }
        axs_list[0].plot(x_interp, y_interp, **plot_style_approx)
        axs_list[1].plot(x_interp, dydx_interp, **plot_style_approx)
        axs_list[2].plot(x_interp, d2ydx2_interp, **plot_style_approx)
    axs_list[0].set_ylabel("y-values")
    axs_list[1].set_ylabel("first derivatives")
    axs_list[2].set_ylabel("second derivatives")
    axs_list[2].set_xlabel("x-values")
    axs_list[1].axhline(y=0, ls="--", color="black", zorder=1)
    axs_list[2].axhline(y=0, ls="--", color="black", zorder=1)
    axs_list[1].legend(loc="upper left")
    plot_manager.save_figure(fig, "interpolate_and_estimate_gradients.png")


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
