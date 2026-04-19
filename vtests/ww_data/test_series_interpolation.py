## { TEST

##
## === DEPENDENCIES
##

## stdlib
from typing import Any

## third-party
import numpy

## local
from jormi.ww_data import interpolate_series
from jormi.ww_data.series_types import DataSeries
from jormi.ww_io import manage_io
from jormi.ww_plots import manage_plots

##
## === HELPER FUNCTIONS
##


def evaluate_function(
    x_values: numpy.ndarray[Any, numpy.dtype[Any]],
) -> numpy.ndarray[Any, numpy.dtype[Any]]:
    """Evaluate the test function: sin(2x) + cos(x)."""
    return numpy.sin(2.0 * x_values) + numpy.cos(x_values)


##
## === INTERPOLATION ACCURACY TEST
##


def main():
    ## parameters: known function sin(2x) + cos(x) sampled at a coarse grid
    num_input_points = 15
    num_interp_points = 100
    spline_orders_to_test = [
        1,
        2,
        3,
    ]
    ## tighter tolerance expected for higher-order splines
    max_error_tols = {
        1: 2e-1,
        2: 4e-2,
        3: 2e-2,
    }
    ## construct coarse dataset over [0, 2*pi]
    x_input_values = numpy.linspace(0.0, 2.0 * numpy.pi, num_input_points)
    y_input_values = evaluate_function(x_input_values)
    series = DataSeries(
        x_values=x_input_values,
        y_values=y_input_values,
    )
    ## dense in-bounds interpolation grid
    x_interp_values = numpy.linspace(
        x_input_values[0],
        x_input_values[-1],
        num_interp_points,
    )
    ## plot: column stack, one row per spline order, shared x axis
    num_orders = len(spline_orders_to_test)
    fig, axs_grid = manage_plots.create_figure(
        num_rows=num_orders,
        num_cols=1,
        share_x=True,
    )
    orders_that_failed = []
    for order_index, spline_order in enumerate(spline_orders_to_test):
        ax = axs_grid[order_index, 0]
        is_top_ax = order_index == 0
        is_bottom_ax = order_index == num_orders - 1
        result = interpolate_series.interpolate_1d(
            data_series=series,
            x_interp=x_interp_values,
            spline_order=spline_order,
        )
        ## check: max abs error vs true function
        max_error_tol = max_error_tols[spline_order]
        true_y_values = evaluate_function(result.x_values)
        max_abs_error = float(
            numpy.max(
                numpy.abs(
                    result.y_values - true_y_values,
                ),
            ),
        )
        ## plot interpolated result, input data, and true function
        ax.plot(
            result.x_values,
            result.y_values,
            color="red",
            label=f"spline order = {spline_order}",
        )
        ax.scatter(
            x_input_values,
            y_input_values,
            color="black",
            zorder=3,
            label="input data" if is_top_ax else None,
        )
        ax.plot(
            result.x_values,
            true_y_values,
            color="black",
            ls="--",
            label="true f(x)" if is_top_ax else None,
        )
        ax.set_ylabel("y")
        ax.legend(
            fontsize=20,
            loc="upper right",
        )
        if is_bottom_ax:
            ax.set_xlabel("x")
        else:
            ax.tick_params(labelbottom=False)
        if max_abs_error > max_error_tol:
            print(
                f"Failed: order={spline_order} - max error {max_abs_error:.2e} > max_error_tol {max_error_tol:.2e}",
            )
            orders_that_failed.append(spline_order)
        else:
            print(f"Passed: order={spline_order} - max error {max_abs_error:.2e}")
    ## save figure always so it can be inspected on failure
    file_dir = manage_io.get_caller_directory()
    fig_name = "interpolated_series.png"
    fig_path = file_dir / fig_name
    manage_plots.save_figure(fig, fig_path)
    assert len(orders_that_failed) == 0, (f"Test failed for spline orders: {orders_that_failed}")
    print("All tests passed successfully!")


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } TEST
