## { V-TEST

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path
from typing import Any

## third-party
import numpy

## local
from jormi import ww_lists
from jormi.ww_data import interpolate_series
from jormi.ww_data import series_types
from jormi.ww_io import manage_log
from jormi.ww_plots import manage_figure, style_figure

##
## === HELPER FUNCTIONS
##


def evaluate_model(
    x_values: numpy.ndarray[Any, numpy.dtype[Any]],
) -> numpy.ndarray[Any, numpy.dtype[Any]]:
    """Evaluate the test function: sin(2x) + cos(x)."""
    return numpy.sin(2.0 * x_values) + numpy.cos(x_values)


def measure_max_error(
    result: series_types.DataSeries,
) -> float:
    true_y_values = evaluate_model(result.x_values)
    return float(numpy.max(numpy.abs(result.y_values - true_y_values)))


def plot_order(
    *,
    panel: manage_figure.Panel,
    data_series: series_types.DataSeries,
    result: series_types.DataSeries,
    spline_order: int,
    order_index: int,
    num_orders: int,
) -> None:
    is_top_panel = order_index == 0
    is_bottom_panel = order_index == num_orders - 1
    panel.plot(
        result.x_values,
        result.y_values,
        color="red",
        label=f"spline order = {spline_order}",
    )
    panel.scatter(
        data_series.x_values,
        data_series.y_values,
        color="black",
        zorder=3,
        label="input data" if is_top_panel else None,
    )
    panel.plot(
        result.x_values,
        evaluate_model(result.x_values),
        color="black",
        ls="--",
        label="true f(x)" if is_top_panel else None,
    )
    panel.set_ylabel("y")
    panel.legend(
        fontsize=20,
        loc="upper right",
    )
    if is_bottom_panel:
        panel.set_xlabel("x")
    else:
        panel.tick_params(labelbottom=False)


##
## === INTERPOLATION ACCURACY TEST
##


class TestSeriesInterpolation:
    """
    Interpolate a coarsely-sampled known function and check that each spline order
    recovers it to within a max-error tolerance that tightens with order.
    """

    def __init__(
        self,
    ):
        ## sample parameters
        self.num_input_points: int = 15
        self.num_interp_points: int = 100
        ## spline orders to test
        self.spline_orders_to_test: list[int] = [1, 2, 3]
        ## pass criterion: tighter max-error tolerance for higher-order splines
        self.max_error_tols: dict[int, float] = {1: 2e-1, 2: 4e-2, 3: 2e-2}

    def run(
        self,
    ) -> None:
        data_series = self._generate_data_series()
        x_interp_values = numpy.linspace(
            data_series.x_bounds[0],
            data_series.x_bounds[1],
            self.num_interp_points,
        )
        num_orders = len(self.spline_orders_to_test)
        figure, panel_grid = manage_figure.create_figure(
            num_panel_rows=num_orders,
            num_panel_cols=1,
            share_x_axis=True,
        )
        failed_orders: list[str] = []
        for order_index, spline_order in enumerate(self.spline_orders_to_test):
            result = interpolate_series.interpolate_1d(
                data_series=data_series,
                x_interp=x_interp_values,
                spline_order=spline_order,
            )
            plot_order(
                panel=panel_grid[order_index, 0],
                data_series=data_series,
                result=result,
                spline_order=spline_order,
                order_index=order_index,
                num_orders=num_orders,
            )
            max_abs_error = measure_max_error(result)
            max_error_tol = self.max_error_tols[spline_order]
            if max_abs_error > max_error_tol:
                manage_log.log_outcome(
                    text=f"order={spline_order}: max error {max_abs_error:.2e} > tol {max_error_tol:.2e}",
                    outcome=manage_log.ActionOutcome.FAILURE,
                )
                failed_orders.append(str(spline_order))
            else:
                manage_log.log_outcome(
                    text=f"order={spline_order}: max error {max_abs_error:.2e}",
                    outcome=manage_log.ActionOutcome.SUCCESS,
                )
        ## always save even on failure
        figure_path = Path(__file__).parent / "interpolated_series.png"
        manage_figure.save_figure(
            figure=figure,
            figure_path=figure_path,
        )
        assert not failed_orders, (
            f"Test failed for spline orders: {ww_lists.as_string(elems=failed_orders)}"
        )
        manage_log.log_action(
            title="Series interpolation",
            outcome=manage_log.ActionOutcome.SUCCESS,
            message="All checks passed.",
        )

    def _generate_data_series(
        self,
    ) -> series_types.DataSeries:
        x_input_values = numpy.linspace(0.0, 2.0 * numpy.pi, self.num_input_points)
        y_input_values = evaluate_model(x_input_values)
        return series_types.DataSeries(
            x_values=x_input_values,
            y_values=y_input_values,
        )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    style_figure.set_figure_params()
    test = TestSeriesInterpolation()
    test.run()

## } V-TEST
