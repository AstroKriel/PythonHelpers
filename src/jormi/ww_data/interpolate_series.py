## { MODULE

##
## === DEPENDENCIES
##

## third-party
import numpy
from typing import Any
from numpy.typing import NDArray
from scipy.interpolate import make_interp_spline as scipy_make_interp_spline

## local
from jormi.ww_data.series_types import DataSeries
from jormi.ww_io import manage_log
from jormi.ww_types import (
    check_arrays,
    check_types,
)

##
## === INTERPOLATION FUNCTIONS
##


def interpolate_1d(
    data_series: DataSeries,
    x_interp: NDArray[Any],
    spline_order: int = 3,
) -> DataSeries:
    """
    Interpolate a `DataSeries` at new x positions, clipping to the data domain.

    `data_series.x_values` must be monotonically increasing. Any `x_interp` points
    outside the data domain are dropped, with a hint logged. Raises if all
    points are out of bounds.

    Returns a new `DataSeries` of the in-bounds interpolated (x, y) values.
    """
    check_types.ensure_type(
        param=data_series,
        valid_types=DataSeries,
        param_name="data_series",
    )
    if spline_order not in (1, 2, 3):
        raise ValueError(f"`spline_order` must be 1, 2, or 3; got {spline_order!r}.")
    ## validate x_interp
    x_interp = numpy.asarray(x_interp, dtype=numpy.float64)
    check_arrays.ensure_nonempty(
        array=x_interp,
        param_name="x_interp",
    )
    check_arrays.ensure_1d(
        array=x_interp,
        param_name="x_interp",
    )
    check_arrays.ensure_finite(
        array=x_interp,
        param_name="x_interp",
    )
    ## require monotonically increasing x_values
    if not numpy.all(numpy.diff(data_series.x_values) > 0):
        raise ValueError("`data_series.x_values` must be monotonically increasing.")
    ## clip x_interp to the data domain
    x_min_data, x_max_data = data_series.x_bounds
    in_bounds_mask = (x_min_data <= x_interp) & (x_interp <= x_max_data)
    num_out_of_bounds = int(numpy.sum(~in_bounds_mask))
    if num_out_of_bounds == x_interp.size:
        raise ValueError(
            f"All `x_interp` points are outside the data domain [{x_min_data}, {x_max_data}].",
        )
    if num_out_of_bounds > 0:
        hint_text = f"Dropping {num_out_of_bounds} `x_interp` point(s) outside the data domain [{x_min_data}, {x_max_data}]."
        manage_log.log_hint(text=hint_text)
    ## interpolate
    interpolator = scipy_make_interp_spline(
        data_series.x_values,
        data_series.y_values,
        k=spline_order,
    )
    x_interp_in_bounds = x_interp[in_bounds_mask]
    return DataSeries(
        x_values=x_interp_in_bounds,
        y_values=interpolator(x_interp_in_bounds),
    )


## } MODULE
