## { MODULE

##
## === DEPENDENCIES
##

import numpy

from scipy.interpolate import make_interp_spline as scipy_make_interp_spline

from jormi.utils import list_utils
from jormi.ww_types import array_checks
from jormi.ww_data import data_series

##
## === INTERPOLATION FUNCTIONS
##

_KIND_TO_SPLINE_ORDER: dict[str, int] = {
    "linear": 1,
    "quadratic": 2,
    "cubic": 3,
}


def interpolate_1d(
    series: data_series.DataSeries,
    x_interp: numpy.ndarray,
    kind: str = "cubic",
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Interpolate a `DataSeries` at new x positions, clipping to the data domain.

    `series.x_values` must be monotonically increasing. Any `x_interp` points
    outside the data domain are dropped, with a hint logged.
    Returns `(x_interp_in_bounds, y_interp_in_bounds)`.
    """
    ## validate interpolation kind
    if kind not in _KIND_TO_SPLINE_ORDER:
        valid_kinds_string = list_utils.as_string(
            elems=list(_KIND_TO_SPLINE_ORDER.keys()),
            wrap_in_quotes=True,
            conjunction="",
        )
        raise ValueError(
            f"Invalid interpolation `kind`: {kind}. Valid options include: {valid_kinds_string}",
        )
    ## validate x_interp
    x_interp = numpy.asarray(x_interp, dtype=numpy.float64)
    array_checks.ensure_1d(x_interp, param_name="x_interp")
    array_checks.ensure_finite(x_interp, param_name="x_interp")
    ## require monotonically increasing x_values
    if not numpy.all(numpy.diff(series.x_values) > 0):
        raise ValueError("`series.x_values` must be monotonically increasing.")
    ## clip x_interp to the data domain
    x_min_data, x_max_data = series.x_bounds
    in_bounds_mask = (x_min_data <= x_interp) & (x_interp <= x_max_data)
    num_out_of_bounds = int(numpy.sum(~in_bounds_mask))
    if num_out_of_bounds > 0:
        from jormi.ww_io import log_manager
        log_manager.log_hint(
            text=
            f"Dropping {num_out_of_bounds} `x_interp` point(s) outside the data domain [{x_min_data}, {x_max_data}].",
        )
    ## interpolate
    spline_order = _KIND_TO_SPLINE_ORDER[kind]
    interpolator = scipy_make_interp_spline(
        series.x_values,
        series.y_values,
        k=spline_order,
    )
    x_interp_in_bounds = x_interp[in_bounds_mask]
    y_interp_in_bounds = interpolator(x_interp_in_bounds)
    return (x_interp_in_bounds, y_interp_in_bounds)


## } MODULE
