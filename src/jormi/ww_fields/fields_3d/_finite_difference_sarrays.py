## { MODULE

##
## === DEPENDENCIES
##

import numpy

from jormi.ww_types import type_checks
from jormi.ww_fields.fields_3d import _fdata_type

##
## === INTERNAL HELPERS
##


def _validate_args(
    *,
    sarray_3d: numpy.ndarray,
    cell_width: float,
    grad_axis: int,
    sarray_name: str,
) -> None:
    _fdata_type.ensure_3d_sarray(
        sarray_3d=sarray_3d,
        param_name=sarray_name,
    )
    type_checks.ensure_finite_float(
        param=cell_width,
        param_name="<cell_width>",
        allow_none=False,
        require_positive=True,
    )
    type_checks.ensure_finite_int(
        param=grad_axis,
        param_name="<grad_axis>",
        allow_none=False,
    )
    if grad_axis not in (0, 1, 2):
        raise ValueError(
            f"`<grad_axis>` must be one of (0, 1, 2); got grad_axis={grad_axis}.",
        )


##
## === FUNCTIONS
##


def get_grad_fn(
    grad_order: int,
):
    type_checks.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    valid_grad_orders = {
        2: second_order_centered_difference,
        4: fourth_order_centered_difference,
        6: sixth_order_centered_difference,
    }
    if grad_order not in valid_grad_orders:
        raise ValueError(f"Gradient order `{grad_order}` is unsupported.")
    return valid_grad_orders[grad_order]


def second_order_centered_difference(
    sarray_3d: numpy.ndarray,
    *,
    cell_width: float,
    grad_axis: int,
) -> numpy.ndarray:
    """Second-order centered finite difference on a 3D scalar array."""
    _validate_args(
        sarray_3d=sarray_3d,
        cell_width=cell_width,
        grad_axis=grad_axis,
        sarray_name="<sarray_3d>",
    )
    forward = -1
    backward = +1
    sarray_3d_f = numpy.roll(sarray_3d, int(1 * forward), axis=grad_axis)
    sarray_3d_b = numpy.roll(sarray_3d, int(1 * backward), axis=grad_axis)
    return (sarray_3d_f - sarray_3d_b) / (2.0 * cell_width)


def fourth_order_centered_difference(
    sarray_3d: numpy.ndarray,
    *,
    cell_width: float,
    grad_axis: int,
) -> numpy.ndarray:
    """Fourth-order centered finite difference on a 3D scalar array."""
    _validate_args(
        sarray_3d=sarray_3d,
        cell_width=cell_width,
        grad_axis=grad_axis,
        sarray_name="<sarray_3d>",
    )
    forward = -1
    backward = +1
    sarray_3d_f1 = numpy.roll(sarray_3d, int(1 * forward), axis=grad_axis)
    sarray_3d_f2 = numpy.roll(sarray_3d, int(2 * forward), axis=grad_axis)
    sarray_3d_b1 = numpy.roll(sarray_3d, int(1 * backward), axis=grad_axis)
    sarray_3d_b2 = numpy.roll(sarray_3d, int(2 * backward), axis=grad_axis)
    return (-sarray_3d_f2 + 8.0 * sarray_3d_f1 - 8.0 * sarray_3d_b1 + sarray_3d_b2) / (12.0 * cell_width)


def sixth_order_centered_difference(
    sarray_3d: numpy.ndarray,
    *,
    cell_width: float,
    grad_axis: int,
) -> numpy.ndarray:
    """Sixth-order centered finite difference on a 3D scalar array."""
    _validate_args(
        sarray_3d=sarray_3d,
        cell_width=cell_width,
        grad_axis=grad_axis,
        sarray_name="<sarray_3d>",
    )
    forward = -1
    backward = +1
    sarray_3d_f1 = numpy.roll(sarray_3d, int(1 * forward), axis=grad_axis)
    sarray_3d_f2 = numpy.roll(sarray_3d, int(2 * forward), axis=grad_axis)
    sarray_3d_f3 = numpy.roll(sarray_3d, int(3 * forward), axis=grad_axis)
    sarray_3d_b1 = numpy.roll(sarray_3d, int(1 * backward), axis=grad_axis)
    sarray_3d_b2 = numpy.roll(sarray_3d, int(2 * backward), axis=grad_axis)
    sarray_3d_b3 = numpy.roll(sarray_3d, int(3 * backward), axis=grad_axis)
    return -(
        sarray_3d_f3 - 9.0 * sarray_3d_f2 + 45.0 * sarray_3d_f1 - 45.0 * sarray_3d_b1 + 9.0 * sarray_3d_b2 -
        sarray_3d_b3
    ) / (60.0 * cell_width)


## } MODULE
