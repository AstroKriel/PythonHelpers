## { MODULE

##
## === DEPENDENCIES
##

import numpy

from jormi.ww_fields.fields_3d import fdata_types

##
## === FUNCTIONS
##


def get_grad_fn(
    grad_order: int,
):
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
    forward = -1
    backward = +1
    fdata_types.ensure_3d_sarray(sarray_3d)
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
    forward = -1
    backward = +1
    fdata_types.ensure_3d_sarray(sarray_3d)
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
    forward = -1
    backward = +1
    fdata_types.ensure_3d_sarray(sarray_3d)
    sarray_3d_f1 = numpy.roll(sarray_3d, int(1 * forward), axis=grad_axis)
    sarray_3d_f2 = numpy.roll(sarray_3d, int(2 * forward), axis=grad_axis)
    sarray_3d_f3 = numpy.roll(sarray_3d, int(3 * forward), axis=grad_axis)
    sarray_3d_b1 = numpy.roll(sarray_3d, int(1 * backward), axis=grad_axis)
    sarray_3d_b2 = numpy.roll(sarray_3d, int(2 * backward), axis=grad_axis)
    sarray_3d_b3 = numpy.roll(sarray_3d, int(3 * backward), axis=grad_axis)
    return -(sarray_3d_f3 - 9.0 * sarray_3d_f2 + 45.0 * sarray_3d_f1 - 45.0 * sarray_3d_b1 + 9.0 * sarray_3d_b2 - sarray_3d_b3) / (60.0 * cell_width)


## } MODULE
