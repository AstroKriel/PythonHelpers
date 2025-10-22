## { MODULE

##
## === DEPENDENCIES
##

import numpy

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
        raise ValueError(f"Gradient order `{grad_order}` is invalid.")
    return valid_grad_orders[grad_order]


def second_order_centered_difference(
    sarray: numpy.ndarray,
    cell_width: float,
    grad_axis: int,
) -> numpy.ndarray:
    forward = -1
    backward = +1
    s_f = numpy.roll(sarray, int(1 * forward), axis=grad_axis)
    s_b = numpy.roll(sarray, int(1 * backward), axis=grad_axis)
    return (s_f - s_b) / (2.0 * cell_width)


def fourth_order_centered_difference(
    sarray: numpy.ndarray,
    cell_width: float,
    grad_axis: int,
) -> numpy.ndarray:
    forward = -1
    backward = +1
    s_f1 = numpy.roll(sarray, int(1 * forward), axis=grad_axis)
    s_f2 = numpy.roll(sarray, int(2 * forward), axis=grad_axis)
    s_b1 = numpy.roll(sarray, int(1 * backward), axis=grad_axis)
    s_b2 = numpy.roll(sarray, int(2 * backward), axis=grad_axis)
    return (-s_f2 + 8.0 * s_f1 - 8.0 * s_b1 + s_b2) / (12.0 * cell_width)


def sixth_order_centered_difference(
    sarray: numpy.ndarray,
    cell_width: float,
    grad_axis: int,
) -> numpy.ndarray:
    forward = -1
    backward = +1
    s_f1 = numpy.roll(sarray, int(1 * forward), axis=grad_axis)
    s_f2 = numpy.roll(sarray, int(2 * forward), axis=grad_axis)
    s_f3 = numpy.roll(sarray, int(3 * forward), axis=grad_axis)
    s_b1 = numpy.roll(sarray, int(1 * backward), axis=grad_axis)
    s_b2 = numpy.roll(sarray, int(2 * backward), axis=grad_axis)
    s_b3 = numpy.roll(sarray, int(3 * backward), axis=grad_axis)
    return -(s_f3 - 9.0 * s_f2 + 45.0 * s_f1 - 45.0 * s_b1 + 9.0 * s_b2 - s_b3) / (60.0 * cell_width)


## } MODULE
