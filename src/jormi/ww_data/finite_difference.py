## { MODULE

##
## === DEPENDENCIES
##

import numpy

##
## === FUNCTIONS
##


def second_order_centered_difference(
    sarray: numpy.ndarray,
    cell_width: float,
    grad_axis: int,
) -> numpy.ndarray:
    forward = -1
    backward = +1
    s_f = numpy.roll(sarray, int(1 * forward), axis=grad_axis)
    s_b = numpy.roll(sarray, int(1 * backward), axis=grad_axis)
    return (s_f - s_b) / (2 * cell_width)


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
    return (-s_f2 + 8 * s_f1 - 8 * s_b1 + s_b2) / (12 * cell_width)


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
    return -(s_f3 - 9 * s_f2 + 45 * s_f1 - 45 * s_b1 + 9 * s_b2 - s_b3) / (60 * cell_width)


## } MODULE
