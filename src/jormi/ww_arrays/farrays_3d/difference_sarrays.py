## { MODULE

##
## === DEPENDENCIES
##

## third-party
from typing import Any

import numpy
from numpy.typing import NDArray

## local
from jormi.ww_arrays.farrays_3d import farray_types
from jormi.ww_validation import validate_types

##
## === INTERNAL HELPERS
##


def _ensure_args(
    *,
    sarray_3d: NDArray[Any],
    cell_width: float,
    grad_axis: int,
    sarray_name: str,
) -> None:
    farray_types.ensure_3d_sarray(
        sarray_3d=sarray_3d,
        param_name=sarray_name,
    )
    validate_types.ensure_finite_float(
        param=cell_width,
        param_name="<cell_width>",
        allow_none=False,
        require_positive=True,
    )
    validate_types.ensure_finite_int(
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
    """Return the centered finite-difference function for `grad_order` (2, 4, or 6)."""
    validate_types.ensure_finite_int(
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
        raise ValueError(f"gradient order `{grad_order}` is unsupported.")
    return valid_grad_orders[grad_order]


def second_order_centered_difference(
    sarray_3d: NDArray[Any],
    *,
    cell_width: float,
    grad_axis: int,
) -> NDArray[Any]:
    """Second-order centered finite difference on a 3D scalar array."""
    _ensure_args(
        sarray_3d=sarray_3d,
        cell_width=cell_width,
        grad_axis=grad_axis,
        sarray_name="<sarray_3d>",
    )
    forward = -1
    backward = +1
    sarray_3d_f = numpy.roll(
        a=sarray_3d,
        shift=int(1 * forward),
        axis=grad_axis,
    )
    sarray_3d_b = numpy.roll(
        a=sarray_3d,
        shift=int(1 * backward),
        axis=grad_axis,
    )
    return (sarray_3d_f - sarray_3d_b) / (2.0 * cell_width)


def fourth_order_centered_difference(
    sarray_3d: NDArray[Any],
    *,
    cell_width: float,
    grad_axis: int,
) -> NDArray[Any]:
    """Fourth-order centered finite difference on a 3D scalar array."""
    _ensure_args(
        sarray_3d=sarray_3d,
        cell_width=cell_width,
        grad_axis=grad_axis,
        sarray_name="<sarray_3d>",
    )
    forward = -1
    backward = +1
    sarray_3d_f1 = numpy.roll(
        a=sarray_3d,
        shift=int(1 * forward),
        axis=grad_axis,
    )
    sarray_3d_f2 = numpy.roll(
        a=sarray_3d,
        shift=int(2 * forward),
        axis=grad_axis,
    )
    sarray_3d_b1 = numpy.roll(
        a=sarray_3d,
        shift=int(1 * backward),
        axis=grad_axis,
    )
    sarray_3d_b2 = numpy.roll(
        a=sarray_3d,
        shift=int(2 * backward),
        axis=grad_axis,
    )
    return (-sarray_3d_f2 + 8.0 * sarray_3d_f1 - 8.0 * sarray_3d_b1 + sarray_3d_b2) / (
        12.0 * cell_width
    )


def sixth_order_centered_difference(
    sarray_3d: NDArray[Any],
    *,
    cell_width: float,
    grad_axis: int,
) -> NDArray[Any]:
    """Sixth-order centered finite difference on a 3D scalar array."""
    _ensure_args(
        sarray_3d=sarray_3d,
        cell_width=cell_width,
        grad_axis=grad_axis,
        sarray_name="<sarray_3d>",
    )
    forward = -1
    backward = +1
    out = numpy.zeros_like(
        sarray_3d, dtype=numpy.result_type(sarray_3d.dtype, numpy.float64)
    )
    sarray_3d_f3 = numpy.roll(
        a=sarray_3d,
        shift=int(3 * forward),
        axis=grad_axis,
    )
    out += sarray_3d_f3
    del sarray_3d_f3
    sarray_3d_f2 = numpy.roll(
        a=sarray_3d,
        shift=int(2 * forward),
        axis=grad_axis,
    )
    out -= 9.0 * sarray_3d_f2
    del sarray_3d_f2
    sarray_3d_f1 = numpy.roll(
        a=sarray_3d,
        shift=int(1 * forward),
        axis=grad_axis,
    )
    out += 45.0 * sarray_3d_f1
    del sarray_3d_f1
    sarray_3d_b1 = numpy.roll(
        a=sarray_3d,
        shift=int(1 * backward),
        axis=grad_axis,
    )
    out -= 45.0 * sarray_3d_b1
    del sarray_3d_b1
    sarray_3d_b2 = numpy.roll(
        a=sarray_3d,
        shift=int(2 * backward),
        axis=grad_axis,
    )
    out += 9.0 * sarray_3d_b2
    del sarray_3d_b2
    sarray_3d_b3 = numpy.roll(
        a=sarray_3d,
        shift=int(3 * backward),
        axis=grad_axis,
    )
    out -= sarray_3d_b3
    del sarray_3d_b3
    out /= 60.0 * cell_width
    return out


## } MODULE
