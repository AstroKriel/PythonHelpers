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
## === CONSTANTS
##

FORWARD = -1
BACKWARD = +1

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
    out_sarray_3d = numpy.zeros_like(
        sarray_3d,
        dtype=numpy.result_type(
            sarray_3d.dtype,
            numpy.float64,
        ),
    )
    f_sarray_3d = numpy.roll(
        a=sarray_3d,
        shift=int(1 * FORWARD),
        axis=grad_axis,
    )
    out_sarray_3d += f_sarray_3d
    del f_sarray_3d
    b_sarray_3d = numpy.roll(
        a=sarray_3d,
        shift=int(1 * BACKWARD),
        axis=grad_axis,
    )
    out_sarray_3d -= b_sarray_3d
    del b_sarray_3d
    out_sarray_3d /= 2.0 * cell_width
    return out_sarray_3d


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
    out_sarray_3d = numpy.zeros_like(
        sarray_3d,
        dtype=numpy.result_type(
            sarray_3d.dtype,
            numpy.float64,
        ),
    )
    f2_sarray_3d = numpy.roll(
        a=sarray_3d,
        shift=int(2 * FORWARD),
        axis=grad_axis,
    )
    out_sarray_3d -= f2_sarray_3d
    del f2_sarray_3d
    f1_sarray_3d = numpy.roll(
        a=sarray_3d,
        shift=int(1 * FORWARD),
        axis=grad_axis,
    )
    out_sarray_3d += 8.0 * f1_sarray_3d
    del f1_sarray_3d
    b1_sarray_3d = numpy.roll(
        a=sarray_3d,
        shift=int(1 * BACKWARD),
        axis=grad_axis,
    )
    out_sarray_3d -= 8.0 * b1_sarray_3d
    del b1_sarray_3d
    b2_sarray_3d = numpy.roll(
        a=sarray_3d,
        shift=int(2 * BACKWARD),
        axis=grad_axis,
    )
    out_sarray_3d += b2_sarray_3d
    del b2_sarray_3d
    out_sarray_3d /= 12.0 * cell_width
    return out_sarray_3d


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
    out_sarray_3d = numpy.zeros_like(
        sarray_3d,
        dtype=numpy.result_type(
            sarray_3d.dtype,
            numpy.float64,
        ),
    )
    f3_sarray_3d = numpy.roll(
        a=sarray_3d,
        shift=int(3 * FORWARD),
        axis=grad_axis,
    )
    out_sarray_3d += f3_sarray_3d
    del f3_sarray_3d
    f2_sarray_3d = numpy.roll(
        a=sarray_3d,
        shift=int(2 * FORWARD),
        axis=grad_axis,
    )
    out_sarray_3d -= 9.0 * f2_sarray_3d
    del f2_sarray_3d
    f1_sarray_3d = numpy.roll(
        a=sarray_3d,
        shift=int(1 * FORWARD),
        axis=grad_axis,
    )
    out_sarray_3d += 45.0 * f1_sarray_3d
    del f1_sarray_3d
    b1_sarray_3d = numpy.roll(
        a=sarray_3d,
        shift=int(1 * BACKWARD),
        axis=grad_axis,
    )
    out_sarray_3d -= 45.0 * b1_sarray_3d
    del b1_sarray_3d
    b2_sarray_3d = numpy.roll(
        a=sarray_3d,
        shift=int(2 * BACKWARD),
        axis=grad_axis,
    )
    out_sarray_3d += 9.0 * b2_sarray_3d
    del b2_sarray_3d
    b3_sarray_3d = numpy.roll(
        a=sarray_3d,
        shift=int(3 * BACKWARD),
        axis=grad_axis,
    )
    out_sarray_3d -= b3_sarray_3d
    del b3_sarray_3d
    out_sarray_3d /= 60.0 * cell_width
    return out_sarray_3d


## } MODULE
