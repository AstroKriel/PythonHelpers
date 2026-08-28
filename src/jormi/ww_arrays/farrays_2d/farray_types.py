## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Any

## third-party
from numpy.typing import NDArray

## local
from jormi.ww_validation import validate_arrays

##
## === SHAPE VALIDATION
##


def ensure_2d_sarray(
    sarray_2d: NDArray[Any],
    *,
    param_name: str = "<sarray_2d>",
) -> None:
    """Ensure `sarray_2d` is a 2D scalar ndarray with shape (num_x0_cells, num_x1_cells)."""
    validate_arrays.ensure_dims(
        array=sarray_2d,
        param_name=param_name,
        num_dims=2,
    )


def ensure_2d_varray(
    varray_2d: NDArray[Any],
    *,
    param_name: str = "<varray_2d>",
) -> None:
    """Ensure `varray_2d` is a 3D vector ndarray with leading axis of length 2."""
    validate_arrays.ensure_dims(
        array=varray_2d,
        param_name=param_name,
        num_dims=3,
    )
    if varray_2d.shape[0] != 2:
        raise ValueError(
            f"`{param_name}` must have shape"
            f" (2, num_cells_x, num_cells_y);"
            f" got shape={varray_2d.shape}.",
        )


## } MODULE
