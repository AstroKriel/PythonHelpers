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


def ensure_1d_sarray(
    sarray_1d: NDArray[Any],
    *,
    param_name: str = "<sarray_1d>",
) -> None:
    """Ensure `sarray_1d` is a 1D scalar ndarray with shape (num_x0_cells,)."""
    validate_arrays.ensure_dims(
        array=sarray_1d,
        param_name=param_name,
        num_dims=1,
    )


def ensure_1d_varray(
    varray_1d: NDArray[Any],
    *,
    param_name: str = "<varray_1d>",
) -> None:
    """Ensure `varray_1d` is a 2D vector ndarray with leading axis of length 1."""
    validate_arrays.ensure_dims(
        array=varray_1d,
        param_name=param_name,
        num_dims=2,
    )
    if varray_1d.shape[0] != 1:
        raise ValueError(
            f"`{param_name}` must have shape"
            f" (1, num_cells_x);"
            f" got shape={varray_1d.shape}.",
        )


## } MODULE
