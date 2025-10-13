## { MODULE

##
## === DEPENDENCIES
##

import numpy
from jormi.utils import type_utils

##
## === DATA TYPE VALIDATION
##


def ensure_numpy_array(
    array: numpy.ndarray,
):
    type_utils.assert_type(
        var_obj=array,
        valid_types=numpy.ndarray,
    )


def ensure_sarray(
    sarray: numpy.ndarray,
) -> None:
    ensure_numpy_array(sarray)
    if sarray.ndim != 3:
        raise ValueError(f"Scalar array must have ndim=3 (got ndim={sarray.ndim}, shape={sarray.shape}).")


def ensure_varray(
    varray: numpy.ndarray,
) -> None:
    ensure_numpy_array(varray)
    if (varray.ndim != 4) or (varray.shape[0] != 3):
        raise ValueError(
            f"Vector arrays mustmust have shape (3, num_cells_x, num_cells_y, num_cells_z) "
            f"(got shape={varray.shape}).",
        )


def ensure_r2tarray(
    r2tarray: numpy.ndarray,
) -> None:
    ensure_numpy_array(r2tarray)
    if (r2tarray.ndim != 5) or (r2tarray.shape[0] != 3) or (r2tarray.shape[1] != 3):
        raise ValueError(
            f"Rank-2 tensor arrays mustmust have shape (3, 3, num_cells_x, num_cells_y, num_cells_z) "
            f"(got shape={r2tarray.shape}).",
        )


def ensure_same_shape(
    array_a: numpy.ndarray,
    array_b: numpy.ndarray,
) -> None:
    ensure_numpy_array(array_a)
    ensure_numpy_array(array_b)
    if array_a.shape != array_b.shape:
        raise ValueError(f"Shape mismatch: {array_a.shape} vs {array_b.shape}")


def ensure_valid_cell_widths(
    cell_widths: tuple[float, float, float] | list[float],
) -> None:
    type_utils.assert_sequence(
        var_obj=cell_widths,
        valid_containers=(tuple, list),
        seq_length=3,
    )
    for width_index, width_value in enumerate(cell_widths):
        try:
            width_float = float(width_value)
        except Exception as error:
            raise ValueError(f"cell_widths[{width_index}] must be numeric.") from error
        if not numpy.isfinite(width_float):
            raise ValueError(f"cell_widths[{width_index}] must be finite.")
        if not (width_float > 0.0):
            raise ValueError(f"cell_widths[{width_index}] must be positive.")


## } MODULE
