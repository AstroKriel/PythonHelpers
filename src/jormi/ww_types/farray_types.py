## { MODULE

##
## === DEPENDENCIES
##

import numpy

from jormi.ww_types import type_manager, array_types

##
## === DATA TYPE VALIDATION
##


def ensure_sarray(
    sarray: numpy.ndarray,
) -> None:
    array_types.ensure_array(sarray)
    if sarray.ndim != 3:
        raise ValueError(
            f"Scalar array must have num_dims=3; got num_dims={sarray.ndim}, with shape={sarray.shape}.",
        )


def ensure_varray(
    varray: numpy.ndarray,
) -> None:
    array_types.ensure_array(varray)
    if (varray.ndim != 4) or (varray.shape[0] != 3):
        raise ValueError(
            f"Vector arrays must have shape (num_dims=3, num_cells_x, num_cells_y, num_cells_z) "
            f"(got shape={varray.shape}).",
        )


def ensure_r2tarray(
    r2tarray: numpy.ndarray,
) -> None:
    array_types.ensure_array(r2tarray)
    if (r2tarray.ndim != 5) or (r2tarray.shape[0] != 3) or (r2tarray.shape[1] != 3):
        raise ValueError(
            f"Rank-2 tensor arrays must have shape: (num_comps=3, num_dims=3, num_cells_x, num_cells_y, num_cells_z)"
            f"; (got shape={r2tarray.shape}).",
        )


def ensure_valid_cell_widths(
    cell_widths: tuple[float, float, float] | list[float],
) -> None:
    type_manager.ensure_sequence(
        param=cell_widths,
        valid_seq_types=(tuple, list),
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
