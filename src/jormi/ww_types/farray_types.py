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
    *,
    param_name: str = "sarray",
) -> None:
    """Ensure `sarray` is a 3D scalar array with shape (Nx, Ny, Nz)."""
    array_types.ensure_array(
        array=sarray,
        param_name=param_name,
    )
    if sarray.ndim != 3:
        raise ValueError(
            f"`{param_name}` must have num_dims=3; got num_dims={sarray.ndim},"
            f" with shape={sarray.shape}.",
        )


def ensure_varray(
    varray: numpy.ndarray,
    *,
    param_name: str = "varray",
) -> None:
    """Ensure `varray` is a 4D vector array with leading component axis of length 3."""
    array_types.ensure_array(
        array=varray,
        param_name=param_name,
    )
    if (varray.ndim != 4) or (varray.shape[0] != 3):
        raise ValueError(
            f"`{param_name}` must have shape"
            f" (num_comps=3, num_cells_x, num_cells_y, num_cells_z);"
            f" got shape={varray.shape}.",
        )


def ensure_r2tarray(
    r2tarray: numpy.ndarray,
    *,
    param_name: str = "r2tarray",
) -> None:
    """Ensure `r2tarray` is a 5D rank-2 tensor array with two leading axes of length 3."""
    array_types.ensure_array(
        array=r2tarray,
        param_name=param_name,
    )
    if (r2tarray.ndim != 5) or (r2tarray.shape[0] != 3) or (r2tarray.shape[1] != 3):
        raise ValueError(
            f"`{param_name}` must have shape"
            f" (num_comps=3, num_dims=3, num_cells_x, num_cells_y, num_cells_z);"
            f" got shape={r2tarray.shape}.",
        )


def ensure_valid_cell_widths(
    cell_widths: tuple[float, float, float] | list[float],
    *,
    param_name: str = "cell_widths",
) -> None:
    """Ensure `cell_widths` is a length-3 sequence of positive, finite values."""
    type_manager.ensure_sequence(
        param=cell_widths,
        param_name=param_name,
        seq_length=3,
        valid_seq_types=type_manager.SequenceTypes.SEQUENCE,
        valid_elem_types=type_manager.NumericTypes.NUMERIC,
    )
    for width_index, width_value in enumerate(cell_widths):
        try:
            width_float = float(width_value)
        except Exception as error:
            raise ValueError(f"{param_name}[{width_index}] must be numeric.") from error
        if not numpy.isfinite(width_float):
            raise ValueError(f"{param_name}[{width_index}] must be finite.")
        if not (width_float > 0.0):
            raise ValueError(f"{param_name}[{width_index}] must be positive.")


## } MODULE
