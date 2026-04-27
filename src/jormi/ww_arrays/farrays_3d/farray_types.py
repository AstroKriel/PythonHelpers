## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Any, TypeAlias

## third-party
import numpy
from numpy.typing import DTypeLike, NDArray

## local
from jormi.ww_validation import validate_arrays, validate_types

##
## === TYPE ALIASES
##

FieldArray: TypeAlias = NDArray[Any]

##
## === BUFFER ALLOCATION
##


def ensure_farray_metadata(
    *,
    farray_shape: tuple[int, ...],
    farray: NDArray[Any] | None = None,
    dtype: DTypeLike | None = None,
) -> NDArray[Any]:
    """
    Return a farray with the requested shape/dtype, reusing the provided farray
    if compatible, otherwise allocate a new farray.
    """
    if dtype is None:
        if farray is not None:
            dtype = farray.dtype
        else:
            dtype = numpy.float64
    else:
        dtype = numpy.dtype(dtype)
    if (farray is None) or (farray.shape != farray_shape) or (farray.dtype != dtype):
        return numpy.empty(farray_shape, dtype=dtype)
    return farray


##
## === SHAPE VALIDATION
##


def ensure_3d_sarray(
    sarray_3d: NDArray[Any],
    *,
    param_name: str = "<sarray_3d>",
) -> None:
    """Ensure `sarray_3d` is a 3D scalar ndarray with shape (num_x0_cells, num_x1_cells, num_x2_cells)."""
    validate_arrays.ensure_dims(
        array=sarray_3d,
        param_name=param_name,
        num_dims=3,
    )


def ensure_3d_varray(
    varray_3d: NDArray[Any],
    *,
    param_name: str = "<varray_3d>",
) -> None:
    """Ensure `varray_3d` is a 4D vector ndarray with leading axis of length 3."""
    validate_arrays.ensure_dims(
        array=varray_3d,
        param_name=param_name,
        num_dims=4,
    )
    if varray_3d.shape[0] != 3:
        raise ValueError(
            f"`{param_name}` must have shape"
            f" (3, num_cells_x, num_cells_y, num_cells_z);"
            f" got shape={varray_3d.shape}.",
        )


def ensure_3d_r2tarray(
    r2tarray_3d: NDArray[Any],
    *,
    param_name: str = "<r2tarray_3d>",
) -> None:
    """Ensure `r2tarray_3d` is a 5D rank-2 tensor ndarray with two leading axes of length 3."""
    validate_arrays.ensure_dims(
        array=r2tarray_3d,
        param_name=param_name,
        num_dims=5,
    )
    if (r2tarray_3d.shape[0] != 3) or (r2tarray_3d.shape[1] != 3):
        raise ValueError(
            f"`{param_name}` must have shape"
            f" (3, 3, num_cells_x, num_cells_y, num_cells_z);"
            f" got shape={r2tarray_3d.shape}.",
        )


def ensure_uvarray_magnitude(
    varray_3d: NDArray[Any],
    *,
    tol: float = 1e-6,
    param_name: str = "<varray_3d>",
) -> None:
    """
    Validate that every vector in a (3, num_x0_cells, num_x1_cells, num_x2_cells) array has unit magnitude.

    Raises ValueError if any element is non-finite, or if any magnitude deviates from 1.0 by more than `tol`.
    """
    ensure_3d_varray(
        varray_3d=varray_3d,
        param_name=param_name,
    )
    sarray_3d_vmagn_sq = numpy.einsum("i...,i...->...", varray_3d, varray_3d)
    if not numpy.all(numpy.isfinite(sarray_3d_vmagn_sq)):
        raise ValueError(
            f"`{param_name}` should not contain any NaN/Inf magnitudes.",
        )
    max_error = float(
        numpy.max(
            numpy.abs(
                numpy.sqrt(sarray_3d_vmagn_sq) - 1.0,
            ),
        ),
    )
    if max_error > tol:
        raise ValueError(
            f"`{param_name}` magnitude deviates from unit-magnitude=1.0 by"
            f" max(error)={max_error:.3e} (tol={tol}).",
        )


def ensure_3d_cell_widths(
    cell_widths_3d: tuple[float, float, float] | list[float],
    *,
    param_name: str = "<cell_widths_3d>",
) -> None:
    """Strictly validate `cell_widths_3d` as a length-3 sequence of finite, positive floats."""
    validate_types.ensure_sequence(
        param=cell_widths_3d,
        param_name=param_name,
        allow_none=False,
        seq_length=3,
        valid_seq_types=validate_types.RuntimeTypes.Sequences.SequenceLike,
        valid_elem_types=validate_types.RuntimeTypes.Numerics.FloatLike,
    )
    for dim_index, cell_width in enumerate(cell_widths_3d):
        validate_types.ensure_finite_float(
            param=cell_width,
            param_name=f"{param_name}[{dim_index}]",
            allow_none=False,
            allow_zero=False,
            require_positive=True,
        )


## } MODULE
