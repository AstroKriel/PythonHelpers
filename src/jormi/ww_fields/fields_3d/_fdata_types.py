## { MODULE

##
## === DEPENDENCIES
##

import numpy

from dataclasses import dataclass
from numpy.typing import DTypeLike

from jormi.ww_types import array_checks
from jormi.ww_fields import _fdata_types

##
## --- 3D SCALAR / VECTOR / RANK-2 TENSOR NDARRAY
##


@dataclass(frozen=True, init=False)
class ScalarFieldData_3D(_fdata_types.FieldData):
    """3D scalar field data: ndarray of shape (Nx, Ny, Nz)."""

    def __init__(
        self,
        *,
        farray: _fdata_types.FieldArray,
        param_name: str = "<sdata_3d>",
    ) -> None:
        super().__init__(
            farray=farray,
            num_ranks=0,
            num_comps=1,
            num_sdims=3,
            param_name=param_name,
        )


@dataclass(frozen=True, init=False)
class VectorFieldData_3D(_fdata_types.FieldData):
    """3D vector field data: ndarray of shape (3, Nx, Ny, Nz)."""

    def __init__(
        self,
        *,
        farray: _fdata_types.FieldArray,
        param_name: str = "<vdata_3d>",
    ) -> None:
        super().__init__(
            farray=farray,
            num_ranks=1,
            num_comps=3,
            num_sdims=3,
            param_name=param_name,
        )


@dataclass(frozen=True, init=False)
class Rank2TensorData_3D(_fdata_types.FieldData):
    """3D rank-2 tensor data: ndarray of shape (3, 3, Nx, Ny, Nz)."""

    def __init__(
        self,
        *,
        farray: _fdata_types.FieldArray,
        param_name: str = "<r2tdata_3d>",
    ) -> None:
        super().__init__(
            farray=farray,
            num_ranks=2,
            num_comps=9,
            num_sdims=3,
            param_name=param_name,
        )


##
## --- 3D NDARRAY VALIDATION
##


def ensure_farray_metadata(
    *,
    farray_shape: tuple[int, ...],
    farray: numpy.ndarray | None = None,
    dtype: DTypeLike | None = None,
) -> numpy.ndarray:
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


def ensure_3d_sdata(
    sdata_3d: ScalarFieldData_3D,
    *,
    param_name: str = "<sdata_3d>",
) -> None:
    """Ensure `sdata_3d` is ScalarFieldData_3D with 3D scalar layout."""
    if not isinstance(sdata_3d, ScalarFieldData_3D):
        raise TypeError(
            f"`{param_name}` must be ScalarFieldData_3D; got type={type(sdata_3d)}.",
        )
    _fdata_types.ensure_fdata_metadata(
        fdata=sdata_3d,
        num_comps=1,
        num_sdims=3,
        num_ranks=0,
        param_name=param_name,
    )


def ensure_3d_vdata(
    vdata_3d: VectorFieldData_3D,
    *,
    param_name: str = "<vdata_3d>",
) -> None:
    """Ensure `vdata_3d` is VectorFieldData_3D with 3 components in 3D."""
    if not isinstance(vdata_3d, VectorFieldData_3D):
        raise TypeError(
            f"`{param_name}` must be VectorFieldData_3D; got type={type(vdata_3d)}.",
        )
    _fdata_types.ensure_fdata_metadata(
        fdata=vdata_3d,
        num_comps=3,
        num_sdims=3,
        num_ranks=1,
        param_name=param_name,
    )


def ensure_3d_r2tdata(
    r2tdata_3d: Rank2TensorData_3D,
    *,
    param_name: str = "<r2tdata_3d>",
) -> None:
    """Ensure `r2tdata_3d` is Rank2TensorData_3D with 3x3 components in 3D."""
    if not isinstance(r2tdata_3d, Rank2TensorData_3D):
        raise TypeError(
            f"`{param_name}` must be Rank2TensorData_3D; got type={type(r2tdata_3d)}.",
        )
    _fdata_types.ensure_fdata_metadata(
        fdata=r2tdata_3d,
        num_comps=9,
        num_sdims=3,
        num_ranks=2,
        param_name=param_name,
    )


##
## --- 3D NDARRAY VALIDATION
##


def ensure_3d_sarray(
    sarray_3d: numpy.ndarray,
    *,
    param_name: str = "<sarray_3d>",
) -> None:
    """Ensure `sarray_3d` is a 3D scalar ndarray with shape (Nx, Ny, Nz)."""
    array_checks.ensure_dims(
        array=sarray_3d,
        param_name=param_name,
        num_dims=3,
    )


def ensure_3d_varray(
    varray_3d: numpy.ndarray,
    *,
    param_name: str = "<varray_3d>",
) -> None:
    """Ensure `varray_3d` is a 4D vector ndarray with leading axis of length 3."""
    array_checks.ensure_dims(
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
    r2tarray_3d: numpy.ndarray,
    *,
    param_name: str = "<r2tarray_3d>",
) -> None:
    """Ensure `r2tarray_3d` is a 5D rank-2 tensor ndarray with two leading axes of length 3."""
    array_checks.ensure_dims(
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


##
## --- 3D NDARRAY NORMALISERS
##


def extract_3d_sarray(
    sdata_3d: ScalarFieldData_3D,
    *,
    param_name: str = "<sdata_3d>",
) -> numpy.ndarray:
    """Normalise `sdata_3d` to a 3D scalar ndarray and validate its structure."""
    ensure_3d_sdata(
        sdata_3d=sdata_3d,
        param_name=param_name,
    )
    sarray_3d = sdata_3d.farray
    ensure_3d_sarray(
        sarray_3d=sarray_3d,
        param_name=f"{param_name}.farray",
    )
    return sarray_3d


def extract_3d_varray(
    vdata_3d: VectorFieldData_3D,
    *,
    param_name: str = "<vdata_3d>",
) -> numpy.ndarray:
    """Normalise `vdata_3d` to a 3D vector ndarray and validate its structure."""
    ensure_3d_vdata(
        vdata_3d=vdata_3d,
        param_name=param_name,
    )
    varray_3d = vdata_3d.farray
    ensure_3d_varray(
        varray_3d=varray_3d,
        param_name=f"{param_name}.farray",
    )
    return varray_3d


def extract_3d_r2tarray(
    r2tdata_3d: Rank2TensorData_3D,
    *,
    param_name: str = "<r2tdata_3d>",
) -> numpy.ndarray:
    """Normalise `r2tdata_3d` to a 3D rank-2 tensor ndarray and validate its structure."""
    ensure_3d_r2tdata(
        r2tdata_3d=r2tdata_3d,
        param_name=param_name,
    )
    r2tarray_3d = r2tdata_3d.farray
    ensure_3d_r2tarray(
        r2tarray_3d=r2tarray_3d,
        param_name=f"{param_name}.farray",
    )
    return r2tarray_3d


## } MODULE
