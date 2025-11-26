## { MODULE

##
## === DEPENDENCIES
##

import numpy

from dataclasses import dataclass

from jormi.ww_types import array_checks
from jormi.ww_fields import _fdata_types

##
## --- 2D SPECIALISATIONS (SCALAR / VECTOR)
##


@dataclass(frozen=True, init=False)
class ScalarFieldData_2D(_fdata_types.FieldData):
    """2D scalar field data: ndarray of shape (Nx, Ny)."""

    def __init__(
        self,
        *,
        farray: _fdata_types.FieldArray,
        param_name: str = "<sdata_2d>",
    ) -> None:
        super().__init__(
            farray=farray,
            num_ranks=0,
            num_comps=1,
            num_sdims=2,
            param_name=param_name,
        )


@dataclass(frozen=True, init=False)
class VectorFieldData_2D(_fdata_types.FieldData):
    """2D vector field data: ndarray of shape (2, Nx, Ny)."""

    def __init__(
        self,
        *,
        farray: _fdata_types.FieldArray,
        param_name: str = "<vdata_2d>",
    ) -> None:
        super().__init__(
            farray=farray,
            num_ranks=1,
            num_comps=2,
            num_sdims=2,
            param_name=param_name,
        )


##
## --- 2D FIELD DATA VALIDATION
##


def ensure_2d_sdata(
    sdata_2d: ScalarFieldData_2D,
    *,
    param_name: str = "<sdata_2d>",
) -> None:
    """Ensure `sdata_2d` is ScalarFieldData_2D with 2D scalar layout."""
    if not isinstance(sdata_2d, ScalarFieldData_2D):
        raise TypeError(
            f"`{param_name}` must be ScalarFieldData_2D; got type={type(sdata_2d)}.",
        )
    _fdata_types.ensure_fdata_metadata(
        fdata=sdata_2d,
        num_comps=1,
        num_sdims=2,
        num_ranks=0,
        param_name=param_name,
    )


def ensure_2d_vdata(
    vdata_2d: VectorFieldData_2D,
    *,
    param_name: str = "<vdata_2d>",
) -> None:
    """Ensure `vdata_2d` is VectorFieldData_2D with 2 components in 2D."""
    if not isinstance(vdata_2d, VectorFieldData_2D):
        raise TypeError(
            f"`{param_name}` must be VectorFieldData_2D; got type={type(vdata_2d)}.",
        )
    _fdata_types.ensure_fdata_metadata(
        fdata=vdata_2d,
        num_comps=2,
        num_sdims=2,
        num_ranks=1,
        param_name=param_name,
    )


##
## --- 2D NDARRAY VALIDATION
##


def ensure_2d_sarray(
    sarray_2d: numpy.ndarray,
    *,
    param_name: str = "<sarray_2d>",
) -> None:
    """Ensure `sarray_2d` is a 2D scalar ndarray with shape (Nx, Ny)."""
    array_checks.ensure_dims(
        array=sarray_2d,
        param_name=param_name,
        num_dims=2,
    )


def ensure_2d_varray(
    varray_2d: numpy.ndarray,
    *,
    param_name: str = "<varray_2d>",
) -> None:
    """Ensure `varray_2d` is a 3D vector ndarray with leading axis of length 2."""
    array_checks.ensure_dims(
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


##
## --- 2D NDARRAY NORMALISERS
##


def as_2d_sarray(
    sdata_2d: ScalarFieldData_2D | numpy.ndarray,
    *,
    param_name: str = "<sdata_2d>",
) -> numpy.ndarray:
    """
    Normalise `sdata_2d` to a 2D scalar ndarray and validate its structure.

    Accepts either:
      - raw ndarray of shape (Nx, Ny), or
      - ScalarFieldData_2D (2D scalar field data).
    """
    if isinstance(sdata_2d, ScalarFieldData_2D):
        farray = sdata_2d.farray
        ensure_2d_sarray(
            sarray_2d=farray,
            param_name="<sdata_2d.farray>",
        )
        return farray
    ensure_2d_sarray(
        sarray_2d=sdata_2d,
        param_name=param_name,
    )
    return sdata_2d


def as_2d_varray(
    vdata_2d: VectorFieldData_2D | numpy.ndarray,
    *,
    param_name: str = "<vdata_2d>",
) -> numpy.ndarray:
    """
    Normalise `vdata_2d` to a 2D vector ndarray and validate its structure.

    Accepts either:
      - raw ndarray of shape (2, Nx, Ny), or
      - VectorFieldData_2D (2D vector field data).
    """
    if isinstance(vdata_2d, VectorFieldData_2D):
        farray = vdata_2d.farray
        ensure_2d_varray(
            varray_2d=farray,
            param_name="<vdata_2d.farray>",
        )
        return farray
    ensure_2d_varray(
        varray_2d=vdata_2d,
        param_name=param_name,
    )
    return vdata_2d


## } MODULE
