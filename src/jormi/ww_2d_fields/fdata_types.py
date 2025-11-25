## { MODULE

##
## === DEPENDENCIES
##

import numpy

from dataclasses import dataclass

from jormi.ww_types import array_checks, fdata_types


##
## --- 2D SPECIALISATIONS (SCALAR / VECTOR)
##


@dataclass(frozen=True, init=False)
class ScalarFieldData(fdata_types.FieldData):
    """2D scalar field data: ndarray of shape (Nx, Ny)."""

    def __init__(
        self,
        *,
        farray: fdata_types.FieldArray,
        param_name: str = "<sdata>",
    ) -> None:
        super().__init__(
            farray=farray,
            num_ranks=0,
            num_comps=1,
            num_sdims=2,
            param_name=param_name,
        )


@dataclass(frozen=True, init=False)
class VectorFieldData(fdata_types.FieldData):
    """2D vector field data: ndarray of shape (2, Nx, Ny)."""

    def __init__(
        self,
        *,
        farray: fdata_types.FieldArray,
        param_name: str = "<vdata>",
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


def ensure_sdata(
    sdata: ScalarFieldData,
    *,
    param_name: str = "<sdata>",
) -> None:
    """Ensure `sdata` is ScalarFieldData with 2D scalar layout."""
    if not isinstance(sdata, ScalarFieldData):
        raise TypeError(
            f"`{param_name}` must be ScalarFieldData; got type={type(sdata)}.",
        )
    fdata_types.ensure_fdata_metadata(
        fdata=sdata,
        num_comps=1,
        num_sdims=2,
        num_ranks=0,
        param_name=param_name,
    )


def ensure_vdata(
    vdata: VectorFieldData,
    *,
    param_name: str = "<vdata>",
) -> None:
    """Ensure `vdata` is VectorFieldData with 2 components in 2D."""
    if not isinstance(vdata, VectorFieldData):
        raise TypeError(
            f"`{param_name}` must be VectorFieldData; got type={type(vdata)}.",
        )
    fdata_types.ensure_fdata_metadata(
        fdata=vdata,
        num_comps=2,
        num_sdims=2,
        num_ranks=1,
        param_name=param_name,
    )


##
## --- 2D NDARRAY VALIDATION
##


def ensure_sarray(
    sarray: numpy.ndarray,
    *,
    param_name: str = "<sarray>",
) -> None:
    """Ensure `sarray` is a 2D scalar ndarray with shape (Nx, Ny)."""
    array_checks.ensure_dims(
        array=sarray,
        param_name=param_name,
        num_dims=2,
    )


def ensure_varray(
    varray: numpy.ndarray,
    *,
    param_name: str = "<varray>",
) -> None:
    """Ensure `varray` is a 3D vector ndarray with leading axis of length 2."""
    array_checks.ensure_dims(
        array=varray,
        param_name=param_name,
        num_dims=3,
    )
    if varray.shape[0] != 2:
        raise ValueError(
            f"`{param_name}` must have shape"
            f" (2, num_cells_x, num_cells_y);"
            f" got shape={varray.shape}.",
        )


##
## --- 2D NDARRAY NORMALISERS
##


def as_sarray(
    sdata: ScalarFieldData | numpy.ndarray,
    *,
    param_name: str = "<sdata>",
) -> numpy.ndarray:
    """
    Normalise `sdata` to a 2D scalar ndarray and validate its structure.

    Accepts either:
      - raw ndarray of shape (Nx, Ny), or
      - ScalarFieldData (2D scalar field data).
    """
    if isinstance(sdata, ScalarFieldData):
        farray = sdata.farray
        ensure_sarray(
            sarray=farray,
            param_name="<sdata.farray>",
        )
        return farray
    ensure_sarray(
        sarray=sdata,
        param_name=param_name,
    )
    return sdata


def as_varray(
    vdata: VectorFieldData | numpy.ndarray,
    *,
    param_name: str = "<vdata>",
) -> numpy.ndarray:
    """
    Normalise `vdata` to a 2D vector ndarray and validate its structure.

    Accepts either:
      - raw ndarray of shape (2, Nx, Ny), or
      - VectorFieldData (2D vector field data).
    """
    if isinstance(vdata, VectorFieldData):
        farray = vdata.farray
        ensure_varray(
            varray=farray,
            param_name="<vdata.farray>",
        )
        return farray
    ensure_varray(
        varray=vdata,
        param_name=param_name,
    )
    return vdata


## } MODULE
