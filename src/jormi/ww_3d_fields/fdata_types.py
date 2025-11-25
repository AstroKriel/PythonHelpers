## { MODULE

##
## === DEPENDENCIES
##

import numpy

from dataclasses import dataclass

from jormi.ww_types import array_checks, fdata_types


##
## --- 3D SPECIALISATIONS (SCALAR / VECTOR / RANK-2 TENSOR)
##


@dataclass(frozen=True, init=False)
class ScalarFieldData(fdata_types.FieldData):
    """3D scalar field data: ndarray of shape (Nx, Ny, Nz)."""

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
            num_sdims=3,
            param_name=param_name,
        )


@dataclass(frozen=True, init=False)
class VectorFieldData(fdata_types.FieldData):
    """3D vector field data: ndarray of shape (3, Nx, Ny, Nz)."""

    def __init__(
        self,
        *,
        farray: fdata_types.FieldArray,
        param_name: str = "<vdata>",
    ) -> None:
        super().__init__(
            farray=farray,
            num_ranks=1,
            num_comps=3,
            num_sdims=3,
            param_name=param_name,
        )


@dataclass(frozen=True, init=False)
class Rank2TensorData(fdata_types.FieldData):
    """3D rank-2 tensor data: ndarray of shape (3, 3, Nx, Ny, Nz)."""

    def __init__(
        self,
        *,
        farray: fdata_types.FieldArray,
        param_name: str = "<r2tdata>",
    ) -> None:
        super().__init__(
            farray=farray,
            num_ranks=2,
            num_comps=9,
            num_sdims=3,
            param_name=param_name,
        )


##
## --- 3D FIELD DATA VALIDATION
##


def ensure_sdata(
    sdata: ScalarFieldData,
    *,
    param_name: str = "<sdata>",
) -> None:
    """Ensure `sdata` is ScalarFieldData with 3D scalar layout."""
    if not isinstance(sdata, ScalarFieldData):
        raise TypeError(
            f"`{param_name}` must be ScalarFieldData; got type={type(sdata)}.",
        )
    fdata_types.ensure_fdata_metadata(
        fdata=sdata,
        num_comps=1,
        num_sdims=3,
        num_ranks=0,
        param_name=param_name,
    )


def ensure_vdata(
    vdata: VectorFieldData,
    *,
    param_name: str = "<vdata>",
) -> None:
    """Ensure `vdata` is VectorFieldData with 3 components in 3D."""
    if not isinstance(vdata, VectorFieldData):
        raise TypeError(
            f"`{param_name}` must be VectorFieldData; got type={type(vdata)}.",
        )
    fdata_types.ensure_fdata_metadata(
        fdata=vdata,
        num_comps=3,
        num_sdims=3,
        num_ranks=1,
        param_name=param_name,
    )


def ensure_r2tdata(
    r2tdata: Rank2TensorData,
    *,
    param_name: str = "<r2tdata>",
) -> None:
    """Ensure `r2tdata` is Rank2TensorData with 3x3 components in 3D."""
    if not isinstance(r2tdata, Rank2TensorData):
        raise TypeError(
            f"`{param_name}` must be Rank2TensorData; got type={type(r2tdata)}.",
        )
    fdata_types.ensure_fdata_metadata(
        fdata=r2tdata,
        num_comps=9,
        num_sdims=3,
        num_ranks=2,
        param_name=param_name,
    )


##
## --- 3D NDARRAY VALIDATION
##


def ensure_sarray(
    sarray: numpy.ndarray,
    *,
    param_name: str = "<sarray>",
) -> None:
    """Ensure `sarray` is a 3D scalar ndarray with shape (Nx, Ny, Nz)."""
    array_checks.ensure_dims(
        array=sarray,
        param_name=param_name,
        num_dims=3,
    )


def ensure_varray(
    varray: numpy.ndarray,
    *,
    param_name: str = "<varray>",
) -> None:
    """Ensure `varray` is a 4D vector ndarray with leading axis of length 3."""
    array_checks.ensure_dims(
        array=varray,
        param_name=param_name,
        num_dims=4,
    )
    if varray.shape[0] != 3:
        raise ValueError(
            f"`{param_name}` must have shape"
            f" (3, num_cells_x, num_cells_y, num_cells_z);"
            f" got shape={varray.shape}.",
        )


def ensure_r2tarray(
    r2tarray: numpy.ndarray,
    *,
    param_name: str = "<r2tarray>",
) -> None:
    """Ensure `r2tarray` is a 5D rank-2 tensor ndarray with two leading axes of length 3."""
    array_checks.ensure_dims(
        array=r2tarray,
        param_name=param_name,
        num_dims=5,
    )
    if (r2tarray.shape[0] != 3) or (r2tarray.shape[1] != 3):
        raise ValueError(
            f"`{param_name}` must have shape"
            f" (3, 3, num_cells_x, num_cells_y, num_cells_z);"
            f" got shape={r2tarray.shape}.",
        )


##
## --- 3D NDARRAY NORMALISERS
##


def as_sarray(
    sdata: ScalarFieldData | numpy.ndarray,
    *,
    param_name: str = "<sdata>",
) -> numpy.ndarray:
    """
    Normalise `sdata` to a 3D scalar ndarray and validate its structure.

    Accepts either:
      - raw ndarray of shape (Nx, Ny, Nz), or
      - ScalarFieldData (3D scalar field data).
    """
    if isinstance(sdata, ScalarFieldData):
        sarray = sdata.farray
        ensure_sarray(
            sarray=sarray,
            param_name="<sdata.farray>",
        )
        return sarray
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
    Normalise `vdata` to a 3D vector ndarray and validate its structure.

    Accepts either:
      - raw ndarray of shape (3, Nx, Ny, Nz), or
      - VectorFieldData (3D vector field data).
    """
    if isinstance(vdata, VectorFieldData):
        varray = vdata.farray
        ensure_varray(
            varray=varray,
            param_name="<vdata.farray>",
        )
        return varray
    ensure_varray(
        varray=vdata,
        param_name=param_name,
    )
    return vdata


def as_r2tarray(
    r2tdata: Rank2TensorData | numpy.ndarray,
    *,
    param_name: str = "<r2tdata>",
) -> numpy.ndarray:
    """
    Normalise `r2tdata` to a 3D rank-2 tensor ndarray and validate its structure.

    Accepts either:
      - raw ndarray of shape (3, 3, Nx, Ny, Nz), or
      - Rank2TensorData (3D rank-2 tensor data).
    """
    if isinstance(r2tdata, Rank2TensorData):
        r2tarray = r2tdata.farray
        ensure_r2tarray(
            r2tarray=r2tarray,
            param_name="<r2tdata.farray>",
        )
        return r2tarray
    ensure_r2tarray(
        r2tarray=r2tdata,
        param_name=param_name,
    )
    return r2tdata


## } MODULE
