## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from typing import Any

## third-party
import numpy
from numpy.typing import NDArray, DTypeLike

## local
from jormi.ww_arrays.farrays_3d import farray_types
from jormi.ww_fields import _fdata_types

##
## --- 3D SCALAR / VECTOR / RANK-2 TENSOR NDARRAY
##


@dataclass(frozen=True, init=False)
class ScalarFieldData_3D(_fdata_types.FieldData):
    """3D scalar field data: ndarray of shape (num_x0_cells, num_x1_cells, num_x2_cells)."""

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
    """3D vector field data: ndarray of shape (3, num_x0_cells, num_x1_cells, num_x2_cells)."""

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
    """3D rank-2 tensor data: ndarray of shape (3, 3, num_x0_cells, num_x1_cells, num_x2_cells)."""

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


def ensure_3d_sdata(
    sdata_3d: ScalarFieldData_3D,
    *,
    param_name: str = "<sdata_3d>",
) -> None:
    """Ensure `sdata_3d` is ScalarFieldData_3D with 3D scalar layout."""
    if not isinstance(sdata_3d, ScalarFieldData_3D):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(  # pyright: ignore[reportUnreachable]
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
    if not isinstance(vdata_3d, VectorFieldData_3D):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(  # pyright: ignore[reportUnreachable]
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
    if not isinstance(r2tdata_3d, Rank2TensorData_3D):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(  # pyright: ignore[reportUnreachable]
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
## --- 3D NDARRAY NORMALISERS
##


def extract_3d_sarray(
    sdata_3d: ScalarFieldData_3D,
    *,
    param_name: str = "<sdata_3d>",
) -> NDArray[Any]:
    """Normalise `sdata_3d` to a 3D scalar ndarray and validate its structure."""
    ensure_3d_sdata(
        sdata_3d=sdata_3d,
        param_name=param_name,
    )
    sarray_3d = sdata_3d.farray
    farray_types.ensure_3d_sarray(
        sarray_3d=sarray_3d,
        param_name=f"{param_name}.farray",
    )
    return sarray_3d


def extract_3d_varray(
    vdata_3d: VectorFieldData_3D,
    *,
    param_name: str = "<vdata_3d>",
) -> NDArray[Any]:
    """Normalise `vdata_3d` to a 3D vector ndarray and validate its structure."""
    ensure_3d_vdata(
        vdata_3d=vdata_3d,
        param_name=param_name,
    )
    varray_3d = vdata_3d.farray
    farray_types.ensure_3d_varray(
        varray_3d=varray_3d,
        param_name=f"{param_name}.farray",
    )
    return varray_3d


def extract_3d_r2tarray(
    r2tdata_3d: Rank2TensorData_3D,
    *,
    param_name: str = "<r2tdata_3d>",
) -> NDArray[Any]:
    """Normalise `r2tdata_3d` to a 3D rank-2 tensor ndarray and validate its structure."""
    ensure_3d_r2tdata(
        r2tdata_3d=r2tdata_3d,
        param_name=param_name,
    )
    r2tarray_3d = r2tdata_3d.farray
    farray_types.ensure_3d_r2tarray(
        r2tarray_3d=r2tarray_3d,
        param_name=f"{param_name}.farray",
    )
    return r2tarray_3d


## } MODULE
