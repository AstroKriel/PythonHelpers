## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from typing import Any

## third-party
from numpy.typing import NDArray

## local
from jormi.ww_arrays.farrays_3d import farray_types
from jormi.ww_fields import _field_data
from jormi.ww_validation import validate_arrays

##
## === 2D SCALAR / VECTOR NDARRAY
##


@dataclass(
    frozen=True,
    init=False,
)
class ScalarFieldData_2D(_field_data.FieldData):
    """2D scalar field data: ndarray of shape (num_x0_cells, num_x1_cells)."""

    def __init__(
        self,
        *,
        farray: farray_types.FieldArray,
        param_name: str = "<sdata_2d>",
    ) -> None:
        super().__init__(
            farray=farray,
            num_ranks=0,
            num_comps=1,
            num_sdims=2,
            param_name=param_name,
        )


@dataclass(
    frozen=True,
    init=False,
)
class VectorFieldData_2D(_field_data.FieldData):
    """2D vector field data: ndarray of shape (2, num_x0_cells, num_x1_cells)."""

    def __init__(
        self,
        *,
        farray: farray_types.FieldArray,
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
## === 2D FIELD DATA VALIDATION
##


def ensure_2d_sdata(
    sdata_2d: ScalarFieldData_2D,
    *,
    param_name: str = "<sdata_2d>",
) -> None:
    """Ensure `sdata_2d` is ScalarFieldData_2D with 2D scalar layout."""
    if not isinstance(sdata_2d, ScalarFieldData_2D):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(  # pyright: ignore[reportUnreachable]
            f"`{param_name}` must be ScalarFieldData_2D; got type={type(sdata_2d)}.",
        )
    _field_data.ensure_fdata_metadata(
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
    if not isinstance(vdata_2d, VectorFieldData_2D):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(  # pyright: ignore[reportUnreachable]
            f"`{param_name}` must be VectorFieldData_2D; got type={type(vdata_2d)}.",
        )
    _field_data.ensure_fdata_metadata(
        fdata=vdata_2d,
        num_comps=2,
        num_sdims=2,
        num_ranks=1,
        param_name=param_name,
    )


##
## === 2D NDARRAY VALIDATION
##


def ensure_2d_sarray(
    sarray_2d: NDArray[Any],
    *,
    param_name: str = "<sarray_2d>",
) -> None:
    """Ensure `sarray_2d` is a 2D scalar ndarray with shape (num_x0_cells, num_x1_cells)."""
    validate_arrays.ensure_dims(
        array=sarray_2d,
        param_name=param_name,
        num_dims=2,
    )


def ensure_2d_varray(
    varray_2d: NDArray[Any],
    *,
    param_name: str = "<varray_2d>",
) -> None:
    """Ensure `varray_2d` is a 3D vector ndarray with leading axis of length 2."""
    validate_arrays.ensure_dims(
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
## === 2D NDARRAY EXTRACTORS
##


def extract_2d_sarray(
    sdata_2d: ScalarFieldData_2D,
    *,
    param_name: str = "<sdata_2d>",
) -> NDArray[Any]:
    """Normalise `sdata_2d` to a 2D scalar ndarray and validate its structure."""
    ensure_2d_sdata(
        sdata_2d=sdata_2d,
        param_name=param_name,
    )
    return sdata_2d.farray


def extract_2d_varray(
    vdata_2d: VectorFieldData_2D,
    *,
    param_name: str = "<vdata_2d>",
) -> NDArray[Any]:
    """Normalise `vdata_2d` to a 2D vector ndarray and validate its structure."""
    ensure_2d_vdata(
        vdata_2d=vdata_2d,
        param_name=param_name,
    )
    return vdata_2d.farray


## } MODULE
