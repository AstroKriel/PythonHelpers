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
from jormi.ww_fields import _field_data

##
## === POINT-CLOUD SCALAR / VECTOR NDARRAY
##


@dataclass(
    frozen=True,
    init=False,
)
class ScalarFieldData_PointCloud(_field_data.FieldData):
    """Point-cloud scalar field data: ndarray of shape (num_cells,)."""

    def __init__(
        self,
        *,
        farray: NDArray[Any],
        param_name: str = "<sdata_pointcloud>",
    ) -> None:
        super().__init__(
            farray=farray,
            num_ranks=0,
            num_comps=1,
            num_sdims=1,
            param_name=param_name,
        )


@dataclass(
    frozen=True,
    init=False,
)
class VectorFieldData_PointCloud(_field_data.FieldData):
    """Point-cloud vector field data: ndarray of shape (3, num_cells)."""

    def __init__(
        self,
        *,
        farray: NDArray[Any],
        param_name: str = "<vdata_pointcloud>",
    ) -> None:
        super().__init__(
            farray=farray,
            num_ranks=1,
            num_comps=3,
            num_sdims=1,
            param_name=param_name,
        )


##
## === POINT-CLOUD FIELD DATA VALIDATION
##


def ensure_pointcloud_sdata(
    sdata_pointcloud: ScalarFieldData_PointCloud,
    *,
    param_name: str = "<sdata_pointcloud>",
) -> None:
    """Ensure `sdata_pointcloud` is ScalarFieldData_PointCloud with point-cloud scalar layout."""
    if not isinstance(sdata_pointcloud, ScalarFieldData_PointCloud):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(  # pyright: ignore[reportUnreachable]
            f"`{param_name}` must be ScalarFieldData_PointCloud; got type={type(sdata_pointcloud)}.",
        )
    _field_data.ensure_fdata_metadata(
        fdata=sdata_pointcloud,
        num_comps=1,
        num_sdims=1,
        num_ranks=0,
        param_name=param_name,
    )


def ensure_pointcloud_vdata(
    vdata_pointcloud: VectorFieldData_PointCloud,
    *,
    param_name: str = "<vdata_pointcloud>",
) -> None:
    """Ensure `vdata_pointcloud` is VectorFieldData_PointCloud with 3 components."""
    if not isinstance(vdata_pointcloud, VectorFieldData_PointCloud):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(  # pyright: ignore[reportUnreachable]
            f"`{param_name}` must be VectorFieldData_PointCloud; got type={type(vdata_pointcloud)}.",
        )
    _field_data.ensure_fdata_metadata(
        fdata=vdata_pointcloud,
        num_comps=3,
        num_sdims=1,
        num_ranks=1,
        param_name=param_name,
    )


##
## === POINT-CLOUD NDARRAY NORMALISERS
##


def extract_pointcloud_sarray(
    sdata_pointcloud: ScalarFieldData_PointCloud,
    *,
    param_name: str = "<sdata_pointcloud>",
) -> NDArray[Any]:
    """Normalise `sdata_pointcloud` to a point-cloud scalar ndarray and validate its structure."""
    ensure_pointcloud_sdata(
        sdata_pointcloud=sdata_pointcloud,
        param_name=param_name,
    )
    return sdata_pointcloud.farray


def extract_pointcloud_varray(
    vdata_pointcloud: VectorFieldData_PointCloud,
    *,
    param_name: str = "<vdata_pointcloud>",
) -> NDArray[Any]:
    """Normalise `vdata_pointcloud` to a point-cloud vector ndarray and validate its structure."""
    ensure_pointcloud_vdata(
        vdata_pointcloud=vdata_pointcloud,
        param_name=param_name,
    )
    return vdata_pointcloud.farray


## } MODULE
