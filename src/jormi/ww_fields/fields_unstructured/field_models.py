## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from typing import Any, Self

## third-party
from numpy.typing import NDArray

## local
from jormi.ww_fields import (
    _domain_models,
    _field_models,
)
from jormi.ww_fields.fields_unstructured import _field_data
from jormi.ww_validation import validate_types

##
## === POINT-CLOUD FIELD TYPES
##


@dataclass(frozen=True)
class ScalarField_PointCloud(_field_models.Field):
    """Point-cloud scalar field: `num_ranks == 0`, `num_comps == 1`, `num_sdims == 1`."""

    fdata: _field_data.ScalarFieldData_PointCloud
    uniform_domain: _domain_models.PointCloudDomain

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()
        _field_data.ensure_pointcloud_sdata(
            sdata_pointcloud=self.fdata,
            param_name="<sfield_pointcloud.fdata>",
        )

    @classmethod
    def from_pointcloud_sarray(
        cls,
        *,
        sarray_pointcloud: NDArray[Any],
        pointcloud_domain: _domain_models.PointCloudDomain,
        field_name: str,
        latex_label: str,
        sim_time: float | None = None,
    ) -> Self:
        """Construct a point-cloud scalar field from a (num_cells,) ndarray."""
        sdata_pointcloud = _field_data.ScalarFieldData_PointCloud(
            farray=sarray_pointcloud,
            param_name="<sdata_pointcloud>",
        )
        return cls(
            fdata=sdata_pointcloud,
            uniform_domain=pointcloud_domain,
            field_name=field_name,
            latex_label=latex_label,
            sim_time=sim_time,
        )


@dataclass(frozen=True)
class VectorField_PointCloud(_field_models.Field):
    """Point-cloud vector field: `num_ranks == 1`, `num_comps == 3`, `num_sdims == 1`."""

    fdata: _field_data.VectorFieldData_PointCloud
    uniform_domain: _domain_models.PointCloudDomain

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()
        _field_data.ensure_pointcloud_vdata(
            vdata_pointcloud=self.fdata,
            param_name="<vfield_pointcloud.fdata>",
        )

    @classmethod
    def from_pointcloud_varray(
        cls,
        *,
        varray_pointcloud: NDArray[Any],
        pointcloud_domain: _domain_models.PointCloudDomain,
        field_name: str,
        latex_label: str,
        sim_time: float | None = None,
    ) -> Self:
        """Construct a point-cloud vector field from a (3, num_cells) ndarray."""
        vdata_pointcloud = _field_data.VectorFieldData_PointCloud(
            farray=varray_pointcloud,
            param_name="<vdata_pointcloud>",
        )
        return cls(
            fdata=vdata_pointcloud,
            uniform_domain=pointcloud_domain,
            field_name=field_name,
            latex_label=latex_label,
            sim_time=sim_time,
        )


##
## === POINT-CLOUD FIELD VALIDATION
##


def ensure_pointcloud_sfield(
    sfield_pointcloud: ScalarField_PointCloud,
    *,
    param_name: str = "<sfield_pointcloud>",
) -> None:
    validate_types.ensure_type(
        param=sfield_pointcloud,
        param_name=param_name,
        valid_types=ScalarField_PointCloud,
    )


def ensure_pointcloud_vfield(
    vfield_pointcloud: VectorField_PointCloud,
    *,
    param_name: str = "<vfield_pointcloud>",
) -> None:
    validate_types.ensure_type(
        param=vfield_pointcloud,
        param_name=param_name,
        valid_types=VectorField_PointCloud,
    )


##
## === EXTRACT NDARRAY FROM FIELDS
##


def extract_pointcloud_sarray(
    sfield_pointcloud: ScalarField_PointCloud,
    *,
    param_name: str = "<sfield_pointcloud>",
) -> NDArray[Any]:
    """Return the underlying (num_cells,) ndarray for a point-cloud scalar field."""
    ensure_pointcloud_sfield(
        sfield_pointcloud=sfield_pointcloud,
        param_name=param_name,
    )
    return sfield_pointcloud.fdata.farray


def extract_pointcloud_varray(
    vfield_pointcloud: VectorField_PointCloud,
    *,
    param_name: str = "<vfield_pointcloud>",
) -> NDArray[Any]:
    """Return the underlying (3, num_cells) ndarray for a point-cloud vector field."""
    ensure_pointcloud_vfield(
        vfield_pointcloud=vfield_pointcloud,
        param_name=param_name,
    )
    return vfield_pointcloud.fdata.farray


## } MODULE
