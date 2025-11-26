## { MODULE

##
## === DEPENDENCIES
##

from dataclasses import dataclass

from jormi.ww_types import type_manager
from jormi.ww_fields import _field_types
from jormi.ww_fields.fields_2d import _fdata_types, domain_types

##
## === 2D FIELD TYPES
##


@dataclass(frozen=True)
class ScalarField_2D(_field_types.Field):
    """2D scalar field: num_ranks=0, num_comps=1, num_sdims=2."""

    fdata: _fdata_types.ScalarFieldData_2D
    udomain: domain_types.UniformDomain_2D

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()
        self._validate_sdata()

    def _validate_sdata(
        self,
    ) -> None:
        _fdata_types.ensure_2d_sdata(
            sdata_2d=self.fdata,
            param_name="<sfield_2d.fdata>",
        )
        _field_types.ensure_field_metadata(
            field=self,
            num_comps=1,
            num_sdims=2,
            num_ranks=0,
            param_name="<sfield_2d>",
        )


@dataclass(frozen=True)
class VectorField_2D(_field_types.Field):
    """2D vector field: num_ranks=1, num_comps=2, num_sdims=2."""

    fdata: _fdata_types.VectorFieldData_2D
    udomain: domain_types.UniformDomain_2D

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()
        self._validate_vdata()

    def _validate_vdata(
        self,
    ) -> None:
        _fdata_types.ensure_2d_vdata(
            vdata_2d=self.fdata,
            param_name="<vfield_2d.fdata>",
        )
        _field_types.ensure_field_metadata(
            field=self,
            num_comps=2,
            num_sdims=2,
            num_ranks=1,
            param_name="<vfield_2d>",
        )


##
## === 2D FIELD VALIDATION
##


def ensure_2d_sfield(
    sfield_2d: ScalarField_2D,
    *,
    param_name: str = "<sfield_2d>",
) -> None:
    type_manager.ensure_type(
        param=sfield_2d,
        param_name=param_name,
        valid_types=ScalarField_2D,
    )


def ensure_2d_vfield(
    vfield_2d: VectorField_2D,
    *,
    param_name: str = "<vfield_2d>",
) -> None:
    type_manager.ensure_type(
        param=vfield_2d,
        param_name=param_name,
        valid_types=VectorField_2D,
    )


## } MODULE
