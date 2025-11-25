## { MODULE

##
## === DEPENDENCIES
##

from dataclasses import dataclass

from jormi.ww_types import type_manager, field_types
from jormi.ww_2d_fields import fdata_types, domain_types


##
## === 2D FIELD TYPES
##


@dataclass(frozen=True)
class ScalarField(field_types.Field):
    """2D scalar field: num_ranks=0, num_comps=1, num_sdims=2."""

    fdata: fdata_types.ScalarFieldData
    udomain: domain_types.UniformDomain

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()
        self._validate_sdata()

    def _validate_sdata(
        self,
    ) -> None:
        fdata_types.ensure_sdata(
            sdata=self.fdata,
            param_name="<sfield.fdata>",
        )
        field_types.ensure_field_metadata(
            field=self,
            num_comps=1,
            num_sdims=2,
            num_ranks=0,
            param_name="<sfield>",
        )


@dataclass(frozen=True)
class VectorField(field_types.Field):
    """2D vector field: num_ranks=1, num_comps=2, num_sdims=2."""

    fdata: fdata_types.VectorFieldData
    udomain: domain_types.UniformDomain

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()
        self._validate_vdata()

    def _validate_vdata(
        self,
    ) -> None:
        fdata_types.ensure_vdata(
            vdata=self.fdata,
            param_name="<vfield.fdata>",
        )
        field_types.ensure_field_metadata(
            field=self,
            num_comps=2,
            num_sdims=2,
            num_ranks=1,
            param_name="<vfield>",
        )


##
## === 2D FIELD VALIDATION
##


def ensure_sfield(
    sfield: ScalarField,
    *,
    param_name: str = "<sfield>",
) -> None:
    type_manager.ensure_type(
        param=sfield,
        param_name=param_name,
        valid_types=ScalarField,
    )


def ensure_vfield(
    vfield: VectorField,
    *,
    param_name: str = "<vfield>",
) -> None:
    type_manager.ensure_type(
        param=vfield,
        param_name=param_name,
        valid_types=VectorField,
    )


## } MODULE
