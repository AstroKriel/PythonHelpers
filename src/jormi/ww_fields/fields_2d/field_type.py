## { MODULE

##
## === DEPENDENCIES
##

import numpy

from typing import Self
from dataclasses import dataclass

from jormi.ww_types import type_checks
from jormi.ww_fields import _field_type
from jormi.ww_fields.fields_2d import (
    domain_type,
    _fdata_type,
)

##
## === 2D FIELD TYPES
##


@dataclass(frozen=True)
class ScalarField_2D(_field_type.Field):
    """2D scalar field: `num_ranks == 0`, `num_comps == 1`, `num_sdims == 2`."""

    fdata: _fdata_type.ScalarFieldData_2D
    udomain: domain_type.UniformDomain_2D

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()
        self._validate_sdata()

    def _validate_sdata(
        self,
    ) -> None:
        _fdata_type.ensure_2d_sdata(
            sdata_2d=self.fdata,
            param_name="<sfield_2d.fdata>",
        )
        _field_type.ensure_field_metadata(
            field=self,
            num_comps=1,
            num_sdims=2,
            num_ranks=0,
            param_name="<sfield_2d>",
        )

    @classmethod
    def from_2d_sarray(
        cls,
        *,
        sarray_2d: numpy.ndarray,
        udomain_2d: domain_type.UniformDomain_2D,
        field_label: str,
        sim_time: float | None = None,
    ) -> Self:
        """Construct a 2D scalar field from a (Nx, Ny) ndarray."""
        _fdata_type.ensure_2d_sarray(
            sarray_2d=sarray_2d,
            param_name="<sarray_2d>",
        )
        sdata_2d = _fdata_type.ScalarFieldData_2D(
            farray=sarray_2d,
            param_name="<sdata_2d>",
        )
        return cls(
            fdata=sdata_2d,
            udomain=udomain_2d,
            field_label=field_label,
            sim_time=sim_time,
        )

    @property
    def is_sliced_from_3d(
        self,
    ) -> bool:
        """Return True if the underlying domain is a 3D-sliced 2D domain_type."""
        return isinstance(self.udomain, domain_type.UniformDomain_2D_Sliced3D)


@dataclass(frozen=True)
class VectorField_2D(_field_type.Field):
    """2D vector field: `num_ranks == 1`, `num_comps == 2`, `num_sdims == 2`."""

    fdata: _fdata_type.VectorFieldData_2D
    udomain: domain_type.UniformDomain_2D

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()
        self._validate_vdata()

    def _validate_vdata(
        self,
    ) -> None:
        _fdata_type.ensure_2d_vdata(
            vdata_2d=self.fdata,
            param_name="<vfield_2d.fdata>",
        )
        _field_type.ensure_field_metadata(
            field=self,
            num_comps=2,
            num_sdims=2,
            num_ranks=1,
            param_name="<vfield_2d>",
        )

    @classmethod
    def from_2d_varray(
        cls,
        *,
        varray_2d: numpy.ndarray,
        udomain_2d: domain_type.UniformDomain_2D,
        field_label: str,
        sim_time: float | None = None,
    ) -> Self:
        """Construct a 2D vector field from a (2, Nx, Ny) ndarray."""
        _fdata_type.ensure_2d_varray(
            varray_2d=varray_2d,
            param_name="<varray_2d>",
        )
        vdata_2d = _fdata_type.VectorFieldData_2D(
            farray=varray_2d,
            param_name="<vdata_2d>",
        )
        return cls(
            fdata=vdata_2d,
            udomain=udomain_2d,
            field_label=field_label,
            sim_time=sim_time,
        )

    @property
    def is_sliced_from_3d(
        self,
    ) -> bool:
        """Return True if the underlying domain is a 3D-sliced 2D domain_type."""
        return isinstance(self.udomain, domain_type.UniformDomain_2D_Sliced3D)


##
## === 2D FIELD VALIDATION
##


def ensure_2d_sfield(
    sfield_2d: ScalarField_2D,
    *,
    param_name: str = "<sfield_2d>",
) -> None:
    type_checks.ensure_type(
        param=sfield_2d,
        param_name=param_name,
        valid_types=ScalarField_2D,
    )


def ensure_2d_vfield(
    vfield_2d: VectorField_2D,
    *,
    param_name: str = "<vfield_2d>",
) -> None:
    type_checks.ensure_type(
        param=vfield_2d,
        param_name=param_name,
        valid_types=VectorField_2D,
    )


def ensure_2d_sfield_sliced_from_3d(
    sfield_2d: ScalarField_2D,
    *,
    param_name: str = "<sfield_2d>",
) -> None:
    """Ensure `sfield_2d` is ScalarField_2D with a 3D-sliced 2D domain_type."""
    ensure_2d_sfield(
        sfield_2d=sfield_2d,
        param_name=param_name,
    )
    domain_type.ensure_2d_udomain_sliced_from_3d(
        udomain_2d=sfield_2d.udomain,
        param_name=f"{param_name}.udomain",
    )


def ensure_2d_vfield_sliced_from_3d(
    vfield_2d: VectorField_2D,
    *,
    param_name: str = "<vfield_2d>",
) -> None:
    """Ensure `vfield_2d` is VectorField_2D with a 3D-sliced 2D domain_type."""
    ensure_2d_vfield(
        vfield_2d=vfield_2d,
        param_name=param_name,
    )
    domain_type.ensure_2d_udomain_sliced_from_3d(
        udomain_2d=vfield_2d.udomain,
        param_name=f"{param_name}.udomain",
    )


## } MODULE
