## { MODULE

##
## === DEPENDENCIES
##

import numpy

from typing import Self
from dataclasses import dataclass

from jormi.ww_types import (
    type_manager,
    cartesian_coordinates,
    field_types as base_field_types,
)
from jormi.ww_3d_fields import (
    fdata_types,
    fdata_operators,
    domain_types,
)


##
## === 3D FIELD TYPES
##


@dataclass(frozen=True)
class ScalarField(base_field_types.Field):
    """3D scalar field: num_ranks=0, num_comps=1, num_sdims=3."""

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
        base_field_types.ensure_field_metadata(
            field=self,
            num_comps=1,
            num_sdims=3,
            num_ranks=0,
            param_name="<sfield>",
        )


@dataclass(frozen=True)
class VectorField(base_field_types.Field):
    """3D vector field: num_ranks=1, num_comps=3, num_sdims=3."""

    fdata: fdata_types.VectorFieldData
    udomain: domain_types.UniformDomain
    comp_axes: cartesian_coordinates.AxisTuple = cartesian_coordinates.DEFAULT_AXES_ORDER

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()
        self._validate_vdata()
        self._validate_axes()

    def _validate_vdata(
        self,
    ) -> None:
        fdata_types.ensure_vdata(
            vdata=self.fdata,
            param_name="<vfield.fdata>",
        )
        base_field_types.ensure_field_metadata(
            field=self,
            num_comps=3,
            num_sdims=3,
            num_ranks=1,
            param_name="<vfield>",
        )

    def _validate_axes(
        self,
    ) -> None:
        cartesian_coordinates.ensure_default_axes_order(
            axes=self.comp_axes,
            param_name="<comp_axes>",
        )
        if self.fdata.num_comps != len(self.comp_axes):
            raise ValueError(
                "VectorField component axes must match number of components:"
                f" num_comps={self.fdata.num_comps},"
                f" len(comp_axes)={len(self.comp_axes)}.",
            )

    def get_comp_data(
        self,
        comp_axis: cartesian_coordinates.AxisLike,
    ) -> numpy.ndarray:
        """Return a (Nx, Ny, Nz) view of the requested component."""
        comp_index = cartesian_coordinates.get_axis_index(comp_axis)
        return self.fdata.farray[comp_index, ...]


@dataclass(frozen=True)
class UnitVectorField(VectorField):
    """3D vector field with unit-magnitude vectors at each cell."""

    tol: float = 1e-6

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()
        self._validate_unit_magnitude()

    def _validate_unit_magnitude(
        self,
    ) -> None:
        ## validate here, rather than in the fdata_types module, since the following
        ## fn-call would yield a circular import there
        f_magn_sq_sarray = fdata_operators.sum_of_varray_comps_squared(
            vdata=self.fdata,
        )
        if not numpy.all(numpy.isfinite(f_magn_sq_sarray)):
            raise ValueError("UnitVectorField should not contain any NaN/Inf magnitudes.")
        if numpy.any(f_magn_sq_sarray <= 1e-300):
            raise ValueError("UnitVectorField should not contain any (near-)zero vectors.")
        max_error = float(numpy.max(numpy.abs(f_magn_sq_sarray - 1.0)))
        if max_error > self.tol:
            raise ValueError(
                "Vector magnitude deviates from unit-magnitude=1.0 by"
                f" max(error)={max_error:.3e} (tol={self.tol}).",
            )

    @classmethod
    def from_vfield(
        cls,
        vfield: VectorField,
        *,
        tol: float = 1e-6,
    ) -> Self:
        return cls(
            fdata=vfield.fdata,
            udomain=vfield.udomain,
            field_label=vfield.field_label,
            comp_axes=vfield.comp_axes,
            sim_time=vfield.sim_time,
            tol=tol,
        )


def as_uvfield(
    vfield: VectorField,
    tol: float = 1e-6,
) -> UnitVectorField:
    """Zero-copy rewrap of a 3D VectorField into a UnitVectorField with validation."""
    return UnitVectorField.from_vfield(
        vfield=vfield,
        tol=tol,
    )


##
## === 3D FIELD VALIDATION
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


def ensure_uvfield(
    uvfield: UnitVectorField,
    *,
    param_name: str = "<uvfield>",
) -> None:
    type_manager.ensure_type(
        param=uvfield,
        param_name=param_name,
        valid_types=UnitVectorField,
    )


def ensure_udomain_matches_sfield(
    *,
    sfield: ScalarField,
    udomain: domain_types.UniformDomain,
    domain_name: str = "<udomain>",
    sfield_name: str = "<sfield>",
) -> None:
    """Ensure UniformDomain matches a 3D ScalarField."""
    ensure_sfield(
        sfield=sfield,
        param_name=sfield_name,
    )
    base_field_types._ensure_udomain_matches_field(
        udomain=udomain,
        field=sfield,
        domain_name=domain_name,
        field_name=sfield_name,
    )


def ensure_udomain_matches_vfield(
    *,
    udomain: domain_types.UniformDomain,
    vfield: VectorField,
    domain_name: str = "<udomain>",
    vfield_name: str = "<vfield>",
) -> None:
    """Ensure UniformDomain matches a 3D VectorField."""
    ensure_vfield(
        vfield=vfield,
        param_name=vfield_name,
    )
    base_field_types._ensure_udomain_matches_field(
        udomain=udomain,
        field=vfield,
        domain_name=domain_name,
        field_name=vfield_name,
    )


def ensure_same_field_shape(
    *,
    field_a: base_field_types.Field,
    field_b: base_field_types.Field,
    field_name_a: str = "<field_a>",
    field_name_b: str = "<field_b>",
) -> None:
    """
    Ensure two 3D Field instances have shape-compatible data arrays.

    This is a thin wrapper around the base ww_types.field_types helper,
    re-exposed here so 3D code can stay within jormi.ww_3d_fields.field_types.
    """
    base_field_types.ensure_same_field_shape(
        field_a=field_a,
        field_b=field_b,
        field_name_a=field_name_a,
        field_name_b=field_name_b,
    )


## } MODULE
