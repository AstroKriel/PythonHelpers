## { MODULE

##
## === DEPENDENCIES
##

import numpy

from typing import Self
from dataclasses import dataclass

from jormi.ww_types import type_manager, array_checks, fdata_types, cartesian_coordinates, domain_types
from jormi.ww_fields import fdata_operators

##
## === BASE FIELD TYPE
##


@dataclass(frozen=True)
class Field:
    """
    Generic field: `FieldData` + `UniformDomain` + label + (optional) simulation time.

    Specialised field types (e.g. ScalarField, VectorField) build on this and
    add additional constraints on the underlying `FieldData` and metadata.
    """

    fdata: fdata_types.FieldData
    udomain: domain_types.UniformDomain
    field_label: str
    sim_time: float | None = None

    def __post_init__(
        self,
    ) -> None:
        self._validate_fdata()
        self._validate_udomain()
        self._validate_label()
        self._validate_sim_time()

    def _validate_fdata(
        self,
    ) -> None:
        fdata_types.ensure_fdata(
            fdata=self.fdata,
            param_name="<field.fdata>",
        )

    def _validate_udomain(
        self,
    ) -> None:
        domain_types.ensure_udomain(
            udomain=self.udomain,
            param_name="<field.udomain>",
        )
        if self.fdata.sdims_shape != self.udomain.resolution:
            raise ValueError(
                "`Field` data-array shape does not match domain resolution:"
                f" sdims_shape={self.fdata.sdims_shape},"
                f" resolution={self.udomain.resolution}.",
            )

    def _validate_label(
        self,
    ) -> None:
        type_manager.ensure_nonempty_string(
            param=self.field_label,
            param_name="<field_label>",
        )

    def _validate_sim_time(
        self,
    ) -> None:
        type_manager.ensure_finite_float(
            param=self.sim_time,
            param_name="<sim_time>",
            allow_none=True,
        )


##
## === SPECIALISED FIELD TYPES
##


@dataclass(frozen=True)
class ScalarField(Field):
    """3D scalar field: num_ranks=0, num_comps=1, num_sdims=3."""

    fdata: fdata_types.ScalarFieldData

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()
        self._validate_sdata()

    def _validate_sdata(
        self,
    ) -> None:
        fdata_types.ensure_3d_sdata(
            sdata=self.fdata,
            param_name="<sfield.fdata>",
        )


@dataclass(frozen=True)
class VectorField(Field):
    """3D vector field: num_ranks=1, num_comps=3, num_sdims=3."""

    fdata: fdata_types.VectorFieldData
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
        fdata_types.ensure_3d_vdata(
            vdata=self.fdata,
            param_name="<vfield.fdata>",
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
        ## validate here, rather than in the fdata-module, since
        ## `fdata_operators.sum_of_squared_components` would yield a circular import there
        q_magn_sq_sarray = fdata_operators.sum_of_squared_components(
            vdata=self.fdata,
        )
        if not numpy.all(numpy.isfinite(q_magn_sq_sarray)):
            raise ValueError("UnitVectorField should not contain any NaN/Inf magnitudes.")
        if numpy.any(q_magn_sq_sarray <= 1e-300):
            raise ValueError("UnitVectorField should not contain any (near-)zero vectors.")
        max_error = float(numpy.max(numpy.abs(q_magn_sq_sarray - 1.0)))
        if max_error > self.tol:
            raise ValueError(
                f"Vector magnitude deviates from unit-magnitude=1.0 by"
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
    """Zero-copy rewrap with validation."""
    return UnitVectorField.from_vfield(vfield, tol=tol)


##
## === TYPE VALIDATION
##


def _ensure_field(
    field: Field,
    *,
    param_name: str = "<field>",
) -> None:
    type_manager.ensure_type(
        param=field,
        param_name=param_name,
        valid_types=Field,
    )


def ensure_field_metadata(
    field: Field,
    *,
    num_comps: int | None = None,
    num_sdims: int | None = None,
    num_ranks: int | None = None,
    param_name: str = "<field>",
) -> None:
    """
    Ensure the `field` metadata matches the requested properties.

    Any of `num_comps`, `num_sdims`, or `num_ranks` can be left as `None`
    to skip that check.
    """
    _ensure_field(
        field=field,
        param_name=param_name,
    )
    fdata_types.ensure_fdata_metadata(
        fdata=field.fdata,
        num_comps=num_comps,
        num_sdims=num_sdims,
        num_ranks=num_ranks,
        param_name=f"{param_name}.fdata",
    )


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


def _ensure_udomain_matches_field(
    *,
    field: Field,
    udomain: domain_types.UniformDomain,
    domain_name: str = "<udomain>",
    field_name: str = "<field>",
) -> None:
    """Ensure UniformDomain matches Field."""
    domain_types.ensure_udomain(
        udomain=udomain,
        param_name=domain_name,
    )
    _ensure_field(
        field=field,
        param_name=field_name,
    )
    if field.udomain != udomain:
        raise ValueError(
            f"{field_name}.udomain does not match {domain_name}.",
        )
    if field.fdata.sdims_shape != udomain.resolution:
        raise ValueError(
            f"{field_name}.fdata.sdims_shape={field.fdata.sdims_shape}"
            f" does not match {domain_name}.resolution={udomain.resolution}.",
        )


def ensure_udomain_matches_sfield(
    *,
    sfield: ScalarField,
    udomain: domain_types.UniformDomain,
    domain_name: str = "<udomain>",
    sfield_name: str = "<sfield>",
) -> None:
    """Ensure UniformDomain matches ScalarField."""
    ensure_sfield(
        sfield=sfield,
        param_name=sfield_name,
    )
    _ensure_udomain_matches_field(
        udomain=udomain,
        field=sfield,
        domain_name=domain_name,
        field_name=sfield_name,
    )


def ensure_udomain_matches_vfield(
    *,
    vfield: VectorField,
    udomain: domain_types.UniformDomain,
    domain_name: str = "<udomain>",
    vfield_name: str = "<vfield>",
) -> None:
    """Ensure UniformDomain matches VectorField."""
    ensure_vfield(
        vfield=vfield,
        param_name=vfield_name,
    )
    _ensure_udomain_matches_field(
        udomain=udomain,
        field=vfield,
        domain_name=domain_name,
        field_name=vfield_name,
    )


def ensure_same_field_shape(
    *,
    field_a: Field,
    field_b: Field,
    field_name_a: str = "<field_a>",
    field_name_b: str = "<field_b>",
) -> None:
    """
    Ensure two Field instances have shape-compatible data arrays.

    This checks that `field_a.fdata.farray` and `field_b.fdata.farray` have
    the same shape.
    """
    _ensure_field(
        field=field_a,
        param_name=field_name_a,
    )
    _ensure_field(
        field=field_b,
        param_name=field_name_b,
    )
    array_checks.ensure_same_shape(
        array_a=field_a.fdata.farray,
        array_b=field_b.fdata.farray,
        param_name_a=f"{field_name_a}.fdata.farray",
        param_name_b=f"{field_name_b}.fdata.farray",
    )


## } MODULE
