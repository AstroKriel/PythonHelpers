## { MODULE

##
## === DEPENDENCIES
##

import numpy

from typing import Self
from dataclasses import dataclass

from jormi.ww_types import type_checks
from jormi.ww_fields import (
    _cartesian_coordinates,
    _field_type,
)
from jormi.ww_fields.fields_3d import (
    _farray_operators,
    _fdata_type,
    domain_type,
)

##
## === 3D FIELD TYPES
##


@dataclass(frozen=True)
class ScalarField_3D(_field_type.Field):
    """3D scalar field: num_ranks=0, num_comps=1, num_sdims=3."""

    fdata: _fdata_type.ScalarFieldData_3D
    udomain: domain_type.UniformDomain_3D

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()
        self._validate_sdata()

    def _validate_sdata(
        self,
    ) -> None:
        _fdata_type.ensure_3d_sdata(
            sdata_3d=self.fdata,
            param_name="<sfield_3d.fdata>",
        )
        _field_type.ensure_field_metadata(
            field=self,
            num_comps=1,
            num_sdims=3,
            num_ranks=0,
            param_name="<sfield_3d>",
        )

    @classmethod
    def from_3d_sarray(
        cls,
        *,
        sarray_3d,
        udomain_3d: domain_type.UniformDomain_3D,
        field_label: str,
        sim_time: float | None = None,
    ) -> "ScalarField_3D":
        """Construct a 3D scalar field directly from a (Nx, Ny, Nz) ndarray and a 3D UniformDomain."""
        return cls._from_farray(
            farray=sarray_3d,
            udomain=udomain_3d,
            field_label=field_label,
            sim_time=sim_time,
            fdata_fn=_fdata_type.ScalarFieldData_3D,
            fdata_param_name="<sarray_3d>",
        )


@dataclass(frozen=True)
class VectorField_3D(_field_type.Field):
    """3D vector field: num_ranks=1, num_comps=3, num_sdims=3."""

    fdata: _fdata_type.VectorFieldData_3D
    udomain: domain_type.UniformDomain_3D
    comp_axes: _cartesian_coordinates.AxisTuple = _cartesian_coordinates.DEFAULT_AXES_ORDER

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()
        self._validate_vdata()
        self._validate_axes()

    def _validate_vdata(
        self,
    ) -> None:
        _fdata_type.ensure_3d_vdata(
            vdata_3d=self.fdata,
            param_name="<vfield_3d.fdata>",
        )
        _field_type.ensure_field_metadata(
            field=self,
            num_comps=3,
            num_sdims=3,
            num_ranks=1,
            param_name="<vfield_3d>",
        )

    def _validate_axes(
        self,
    ) -> None:
        _cartesian_coordinates.ensure_default_axes_order(
            axes=self.comp_axes,
            param_name="<comp_axes>",
        )
        if self.fdata.num_comps != len(self.comp_axes):
            raise ValueError(
                "VectorField_3D component axes must match number of components:"
                f" num_comps={self.fdata.num_comps},"
                f" len(comp_axes)={len(self.comp_axes)}.",
            )

    @classmethod
    def from_3d_varray(
        cls,
        *,
        varray_3d,
        udomain_3d: domain_type.UniformDomain_3D,
        field_label: str,
        sim_time: float | None = None,
    ) -> "VectorField_3D":
        """Construct a 3D vector field directly from a (3, Nx, Ny, Nz) ndarray and a 3D UniformDomain."""
        return cls._from_farray(
            farray=varray_3d,
            udomain=udomain_3d,
            field_label=field_label,
            sim_time=sim_time,
            fdata_fn=_fdata_type.VectorFieldData_3D,
            fdata_param_name="<varray_3d>",
        )

    def get_vcomp_sarray_3d(
        self,
        comp_axis: _cartesian_coordinates.AxisLike,
    ) -> numpy.ndarray:
        """Return a (Nx, Ny, Nz) view of the requested component."""
        comp_index = _cartesian_coordinates.get_axis_index(comp_axis)
        varray_3d = extract_3d_varray(
            vfield_3d=self,
            param_name="<vfield_3d>",
        )
        return varray_3d[comp_index, ...]


@dataclass(frozen=True)
class UnitVectorField_3D(VectorField_3D):
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
        varray_3d = extract_3d_varray(
            vfield_3d=self,
            param_name="<uvfield_3d>",
        )
        ## validate here, rather than in the _fdata_type module, since the following
        ## fn-call would yield a circular import there
        sarray_3d_vmagn_sq = _farray_operators.sum_of_varray_comps_squared(
            varray_3d=varray_3d,
        )
        if not numpy.all(numpy.isfinite(sarray_3d_vmagn_sq)):
            raise ValueError("UnitVectorField_3D should not contain any NaN/Inf magnitudes.")
        if numpy.any(sarray_3d_vmagn_sq <= 1e-300):
            raise ValueError("UnitVectorField_3D should not contain any (near-)zero vectors.")
        max_error = float(
            numpy.max(
                numpy.abs(
                    numpy.sqrt(sarray_3d_vmagn_sq) - 1.0,
                ),
            ),
        )
        if max_error > self.tol:
            raise ValueError(
                "Vector magnitude deviates from unit-magnitude=1.0 by"
                f" max(error)={max_error:.3e} (tol={self.tol}).",
            )

    @classmethod
    def from_3d_vfield(
        cls,
        vfield_3d: VectorField_3D,
        *,
        tol: float = 1e-6,
    ) -> Self:
        return cls(
            fdata=vfield_3d.fdata,
            udomain=vfield_3d.udomain,
            field_label=vfield_3d.field_label,
            comp_axes=vfield_3d.comp_axes,
            sim_time=vfield_3d.sim_time,
            tol=tol,
        )


def as_3d_uvfield(
    vfield_3d: VectorField_3D,
    tol: float = 1e-6,
) -> UnitVectorField_3D:
    """Zero-copy rewrap of a 3D VectorField_3D into a UnitVectorField_3D with validation."""
    return UnitVectorField_3D.from_3d_vfield(
        vfield_3d=vfield_3d,
        tol=tol,
    )


##
## === 3D FIELD VALIDATION
##


def ensure_3d_sfield(
    sfield_3d: ScalarField_3D,
    *,
    param_name: str = "<sfield_3d>",
) -> None:
    type_checks.ensure_type(
        param=sfield_3d,
        param_name=param_name,
        valid_types=ScalarField_3D,
    )


def ensure_3d_vfield(
    vfield_3d: VectorField_3D,
    *,
    param_name: str = "<vfield_3d>",
) -> None:
    type_checks.ensure_type(
        param=vfield_3d,
        param_name=param_name,
        valid_types=VectorField_3D,
    )


def ensure_3d_uvfield(
    uvfield_3d: UnitVectorField_3D,
    *,
    param_name: str = "<uvfield_3d>",
) -> None:
    type_checks.ensure_type(
        param=uvfield_3d,
        param_name=param_name,
        valid_types=UnitVectorField_3D,
    )


def ensure_3d_udomain_matches_sfield(
    *,
    sfield_3d: ScalarField_3D,
    udomain_3d: domain_type.UniformDomain_3D,
    domain_name: str = "<udomain_3d>",
    sfield_name: str = "<sfield_3d>",
) -> None:
    """Ensure UniformDomain matches a 3D ScalarField_3D."""
    ensure_3d_sfield(
        sfield_3d=sfield_3d,
        param_name=sfield_name,
    )
    _field_type.ensure_udomain_matches_field(
        udomain=udomain_3d,
        field=sfield_3d,
        domain_name=domain_name,
        field_name=sfield_name,
    )


def ensure_3d_udomain_matches_vfield(
    *,
    udomain_3d: domain_type.UniformDomain_3D,
    vfield_3d: VectorField_3D,
    domain_name: str = "<udomain_3d>",
    vfield_name: str = "<vfield_3d>",
) -> None:
    """Ensure UniformDomain matches a 3D VectorField_3D."""
    ensure_3d_vfield(
        vfield_3d=vfield_3d,
        param_name=vfield_name,
    )
    _field_type.ensure_udomain_matches_field(
        udomain=udomain_3d,
        field=vfield_3d,
        domain_name=domain_name,
        field_name=vfield_name,
    )


def ensure_same_3d_field_shape(
    *,
    field_3d_a: _field_type.Field,
    field_3d_b: _field_type.Field,
    field_name_a: str = "<field_3d_a>",
    field_name_b: str = "<field_3d_b>",
) -> None:
    """
    Ensure two 3D Field instances have shape-compatible data arrays.

    This is a thin wrapper around the base ww_types.field helper,
    re-exposed here so 3D code can stay within jormi.ww_fields.fields_3d.field.
    """
    _field_type.ensure_field_metadata(
        field=field_3d_a,
        param_name=field_name_a,
        num_sdims=3,
    )
    _field_type.ensure_field_metadata(
        field=field_3d_b,
        param_name=field_name_b,
        num_sdims=3,
    )
    _field_type.ensure_same_field_shape(
        field_a=field_3d_a,
        field_b=field_3d_b,
        field_name_a=field_name_a,
        field_name_b=field_name_b,
    )


def ensure_same_3d_field_udomains(
    *,
    field_3d_a: _field_type.Field,
    field_3d_b: _field_type.Field,
    field_name_a: str = "<field_3d_a>",
    field_name_b: str = "<field_3d_b>",
) -> None:
    """
    Ensure two 3D Field instances have matching UniformDomain objects.

    Parameters
    ----------
    field_3d_a, field_3d_b
        3D Field instances whose udomain attributes should match.
    """
    _field_type.ensure_field_metadata(
        field=field_3d_a,
        param_name=field_name_a,
        num_sdims=3,
    )
    _field_type.ensure_field_metadata(
        field=field_3d_b,
        param_name=field_name_b,
        num_sdims=3,
    )
    if field_3d_a.udomain != field_3d_b.udomain:
        raise ValueError(
            f"`{field_name_a}.udomain` must match `{field_name_b}.udomain`.",
        )


def ensure_same_3d_field_shape_and_udomains(
    *,
    field_3d_a: _field_type.Field,
    field_3d_b: _field_type.Field,
    field_name_a: str = "<field_3d_a>",
    field_name_b: str = "<field_3d_b>",
) -> None:
    """
    Ensure two 3D Field instances have matching data shapes and UniformDomains.

    This is a convenience wrapper that combines `ensure_same_3d_field_shape`
    and `ensure_same_3d_field_udomains` for the common case where both
    conditions are required.
    """
    ensure_same_3d_field_shape(
        field_3d_a=field_3d_a,
        field_3d_b=field_3d_b,
        field_name_a=field_name_a,
        field_name_b=field_name_b,
    )
    ensure_same_3d_field_udomains(
        field_3d_a=field_3d_a,
        field_3d_b=field_3d_b,
        field_name_a=field_name_a,
        field_name_b=field_name_b,
    )


##
## === EXTRACT NDARRAY FROM FIELDS
##


def extract_3d_sarray(
    sfield_3d: ScalarField_3D,
    *,
    param_name: str = "<sfield_3d>",
) -> numpy.ndarray:
    """Validate and extract the underlying (Nx, Ny, Nz) ndarray from a 3D scalar field."""
    ensure_3d_sfield(
        sfield_3d=sfield_3d,
        param_name=param_name,
    )
    return _fdata_type.extract_3d_sarray(
        sdata_3d=sfield_3d.fdata,
        param_name=f"{param_name}.fdata",
    )


def extract_3d_varray(
    vfield_3d: VectorField_3D,
    *,
    param_name: str = "<vfield_3d>",
) -> numpy.ndarray:
    """Validate and extract the underlying (3, Nx, Ny, Nz) ndarray from a 3D vector field."""
    ensure_3d_vfield(
        vfield_3d=vfield_3d,
        param_name=param_name,
    )
    return _fdata_type.extract_3d_varray(
        vdata_3d=vfield_3d.fdata,
        param_name=f"{param_name}.fdata",
    )


## } MODULE
