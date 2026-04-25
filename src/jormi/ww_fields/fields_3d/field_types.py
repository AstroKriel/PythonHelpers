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
    _field_types,
    cartesian_axes,
)
from jormi.ww_arrays.farrays_3d import farray_operators
from jormi.ww_fields.fields_3d import (
    fdata_types,
    domain_types,
)
from jormi.ww_validation import validate_types

##
## === 3D FIELD TYPES
##


@dataclass(frozen=True)
class ScalarField_3D(_field_types.Field):
    """3D scalar field: num_ranks=0, num_comps=1, num_sdims=3."""

    fdata: fdata_types.ScalarFieldData_3D
    udomain: domain_types.UniformDomain_3D

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()
        self._ensure_sdata()

    def _ensure_sdata(
        self,
    ) -> None:
        fdata_types.ensure_3d_sdata(
            sdata_3d=self.fdata,
            param_name="<sfield_3d.fdata>",
        )
        _field_types.ensure_field_metadata(
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
        sarray_3d: NDArray[Any],
        udomain_3d: domain_types.UniformDomain_3D,
        field_label: str,
        sim_time: float | None = None,
    ) -> "ScalarField_3D":
        """Construct a 3D scalar field directly from a (num_x0_cells, num_x1_cells, num_x2_cells) ndarray and a 3D UniformDomain."""
        return cls._from_farray(
            farray=sarray_3d,
            udomain=udomain_3d,
            field_label=field_label,
            sim_time=sim_time,
            fdata_fn=fdata_types.ScalarFieldData_3D,
            fdata_param_name="<sarray_3d>",
        )


@dataclass(frozen=True)
class VectorField_3D(_field_types.Field):
    """3D vector field: num_ranks=1, num_comps=3, num_sdims=3."""

    fdata: fdata_types.VectorFieldData_3D
    udomain: domain_types.UniformDomain_3D
    comp_axes: cartesian_axes.AxisTuple_3D = cartesian_axes.DEFAULT_3D_AXES_ORDER

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()
        self._ensure_vdata()
        self._ensure_axes()

    def _ensure_vdata(
        self,
    ) -> None:
        fdata_types.ensure_3d_vdata(
            vdata_3d=self.fdata,
            param_name="<vfield_3d.fdata>",
        )
        _field_types.ensure_field_metadata(
            field=self,
            num_comps=3,
            num_sdims=3,
            num_ranks=1,
            param_name="<vfield_3d>",
        )

    def _ensure_axes(
        self,
    ) -> None:
        validate_types.ensure_type(
            param=self.comp_axes,
            param_name="<comp_axes>",
            valid_types=tuple,
        )
        if self.comp_axes != cartesian_axes.DEFAULT_3D_AXES_ORDER:
            raise ValueError(
                "`<comp_axes>` must equal cartesian_axes.DEFAULT_3D_AXES_ORDER:"
                f" got {self.comp_axes!r}.",
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
        varray_3d: NDArray[Any],
        udomain_3d: domain_types.UniformDomain_3D,
        field_label: str,
        sim_time: float | None = None,
    ) -> "VectorField_3D":
        """Construct a 3D vector field directly from a (3, num_x0_cells, num_x1_cells, num_x2_cells) ndarray and a 3D UniformDomain."""
        return cls._from_farray(
            farray=varray_3d,
            udomain=udomain_3d,
            field_label=field_label,
            sim_time=sim_time,
            fdata_fn=fdata_types.VectorFieldData_3D,
            fdata_param_name="<varray_3d>",
        )

    def get_vcomp_3d_sarray(
        self,
        comp_axis: cartesian_axes.AxisLike_3D,
    ) -> NDArray[Any]:
        """Return a (num_x0_cells, num_x1_cells, num_x2_cells) view of the requested component."""
        comp_index = cartesian_axes.get_axis_index(comp_axis)
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
        self._ensure_unit_magnitude()

    def _ensure_unit_magnitude(
        self,
    ) -> None:
        varray_3d = extract_3d_varray(
            vfield_3d=self,
            param_name="<uvfield_3d>",
        )
        farray_operators.ensure_uvarray_magnitude(
            varray_3d=varray_3d,
            tol=self.tol,
            param_name="<uvfield_3d>",
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
    *,
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
    *,
    sfield_3d: ScalarField_3D,
    param_name: str = "<sfield_3d>",
) -> None:
    validate_types.ensure_type(
        param=sfield_3d,
        param_name=param_name,
        valid_types=ScalarField_3D,
    )


def ensure_3d_vfield(
    *,
    vfield_3d: VectorField_3D,
    param_name: str = "<vfield_3d>",
) -> None:
    validate_types.ensure_type(
        param=vfield_3d,
        param_name=param_name,
        valid_types=VectorField_3D,
    )


def ensure_3d_uvfield(
    *,
    uvfield_3d: UnitVectorField_3D,
    param_name: str = "<uvfield_3d>",
) -> None:
    validate_types.ensure_type(
        param=uvfield_3d,
        param_name=param_name,
        valid_types=UnitVectorField_3D,
    )


def ensure_3d_udomain_matches_sfield(
    *,
    sfield_3d: ScalarField_3D,
    udomain_3d: domain_types.UniformDomain_3D,
    domain_name: str = "<udomain_3d>",
    sfield_name: str = "<sfield_3d>",
) -> None:
    """Ensure UniformDomain matches a 3D ScalarField_3D."""
    ensure_3d_sfield(
        sfield_3d=sfield_3d,
        param_name=sfield_name,
    )
    _field_types.ensure_udomain_matches_field(
        udomain=udomain_3d,
        field=sfield_3d,
        domain_name=domain_name,
        field_name=sfield_name,
    )


def ensure_3d_udomain_matches_vfield(
    *,
    udomain_3d: domain_types.UniformDomain_3D,
    vfield_3d: VectorField_3D,
    domain_name: str = "<udomain_3d>",
    vfield_name: str = "<vfield_3d>",
) -> None:
    """Ensure UniformDomain matches a 3D VectorField_3D."""
    ensure_3d_vfield(
        vfield_3d=vfield_3d,
        param_name=vfield_name,
    )
    _field_types.ensure_udomain_matches_field(
        udomain=udomain_3d,
        field=vfield_3d,
        domain_name=domain_name,
        field_name=vfield_name,
    )


def ensure_same_3d_field_shape(
    *,
    field_3d_a: _field_types.Field,
    field_3d_b: _field_types.Field,
    field_name_a: str = "<field_3d_a>",
    field_name_b: str = "<field_3d_b>",
) -> None:
    """Ensure two 3D Field instances have shape-compatible data arrays."""
    _field_types.ensure_field_metadata(
        field=field_3d_a,
        param_name=field_name_a,
        num_sdims=3,
    )
    _field_types.ensure_field_metadata(
        field=field_3d_b,
        param_name=field_name_b,
        num_sdims=3,
    )
    _field_types.ensure_same_field_shape(
        field_a=field_3d_a,
        field_b=field_3d_b,
        field_name_a=field_name_a,
        field_name_b=field_name_b,
    )


def ensure_same_3d_field_udomains(
    *,
    field_3d_a: _field_types.Field,
    field_3d_b: _field_types.Field,
    field_name_a: str = "<field_3d_a>",
    field_name_b: str = "<field_3d_b>",
) -> None:
    """Ensure two 3D Field instances have matching UniformDomain objects."""
    _field_types.ensure_field_metadata(
        field=field_3d_a,
        param_name=field_name_a,
        num_sdims=3,
    )
    _field_types.ensure_field_metadata(
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
    field_3d_a: _field_types.Field,
    field_3d_b: _field_types.Field,
    field_name_a: str = "<field_3d_a>",
    field_name_b: str = "<field_3d_b>",
) -> None:
    """Ensure two 3D Field instances have matching data shapes and UniformDomains."""
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
    *,
    sfield_3d: ScalarField_3D,
    param_name: str = "<sfield_3d>",
) -> NDArray[Any]:
    """Validate and extract the underlying (num_x0_cells, num_x1_cells, num_x2_cells) ndarray from a 3D scalar field."""
    ensure_3d_sfield(
        sfield_3d=sfield_3d,
        param_name=param_name,
    )
    return fdata_types.extract_3d_sarray(
        sdata_3d=sfield_3d.fdata,
        param_name=f"{param_name}.fdata",
    )


def extract_3d_varray(
    *,
    vfield_3d: VectorField_3D,
    param_name: str = "<vfield_3d>",
) -> NDArray[Any]:
    """Validate and extract the underlying (3, num_x0_cells, num_x1_cells, num_x2_cells) ndarray from a 3D vector field."""
    ensure_3d_vfield(
        vfield_3d=vfield_3d,
        param_name=param_name,
    )
    return fdata_types.extract_3d_varray(
        vdata_3d=vfield_3d.fdata,
        param_name=f"{param_name}.fdata",
    )


##
## === RENDERING FIELD LABEL
##


def get_label(
    *,
    field: _field_types.Field,
    param_name: str = "<field>",
) -> str:
    """Return the render-ready label for any field: wraps `field.field_label` in `$...$`."""
    validate_types.ensure_type(
        param=field,
        param_name=param_name,
        valid_types=_field_types.Field,
    )
    return f"${field.field_label}$"


def get_vcomp_label(
    *,
    vfield_3d: VectorField_3D,
    comp_axis: cartesian_axes.AxisLike_3D,
    param_name: str = "<vfield_3d>",
) -> str:
    """Return the render-ready label for a vector field component.

    Uses big square brackets with a numeric subscript.
    Example: vfield with label `\\vec{v}` + axis X0 -> `$\\left[\\vec{v}\\right]_0$`
    """
    ensure_3d_vfield(
        vfield_3d=vfield_3d,
        param_name=param_name,
    )
    comp_index = cartesian_axes.get_axis_index(comp_axis)
    return f"$\\left[{vfield_3d.field_label}\\right]_{comp_index}$"


## } MODULE