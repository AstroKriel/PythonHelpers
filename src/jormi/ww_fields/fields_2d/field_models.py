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
from jormi.ww_fields import _field_models
from jormi.ww_fields.fields_2d import (
    _field_data,
    domain_models,
)
from jormi.ww_validation import validate_types

##
## === 2D FIELD TYPES
##


@dataclass(frozen=True)
class ScalarField_2D(_field_models.Field):
    """2D scalar field: `num_ranks == 0`, `num_comps == 1`, `num_sdims == 2`."""

    fdata: _field_data.ScalarFieldData_2D
    udomain: domain_models.UniformDomain_2D

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()
        _field_data.ensure_2d_sdata(
            sdata_2d=self.fdata,
            param_name="<sfield_2d.fdata>",
        )

    @classmethod
    def from_2d_sarray(
        cls,
        *,
        sarray_2d: NDArray[Any],
        udomain_2d: domain_models.UniformDomain_2D,
        field_name: str,
        latex_label: str,
        sim_time: float | None = None,
    ) -> Self:
        """Construct a 2D scalar field from a (num_x0_cells, num_x1_cells) ndarray."""
        sdata_2d = _field_data.ScalarFieldData_2D(
            farray=sarray_2d,
            param_name="<sdata_2d>",
        )
        return cls(
            fdata=sdata_2d,
            udomain=udomain_2d,
            field_name=field_name,
            latex_label=latex_label,
            sim_time=sim_time,
        )

    @property
    def is_sliced_from_3d(
        self,
    ) -> bool:
        """Return True if the underlying domain is a 3D-sliced 2D domain."""
        return isinstance(self.udomain, domain_models.UniformDomain_2D_Sliced3D)


@dataclass(frozen=True)
class VectorField_2D(_field_models.Field):
    """2D vector field: `num_ranks == 1`, `num_comps == 2`, `num_sdims == 2`."""

    fdata: _field_data.VectorFieldData_2D
    udomain: domain_models.UniformDomain_2D

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()
        _field_data.ensure_2d_vdata(
            vdata_2d=self.fdata,
            param_name="<vfield_2d.fdata>",
        )

    @classmethod
    def from_2d_varray(
        cls,
        *,
        varray_2d: NDArray[Any],
        udomain_2d: domain_models.UniformDomain_2D,
        field_name: str,
        latex_label: str,
        sim_time: float | None = None,
    ) -> Self:
        """Construct a 2D vector field from a (2, num_x0_cells, num_x1_cells) ndarray."""
        vdata_2d = _field_data.VectorFieldData_2D(
            farray=varray_2d,
            param_name="<vdata_2d>",
        )
        return cls(
            fdata=vdata_2d,
            udomain=udomain_2d,
            field_name=field_name,
            latex_label=latex_label,
            sim_time=sim_time,
        )

    @property
    def is_sliced_from_3d(
        self,
    ) -> bool:
        """Return True if the underlying domain is a 3D-sliced 2D domain."""
        return isinstance(self.udomain, domain_models.UniformDomain_2D_Sliced3D)


@dataclass(frozen=True)
class SlicedVectorFields_2D:
    """
    In-plane `VectorField_2D` (2 comps) paired with an out-of-plane `ScalarField_2D`
    (1 comp); jormi has no single 2D field type for a 3-component vector, so this
    bundle covers the case instead.
    """

    inplane_vfield_2d: VectorField_2D
    outofplane_sfield_2d: ScalarField_2D

    def __post_init__(
        self,
    ) -> None:
        ensure_2d_vfield(
            vfield_2d=self.inplane_vfield_2d,
            param_name="<inplane_vfield_2d>",
        )
        ensure_2d_sfield(
            sfield_2d=self.outofplane_sfield_2d,
            param_name="<outofplane_sfield_2d>",
        )
        if self.inplane_vfield_2d.udomain != self.outofplane_sfield_2d.udomain:
            raise ValueError(
                "<inplane_vfield_2d>.udomain does not match <outofplane_sfield_2d>.udomain.",
            )

    @classmethod
    def from_2d_varray(
        cls,
        *,
        varray_2d: NDArray[Any],
        udomain_2d: domain_models.UniformDomain_2D,
        field_name: str,
        latex_label: str,
        sim_time: float | None = None,
    ) -> Self:
        """
        Construct from a (3, num_x0_cells, num_x1_cells) ndarray.

        Components [0, 1] become the in-plane vector; component [2] becomes
        the out-of-plane scalar.
        """
        return cls(
            inplane_vfield_2d=VectorField_2D.from_2d_varray(
                varray_2d=varray_2d[:2],
                udomain_2d=udomain_2d,
                field_name=f"{field_name}_inplane",
                latex_label=latex_label,
                sim_time=sim_time,
            ),
            outofplane_sfield_2d=ScalarField_2D.from_2d_sarray(
                sarray_2d=varray_2d[2],
                udomain_2d=udomain_2d,
                field_name=f"{field_name}_outofplane",
                latex_label=latex_label,
                sim_time=sim_time,
            ),
        )


##
## === 2D FIELD VALIDATION
##


def ensure_2d_sfield(
    sfield_2d: ScalarField_2D,
    *,
    param_name: str = "<sfield_2d>",
) -> None:
    validate_types.ensure_type(
        param=sfield_2d,
        param_name=param_name,
        valid_types=ScalarField_2D,
    )


def ensure_2d_vfield(
    vfield_2d: VectorField_2D,
    *,
    param_name: str = "<vfield_2d>",
) -> None:
    validate_types.ensure_type(
        param=vfield_2d,
        param_name=param_name,
        valid_types=VectorField_2D,
    )


def ensure_2d_sfield_sliced_from_3d(
    sfield_2d: ScalarField_2D,
    *,
    param_name: str = "<sfield_2d>",
) -> None:
    """Ensure `sfield_2d` is ScalarField_2D with a 3D-sliced 2D domain."""
    ensure_2d_sfield(
        sfield_2d=sfield_2d,
        param_name=param_name,
    )
    domain_models.ensure_2d_udomain_sliced_from_3d(
        udomain_2d=sfield_2d.udomain,
        param_name=f"{param_name}.udomain",
    )


def ensure_2d_vfield_sliced_from_3d(
    vfield_2d: VectorField_2D,
    *,
    param_name: str = "<vfield_2d>",
) -> None:
    """Ensure `vfield_2d` is VectorField_2D with a 3D-sliced 2D domain."""
    ensure_2d_vfield(
        vfield_2d=vfield_2d,
        param_name=param_name,
    )
    domain_models.ensure_2d_udomain_sliced_from_3d(
        udomain_2d=vfield_2d.udomain,
        param_name=f"{param_name}.udomain",
    )


##
## === EXTRACT NDARRAY FROM FIELDS
##


def extract_2d_sarray(
    sfield_2d: ScalarField_2D,
    *,
    param_name: str = "<sfield_2d>",
) -> NDArray[Any]:
    """Return the underlying (num_x0_cells, num_x1_cells) ndarray for a 2D scalar field."""
    ensure_2d_sfield(
        sfield_2d=sfield_2d,
        param_name=param_name,
    )
    return sfield_2d.fdata.farray


def extract_2d_varray(
    vfield_2d: VectorField_2D,
    *,
    param_name: str = "<vfield_2d>",
) -> NDArray[Any]:
    """Return the underlying (2, num_x0_cells, num_x1_cells) ndarray for a 2D vector field."""
    ensure_2d_vfield(
        vfield_2d=vfield_2d,
        param_name=param_name,
    )
    return vfield_2d.fdata.farray


## } MODULE
