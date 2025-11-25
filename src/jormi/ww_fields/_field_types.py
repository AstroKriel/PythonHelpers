## { MODULE

##
## === DEPENDENCIES
##

from dataclasses import dataclass

from jormi.ww_types import type_manager, array_checks
from jormi.ww_fields import _fdata_types, _domain_types


##
## === BASE FIELD TYPE
##


@dataclass(frozen=True)
class Field:
    """
    Generic field: `FieldData` + `UniformDomain` + label + (optional) simulation time.

    Specialised field types in 2D/3D libraries build on this and
    add additional constraints on the underlying `FieldData` and metadata.
    """

    fdata: _fdata_types.FieldData
    udomain: _domain_types.UniformDomain
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
        _fdata_types.ensure_fdata(
            fdata=self.fdata,
            param_name="<field.fdata>",
        )

    def _validate_udomain(
        self,
    ) -> None:
        _domain_types.ensure_udomain(
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
## === GENERIC FIELD VALIDATION
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
    _fdata_types.ensure_fdata_metadata(
        fdata=field.fdata,
        num_comps=num_comps,
        num_sdims=num_sdims,
        num_ranks=num_ranks,
        param_name=f"{param_name}.fdata",
    )


def ensure_udomain_matches_field(
    *,
    field: Field,
    udomain: _domain_types.UniformDomain,
    domain_name: str = "<udomain>",
    field_name: str = "<field>",
) -> None:
    """Ensure UniformDomain matches Field."""
    _domain_types.ensure_udomain(
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
