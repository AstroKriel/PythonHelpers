## { MODULE

##
## === DEPENDENCIES
##

## local
from jormi.ww_arrays.farrays_3d import generate_farrays
from jormi.ww_fields.fields_3d import (
    domain_types,
    field_types,
)
from jormi.ww_validation import validate_types

##
## === GAUSSIAN RANDOM FIELDS
##


def generate_gaussian_random_3d_sfield(
    *,
    udomain_3d: domain_types.UniformDomain_3D,
    correlation_length: float,
    field_label: str = "G(x)",
    sim_time: float | None = None,
) -> field_types.ScalarField_3D:
    """Generate a 3D scalar field with a Gaussian correlation length."""
    domain_types.ensure_3d_udomain(
        udomain_3d=udomain_3d,
        param_name="<udomain_3d>",
    )
    validate_types.ensure_finite_float(
        param=correlation_length,
        param_name="<correlation_length>",
        allow_none=False,
        require_positive=True,
    )
    sarray_3d = generate_farrays.generate_gaussian_random_3d_sarray(
        resolution=udomain_3d.resolution,
        correlation_length=correlation_length,
    )
    return field_types.ScalarField_3D.from_3d_sarray(
        sarray_3d=sarray_3d,
        udomain_3d=udomain_3d,
        field_label=field_label,
        sim_time=sim_time,
    )


##
## === POWER-LAW RANDOM FIELDS
##


def generate_powerlaw_random_3d_sfield(
    *,
    udomain_3d: domain_types.UniformDomain_3D,
    alpha_perp: float,
    alpha_para: float | None = None,
    field_label: str = "P(x)",
    sim_time: float | None = None,
) -> field_types.ScalarField_3D:
    """Generate a 3D scalar field with a power-law power spectrum."""
    domain_types.ensure_3d_udomain(
        udomain_3d=udomain_3d,
        param_name="<udomain_3d>",
    )
    validate_types.ensure_finite_float(
        param=alpha_perp,
        param_name="<alpha_perp>",
        allow_none=False,
        require_positive=False,
    )
    if alpha_para is not None:
        validate_types.ensure_finite_float(
            param=alpha_para,
            param_name="<alpha_para>",
            allow_none=False,
            require_positive=False,
        )
    sarray_3d = generate_farrays.generate_powerlaw_random_3d_sarray(
        resolution=udomain_3d.resolution,
        alpha_perp=alpha_perp,
        alpha_para=alpha_para,
    )
    return field_types.ScalarField_3D.from_3d_sarray(
        sarray_3d=sarray_3d,
        udomain_3d=udomain_3d,
        field_label=field_label,
        sim_time=sim_time,
    )


## } MODULE
