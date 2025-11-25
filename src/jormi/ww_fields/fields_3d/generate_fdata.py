## { MODULE
##
## === DEPENDENCIES
##

from jormi.ww_fields.fields_3d import (
    fdata_types,
    generate_farrays,
)


##
## === FUNCTIONS
##


def generate_gaussian_random_3d_sdata(
    *,
    resolution: tuple[int, int, int],
    correlation_length: float,
) -> fdata_types.ScalarFieldData_3D:
    """Generate ScalarFieldData_3D with a Gaussian correlation length."""
    sarray_3d_random = generate_farrays.generate_gaussian_random_3d_sarray(
        resolution=resolution,
        correlation_length=correlation_length,
    )
    return fdata_types.ScalarFieldData_3D(
        farray=sarray_3d_random,
        param_name="<gaussian_random_3d_sdata>",
    )


def generate_powerlaw_random_3d_sdata(
    *,
    resolution: tuple[int, int, int],
    alpha_perp: float,
    alpha_para: float | None = None,
) -> fdata_types.ScalarFieldData_3D:
    """Generate ScalarFieldData_3D with a power-law power spectrum."""
    sarray_3d_random = generate_farrays.generate_powerlaw_random_3d_sarray(
        resolution=resolution,
        alpha_perp=alpha_perp,
        alpha_para=alpha_para,
    )
    return fdata_types.ScalarFieldData_3D(
        farray=sarray_3d_random,
        param_name="<powerlaw_random_3d_sdata>",
    )


## } MODULE
