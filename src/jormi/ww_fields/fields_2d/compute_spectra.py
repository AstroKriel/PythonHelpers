## { MODULE

##
## === DEPENDENCIES
##

## local
from jormi.ww_arrays.farrays_2d.compute_spectra import IsotropicPowerSpectrum as IsotropicPowerSpectrum
from jormi.ww_arrays.farrays_2d import compute_spectra as _compute_spectra
from jormi.ww_fields.fields_2d import field_models
from jormi.ww_validation import validate_types

##
## === TYPE ALIASES
##

## every concrete 2D field type this function accepts; extend this tuple (and nothing
## else) when a new rank is added to field_models.AnyField_2D
_SUPPORTED_2D_FIELD_TYPES = (
    field_models.ScalarField_2D,
    field_models.VectorField_2D,
)

##
## === PUBLIC FUNCTIONS
##


def compute_isotropic_power_spectrum_field(
    field_2d: field_models.AnyField_2D,
) -> IsotropicPowerSpectrum:
    """Compute the 1D (shell-integrated) power spectrum of a 2D field of any rank: scalar, vector, ..."""
    validate_types.ensure_type(
        param=field_2d,
        param_name="<field_2d>",
        valid_types=_SUPPORTED_2D_FIELD_TYPES,
    )
    if isinstance(field_2d, field_models.ScalarField_2D):
        farray_2d = field_models.extract_2d_sarray(sfield_2d=field_2d)
    else:
        farray_2d = field_models.extract_2d_varray(vfield_2d=field_2d)
    return _compute_spectra.compute_isotropic_power_spectrum_farray(
        farray_2d=farray_2d,
        resolution_2d=field_2d.uniform_domain.resolution,
    )


## } MODULE
