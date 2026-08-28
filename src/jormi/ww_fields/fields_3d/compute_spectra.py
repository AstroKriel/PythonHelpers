## { MODULE

##
## === DEPENDENCIES
##

## local
from jormi.ww_arrays.farrays_3d.compute_spectra import IsotropicPowerSpectrum as IsotropicPowerSpectrum
from jormi.ww_arrays.farrays_3d import compute_spectra as _compute_spectra
from jormi.ww_fields.fields_3d import field_models
from jormi.ww_validation import validate_types

##
## === TYPE ALIASES
##

## every concrete 3D field type this function accepts; extend this tuple (and nothing
## else) when a new rank is added to field_models.AnyField_3D
_SUPPORTED_3D_FIELD_TYPES = (
    field_models.ScalarField_3D,
    field_models.VectorField_3D,
    field_models.RankTwoTensorField_3D,
)

##
## === PUBLIC FUNCTIONS
##


def compute_isotropic_power_spectrum_field(
    field_3d: field_models.AnyField_3D,
) -> IsotropicPowerSpectrum:
    """Compute the 1D (shell-integrated) power spectrum of a 3D field of any rank: scalar, vector, rank-2 tensor, ..."""
    validate_types.ensure_type(
        param=field_3d,
        param_name="<field_3d>",
        valid_types=_SUPPORTED_3D_FIELD_TYPES,
    )
    return _compute_spectra.compute_isotropic_power_spectrum_farray(
        farray_3d=field_3d.fdata.farray,
        resolution_3d=field_3d.uniform_domain.resolution,
        num_ranks=field_3d.fdata.num_ranks,
    )


## } MODULE
