## { MODULE

##
## === DEPENDENCIES
##

## local
from jormi.ww_arrays.farrays_3d.compute_spectra import IsotropicPowerSpectrum as IsotropicPowerSpectrum
from jormi.ww_arrays.farrays_3d import compute_spectra as _compute_spectra
from jormi.ww_fields.fields_3d import field_types

##
## === PUBLIC FUNCTIONS
##


def compute_isotropic_power_spectrum_sfield(
    sfield_3d: field_types.ScalarField_3D,
) -> IsotropicPowerSpectrum:
    """Compute the 1D (shell-integrated) power spectrum of a 3D scalar field."""
    sarray_3d = field_types.extract_3d_sarray(sfield_3d=sfield_3d)
    udomain_3d = sfield_3d.udomain
    resolution_3d = udomain_3d.resolution
    return _compute_spectra.compute_isotropic_power_spectrum_sarray(
        sarray_3d=sarray_3d,
        resolution_3d=resolution_3d,
    )


## } MODULE
