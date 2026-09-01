## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import dataclasses

## local
from jormi.ww_arrays.farrays_3d import filter_farrays as _filter_farrays
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


def compute_bandpass_filtered_field(
    field_3d: field_models.AnyField_3D,
    *,
    k_min: float,
    k_max: float,
    field_name: str | None = None,
    latex_label: str | None = None,
) -> field_models.AnyField_3D:
    """
    Band-pass filter a 3D field of any rank in Fourier space: scalar, vector, rank-2
    tensor, ..., retaining modes with k_min <= |k_index| <= k_max.

    Returns a field of the same concrete type on the same domain; `field_name` and
    `latex_label` default to the input field's own, so the caller can rename the
    filtered result or leave it as-is.
    """
    validate_types.ensure_type(
        param=field_3d,
        param_name="<field_3d>",
        valid_types=_SUPPORTED_3D_FIELD_TYPES,
    )
    filtered_farray_3d = _filter_farrays.compute_bandpass_filtered_farray(
        farray_3d=field_3d.fdata.farray,
        resolution_3d=field_3d.uniform_domain.resolution,
        num_ranks=field_3d.fdata.num_ranks,
        k_min=k_min,
        k_max=k_max,
    )
    filtered_fdata = type(field_3d.fdata)(farray=filtered_farray_3d)
    return dataclasses.replace(
        field_3d,
        fdata=filtered_fdata,
        field_name=field_3d.field_name if (field_name is None) else field_name,
        latex_label=field_3d.latex_label if (latex_label is None) else latex_label,
    )


## } MODULE
