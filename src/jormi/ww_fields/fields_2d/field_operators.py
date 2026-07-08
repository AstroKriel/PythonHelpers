## { MODULE

##
## === DEPENDENCIES
##

## third-party
import numpy

## local
from jormi.ww_fields.fields_2d import field_models

##
## === OPERATORS WORKING ON SLICED VECTOR FIELDS
##


def compute_sliced_vfield_magnitude(
    sliced_vfield_2d: field_models.SlicedVectorFields_2D,
    *,
    field_name: str,
    latex_label: str,
) -> field_models.ScalarField_2D:
    """Compute the magnitude sqrt(f_i f_i) of a `SlicedVectorFields_2D`, in-plane and out-of-plane combined."""
    inplane_varray_2d = field_models.extract_2d_varray(
        vfield_2d=sliced_vfield_2d.inplane_vfield_2d,
        param_name="<sliced_vfield_2d.inplane_vfield_2d>",
    )
    outofplane_sarray_2d = field_models.extract_2d_sarray(
        sfield_2d=sliced_vfield_2d.outofplane_sfield_2d,
        param_name="<sliced_vfield_2d.outofplane_sfield_2d>",
    )
    magnitude_sq_sarray_2d = numpy.sum(inplane_varray_2d**2, axis=0) + outofplane_sarray_2d**2
    magnitude_sarray_2d = numpy.sqrt(magnitude_sq_sarray_2d)
    return field_models.ScalarField_2D.from_2d_sarray(
        sarray_2d=magnitude_sarray_2d,
        uniform_domain_2d=sliced_vfield_2d.inplane_vfield_2d.uniform_domain,
        field_name=field_name,
        latex_label=latex_label,
        sim_time=sliced_vfield_2d.inplane_vfield_2d.sim_time,
    )


## } MODULE
