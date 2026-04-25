## { MODULE

##
## === DEPENDENCIES
##

## local
from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_2d import (
    domain_types as _2d_domain_type,
    field_types as _2d_field_type,
)
from jormi.ww_fields.fields_3d import (
    domain_types as _3d_domain_type,
    field_types as _3d_field_type,
)
from jormi.ww_checks import check_python_types

##
## === SLICE HELPERS
##


def _slice_3d_udomain(
    *,
    udomain_3d: _3d_domain_type.UniformDomain_3D,
    out_of_plane_axis: cartesian_axes.AxisLike_3D,
    slice_index: int,
    param_name: str = "<udomain_3d>",
) -> _2d_domain_type.UniformDomain_2D_Sliced3D:
    """Construct a 2D sliced domain from a 3D uniform domain."""
    out_of_plane_axis = cartesian_axes.as_axis(
        axis=out_of_plane_axis,
        param_name=f"{param_name}.out_of_plane_axis",
    )
    out_of_plane_axis_index = out_of_plane_axis.axis_index
    check_python_types.ensure_finite_int(
        param=slice_index,
        param_name=f"{param_name}.slice_index",
        allow_none=False,
        require_positive=False,
    )
    if slice_index < 0:
        raise ValueError(
            f"`{param_name}.slice_index` must be non-negative:"
            f" got {slice_index}",
        )
    periodicity_3d = udomain_3d.periodicity
    resolution_3d = udomain_3d.resolution
    domain_bounds_3d = udomain_3d.domain_bounds
    if slice_index >= resolution_3d[out_of_plane_axis_index]:
        raise ValueError(
            f"`{param_name}.slice_index` = {slice_index} must be smaller than"
            f" resolution[{out_of_plane_axis_index}] = {resolution_3d[out_of_plane_axis_index]}.",
        )
    in_plane_axes = [axis for axis in cartesian_axes.DEFAULT_3D_AXES_ORDER if axis is not out_of_plane_axis]
    x0_in_plane_axis, x1_in_plane_axis = in_plane_axes
    x0_in_plane_axis_index = x0_in_plane_axis.axis_index
    x1_in_plane_axis_index = x1_in_plane_axis.axis_index
    periodicity_2d = (
        periodicity_3d[x0_in_plane_axis_index],
        periodicity_3d[x1_in_plane_axis_index],
    )
    resolution_2d = (
        resolution_3d[x0_in_plane_axis_index],
        resolution_3d[x1_in_plane_axis_index],
    )
    domain_bounds_2d = (
        domain_bounds_3d[x0_in_plane_axis_index],
        domain_bounds_3d[x1_in_plane_axis_index],
    )
    cell_centers_3d = udomain_3d.cell_centers
    out_of_plane_cell_centers = cell_centers_3d[out_of_plane_axis_index]
    slice_position = float(out_of_plane_cell_centers[slice_index])
    return _2d_domain_type.UniformDomain_2D_Sliced3D.from_slice(
        periodicity=periodicity_2d,
        resolution=resolution_2d,
        domain_bounds=domain_bounds_2d,
        out_of_plane_axis=out_of_plane_axis,
        slice_index=slice_index,
        slice_position=slice_position,
    )


##
## === SLICE 3D SCALAR FIELD
##


def slice_3d_sfield(
    *,
    sfield_3d: _3d_field_type.ScalarField_3D,
    out_of_plane_axis: cartesian_axes.AxisLike_3D,
    slice_index: int,
    field_label: str | None = None,
) -> _2d_field_type.ScalarField_2D:
    """
    Slice a 3D scalar field into a 2D scalar field.

    The 2D domain is a UniformDomain_2D_Sliced3D carrying metadata
    about the out-of-plane axis and slice position.
    """
    _3d_field_type.ensure_3d_sfield(sfield_3d)
    sarray_3d = _3d_field_type.extract_3d_sarray(sfield_3d)
    udomain_3d = sfield_3d.udomain
    sim_time = sfield_3d.sim_time
    slice_param_name = "<sfield_3d_slice>"
    udomain_2d = _slice_3d_udomain(
        udomain_3d=udomain_3d,
        out_of_plane_axis=out_of_plane_axis,
        slice_index=slice_index,
        param_name=slice_param_name,
    )
    out_of_plane_axis = cartesian_axes.as_axis(
        axis=out_of_plane_axis,
        param_name=f"{slice_param_name}.out_of_plane_axis",
    )
    out_of_plane_axis_index = out_of_plane_axis.axis_index
    if out_of_plane_axis_index == 0:
        sarray_2d = sarray_3d[slice_index, :, :]
    elif out_of_plane_axis_index == 1:
        sarray_2d = sarray_3d[:, slice_index, :]
    else:
        sarray_2d = sarray_3d[:, :, slice_index]
    if field_label is None:
        field_label = sfield_3d.field_label
    return _2d_field_type.ScalarField_2D.from_2d_sarray(
        sarray_2d=sarray_2d,
        udomain_2d=udomain_2d,
        field_label=field_label,
        sim_time=sim_time,
    )


##
## === SLICE 3D VECTOR FIELD
##


def slice_3d_vfield_inplane(
    *,
    vfield_3d: _3d_field_type.VectorField_3D,
    out_of_plane_axis: cartesian_axes.AxisLike_3D,
    slice_index: int,
    field_label: str | None = None,
) -> _2d_field_type.VectorField_2D:
    """
    Slice a 3D vector field into a 2D vector field of in-plane components.

    The resulting 2D vector has two components corresponding to the
    components tangent to the slice plane.
    """
    _3d_field_type.ensure_3d_vfield(vfield_3d)
    varray_3d = _3d_field_type.extract_3d_varray(vfield_3d)
    udomain_3d = vfield_3d.udomain
    sim_time = vfield_3d.sim_time
    slice_param_name = "<vfield_3d_slice_inplane>"
    udomain_2d = _slice_3d_udomain(
        udomain_3d=udomain_3d,
        out_of_plane_axis=out_of_plane_axis,
        slice_index=slice_index,
        param_name=slice_param_name,
    )
    out_of_plane_axis = cartesian_axes.as_axis(
        axis=out_of_plane_axis,
        param_name=f"{slice_param_name}.out_of_plane_axis",
    )
    out_of_plane_axis_index = out_of_plane_axis.axis_index
    comp_indices = list(cartesian_axes.VALID_3D_AXIS_INDICES)
    comp_indices.remove(out_of_plane_axis_index)
    if out_of_plane_axis_index == 0:
        varray_2d = varray_3d[comp_indices, slice_index, :, :]
    elif out_of_plane_axis_index == 1:
        varray_2d = varray_3d[comp_indices, :, slice_index, :]
    else:
        varray_2d = varray_3d[comp_indices, :, :, slice_index]
    if field_label is None:
        field_label = vfield_3d.field_label
    return _2d_field_type.VectorField_2D.from_2d_varray(
        varray_2d=varray_2d,
        udomain_2d=udomain_2d,
        field_label=field_label,
        sim_time=sim_time,
    )


def slice_3d_vfield_outofplane(
    *,
    vfield_3d: _3d_field_type.VectorField_3D,
    out_of_plane_axis: cartesian_axes.AxisLike_3D,
    slice_index: int,
    field_label: str | None = None,
) -> _2d_field_type.ScalarField_2D:
    """Slice a 3D vector field into a 2D scalar field of the out-of-plane component."""
    _3d_field_type.ensure_3d_vfield(vfield_3d)
    varray_3d = _3d_field_type.extract_3d_varray(vfield_3d)
    udomain_3d = vfield_3d.udomain
    sim_time = vfield_3d.sim_time
    slice_param_name = "<vfield_3d_slice_outofplane>"
    udomain_2d = _slice_3d_udomain(
        udomain_3d=udomain_3d,
        out_of_plane_axis=out_of_plane_axis,
        slice_index=slice_index,
        param_name=slice_param_name,
    )
    out_of_plane_axis = cartesian_axes.as_axis(
        axis=out_of_plane_axis,
        param_name=f"{slice_param_name}.out_of_plane_axis",
    )
    out_of_plane_axis_index = out_of_plane_axis.axis_index
    if out_of_plane_axis_index == 0:
        sarray_2d = varray_3d[0, slice_index, :, :]
    elif out_of_plane_axis_index == 1:
        sarray_2d = varray_3d[1, :, slice_index, :]
    else:
        sarray_2d = varray_3d[2, :, :, slice_index]
    if field_label is None:
        field_label = vfield_3d.field_label
    return _2d_field_type.ScalarField_2D.from_2d_sarray(
        sarray_2d=sarray_2d,
        udomain_2d=udomain_2d,
        field_label=field_label,
        sim_time=sim_time,
    )


## } MODULE
