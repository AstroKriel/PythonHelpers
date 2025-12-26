## { MODULE

##
## === DEPENDENCIES
##

from jormi.ww_types import type_checks, sequence_positions
from jormi.ww_fields import _cartesian_coordinates
from jormi.ww_fields.fields_2d import (
    domain_type as _2d_domain_type,
    field_type as _2d_field_type,
)
from jormi.ww_fields.fields_3d import (
    domain_type as _3d_domain_type,
    field_type as _3d_field_type,
)

##
## === SLICE HELPERS
##


def _slice_3d_udomain(
    *,
    udomain_3d: _3d_domain_type.UniformDomain_3D,
    out_of_plane_axis: _cartesian_coordinates.AxisLike,
    slice_index: int,
    param_name: str = "<udomain_3d>",
) -> _2d_domain_type.UniformDomain_2D_Sliced3D:
    """Construct a 2D sliced domain from a 3D uniform domain."""
    ## normalise out-of-plane axis
    out_of_plane_axis = _cartesian_coordinates.as_axis(
        axis=out_of_plane_axis,
        param_name=f"{param_name}.out_of_plane_axis",
    )
    out_of_plane_axis_index = out_of_plane_axis.axis_index
    ## validate slice index
    type_checks.ensure_finite_int(
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
    ## extract 3D metadata
    periodicity_3d = udomain_3d.periodicity
    resolution_3d = udomain_3d.resolution
    domain_bounds_3d = udomain_3d.domain_bounds
    if slice_index >= resolution_3d[out_of_plane_axis_index]:
        raise ValueError(
            f"`{param_name}.slice_index` = {slice_index} must be smaller than"
            f" resolution[{out_of_plane_axis_index}] = {resolution_3d[out_of_plane_axis_index]}.",
        )
    ## determine in-plane axes by dropping the out-of-plane axis
    in_plane_axes = [
        axis for axis in _cartesian_coordinates.DEFAULT_AXES_ORDER if axis is not out_of_plane_axis
    ]
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
    ## compute physical coordinate of the slice along the out-of-plane axis
    cell_centers_3d = udomain_3d.cell_centers
    out_of_plane_cell_centers = cell_centers_3d[out_of_plane_axis_index]
    slice_position = float(out_of_plane_cell_centers[slice_index])
    ## construct 2D domain that remembers its 3D origin
    return _2d_domain_type.UniformDomain_2D_Sliced3D(
        periodicity=periodicity_2d,
        resolution=resolution_2d,
        domain_bounds=domain_bounds_2d,
        out_of_plane_axis=out_of_plane_axis,
        slice_index=slice_index,
        slice_position=slice_position,
    )


def get_slice_index(
    *,
    udomain_3d: _3d_domain_type.UniformDomain_3D,
    slice_axis: _cartesian_coordinates.AxisLike,
    position: sequence_positions.SequencePositionLike,
    param_name: str = "<udomain_3d>",
) -> int:
    """
    Return a canonical slice index along a given axis.

    `position` can be:
    - SequencePositionLike: "first" / "middle" / "last"
    """
    slice_axis = _cartesian_coordinates.as_axis(
        axis=slice_axis,
        param_name=f"{param_name}.slice_axis",
    )
    slice_axis_index = slice_axis.axis_index
    resolution_3d = udomain_3d.resolution
    axis_resolution = resolution_3d[slice_axis_index]
    last_index = axis_resolution - 1
    seq_position = sequence_positions.as_sequence_position(position)
    if seq_position is sequence_positions.SequencePosition.First: return 0
    if seq_position is sequence_positions.SequencePosition.Middle: return last_index // 2
    if seq_position is sequence_positions.SequencePosition.Last: return last_index
    raise RuntimeError(
        f"Unexpected SequencePosition {seq_position!r} for {param_name}.slice_axis.",
    )


##
## === SLICE 3D SCALAR FIELD
##


def slice_3d_sfield(
    *,
    sfield_3d: _3d_field_type.ScalarField_3D,
    out_of_plane_axis: _cartesian_coordinates.AxisLike,
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
    axis_enum = _cartesian_coordinates.as_axis(
        axis=out_of_plane_axis,
        param_name=f"{slice_param_name}.out_of_plane_axis",
    )
    axis_index = axis_enum.axis_index
    if axis_index == 0:
        sarray_2d = sarray_3d[slice_index, :, :]
    elif axis_index == 1:
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
    out_of_plane_axis: _cartesian_coordinates.AxisLike,
    slice_index: int,
    field_label: str | None = None,
) -> _2d_field_type.VectorField_2D:
    """
    Slice a 3D vector field into a 2D vector field of in-plane components.

    The resulting 2D vector has two components corresponding to the
    components tangent to the slice plane (i.e. all components except
    along `out_of_plane_axis`).
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
    axis_enum = _cartesian_coordinates.as_axis(
        axis=out_of_plane_axis,
        param_name=f"{slice_param_name}.out_of_plane_axis",
    )
    axis_index = axis_enum.axis_index
    ## in-plane components: drop the component aligned with out_of_plane_axis
    comp_indices = [0, 1, 2]
    comp_indices.remove(axis_index)
    if axis_index == 0:
        varray_2d = varray_3d[comp_indices, slice_index, :, :]
    elif axis_index == 1:
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
    out_of_plane_axis: _cartesian_coordinates.AxisLike,
    slice_index: int,
    field_label: str | None = None,
) -> _2d_field_type.ScalarField_2D:
    """
    Slice a 3D vector field into a 2D scalar field of the out-of-plane component.

    The resulting 2D scalar field contains the component of the vector
    aligned with `out_of_plane_axis`, evaluated on the slice plane.
    """
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
    axis_enum = _cartesian_coordinates.as_axis(
        axis=out_of_plane_axis,
        param_name=f"{slice_param_name}.out_of_plane_axis",
    )
    axis_index = axis_enum.axis_index
    if axis_index == 0:
        sarray_2d = varray_3d[0, slice_index, :, :]
    elif axis_index == 1:
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
