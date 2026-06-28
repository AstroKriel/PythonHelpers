## { MODULE

##
## === DEPENDENCIES
##

## third-party
from typing import Any

import numpy
from numpy.typing import NDArray

## local
from jormi.ww_arrays.farrays_3d import (
    difference_sarrays,
    farray_types,
)
from jormi.ww_validation import validate_arrays, validate_types

##
## === WORKSPACE UTILITIES
##


def _as_float_view(
    farray: NDArray[Any],
) -> NDArray[Any]:
    """Promote from integers/low-precision to float64 for safe reductions."""
    validate_arrays.ensure_array(
        array=farray,
        param_name="<farray>",
    )
    dtype = numpy.result_type(farray.dtype, numpy.float64)
    return farray.astype(dtype, copy=False)


##
## === SCALAR (3D) ARRAY OPERATORS
##


def compute_sarray_rms(
    sarray_3d: NDArray[Any],
) -> float:
    """Compute the RMS of a 3D scalar array."""
    farray_types.ensure_3d_sarray(
        sarray_3d=sarray_3d,
        param_name="<sarray_3d>",
    )
    float_sarray_3d = _as_float_view(sarray_3d)
    return float(
        numpy.sqrt(
            numpy.mean(
                numpy.square(
                    float_sarray_3d,
                ),
            ),
        ),
    )


def compute_sarray_volume_integral(
    sarray_3d: NDArray[Any],
    *,
    cell_volume: float,
) -> float:
    """Compute the volume integral of a 3D scalar array."""
    farray_types.ensure_3d_sarray(
        sarray_3d=sarray_3d,
        param_name="<sarray_3d>",
    )
    validate_types.ensure_finite_float(
        param=cell_volume,
        param_name="<cell_volume>",
        allow_none=False,
        require_positive=True,
    )
    float_sarray_3d = _as_float_view(sarray_3d)
    return float(cell_volume * numpy.sum(float_sarray_3d))


def compute_sarray_grad(
    sarray_3d: NDArray[Any],
    *,
    cell_widths_3d: tuple[float, float, float],
    out_varray_3d: NDArray[Any] | None = None,
    grad_order: int = 2,
) -> NDArray[Any]:
    """
    Compute the gradient of a 3D scalar array.

    Returns a 4D ndarray with shape (3, num_x0_cells, num_x1_cells, num_x2_cells).
    """
    farray_types.ensure_3d_sarray(
        sarray_3d=sarray_3d,
        param_name="<sarray_3d>",
    )
    farray_types.ensure_3d_cell_widths(cell_widths_3d)
    validate_types.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    nabla = difference_sarrays.get_grad_fn(grad_order)
    num_cells_x, num_cells_y, num_cells_z = sarray_3d.shape
    dtype = numpy.result_type(sarray_3d.dtype, numpy.float64)
    grad_f_varray_3d = farray_types.ensure_farray_metadata(
        farray_shape=(3, num_cells_x, num_cells_y, num_cells_z),
        farray=out_varray_3d,
        dtype=dtype,
    )
    cell_width_x, cell_width_y, cell_width_z = cell_widths_3d
    ## fill d_i f vector: (gradient-dir-i, x0, x1, x2)
    grad_f_varray_3d[0, ...] = nabla(
        sarray_3d=sarray_3d,
        cell_width=cell_width_x,
        grad_axis=0,
    )
    grad_f_varray_3d[1, ...] = nabla(
        sarray_3d=sarray_3d,
        cell_width=cell_width_y,
        grad_axis=1,
    )
    grad_f_varray_3d[2, ...] = nabla(
        sarray_3d=sarray_3d,
        cell_width=cell_width_z,
        grad_axis=2,
    )
    return grad_f_varray_3d


def scale_sarray_inplace(
    sarray_3d: NDArray[Any],
    *,
    scale: float,
) -> None:
    """Scale a 3D scalar array in-place by a scalar factor."""
    farray_types.ensure_3d_sarray(
        sarray_3d=sarray_3d,
        param_name="<sarray_3d>",
    )
    sarray_3d *= numpy.asarray(scale, dtype=sarray_3d.dtype)


def sqrt_sarray_inplace(
    sarray_3d: NDArray[Any],
) -> None:
    """Take the square-root of a 3D scalar array in-place."""
    farray_types.ensure_3d_sarray(
        sarray_3d=sarray_3d,
        param_name="<sarray_3d>",
    )
    numpy.sqrt(sarray_3d, out=sarray_3d)


##
## === VECTOR (3D) ARRAY OPERATORS
##


def compute_sum_of_varray_comps_squared(
    varray_3d: NDArray[Any],
    *,
    out_sarray_3d: NDArray[Any] | None = None,
    tmp_sarray_3d: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """
    Compute sum_i (v_i v_i) per cell for a 3D vector field.

    varray_3d has shape (3, num_x0_cells, num_x1_cells, num_x2_cells).
    Returns a 3D ndarray with shape (num_x0_cells, num_x1_cells, num_x2_cells).
    """
    farray_types.ensure_3d_varray(
        varray_3d=varray_3d,
        param_name="<varray_3d>",
    )
    domain_shape = varray_3d.shape[1:]
    dtype = numpy.result_type(varray_3d.dtype, numpy.float64)
    out_sarray_3d = farray_types.ensure_farray_metadata(
        farray_shape=domain_shape,
        farray=out_sarray_3d,
        dtype=dtype,
    )
    tmp_sarray_3d = farray_types.ensure_farray_metadata(
        farray_shape=domain_shape,
        farray=tmp_sarray_3d,
        dtype=dtype,
    )
    ## out = v_x^2
    numpy.multiply(
        varray_3d[0],
        varray_3d[0],
        out=out_sarray_3d,
    )
    ## tmp = v_y^2
    numpy.multiply(
        varray_3d[1],
        varray_3d[1],
        out=tmp_sarray_3d,
    )
    ## out = v_x^2 + v_y^2
    numpy.add(
        out_sarray_3d,
        tmp_sarray_3d,
        out=out_sarray_3d,
    )
    ## tmp = v_z^2
    numpy.multiply(
        varray_3d[2],
        varray_3d[2],
        out=tmp_sarray_3d,
    )
    ## out = v_x^2 + v_y^2 + v_z^2
    numpy.add(
        out_sarray_3d,
        tmp_sarray_3d,
        out=out_sarray_3d,
    )
    return out_sarray_3d


def compute_dot_over_varray_comps(
    *,
    f_varray_3d: NDArray[Any],
    g_varray_3d: NDArray[Any],
    out_sarray_3d: NDArray[Any] | None = None,
    tmp_sarray_3d: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """
    Compute vec(a) dot vec(b) per cell for 3D vector fields.

    Each input has shape (3, num_x0_cells, num_x1_cells, num_x2_cells).
    Returns a 3D ndarray with shape (num_x0_cells, num_x1_cells, num_x2_cells).
    """
    farray_types.ensure_3d_varray(
        varray_3d=f_varray_3d,
        param_name="<f_varray_3d>",
    )
    farray_types.ensure_3d_varray(
        varray_3d=g_varray_3d,
        param_name="<g_varray_3d>",
    )
    validate_arrays.ensure_same_shape(
        array_a=f_varray_3d,
        array_b=g_varray_3d,
        param_name_a="<f_varray_3d>",
        param_name_b="<g_varray_3d>",
    )
    domain_shape = f_varray_3d.shape[1:]
    dtype = numpy.result_type(f_varray_3d.dtype, g_varray_3d.dtype)
    out_sarray_3d = farray_types.ensure_farray_metadata(
        farray_shape=domain_shape,
        farray=out_sarray_3d,
        dtype=dtype,
    )
    tmp_sarray_3d = farray_types.ensure_farray_metadata(
        farray_shape=domain_shape,
        farray=tmp_sarray_3d,
        dtype=dtype,
    )
    ## out = a_x b_x
    numpy.multiply(
        f_varray_3d[0],
        g_varray_3d[0],
        out=out_sarray_3d,
    )
    ## tmp = a_y b_y
    numpy.multiply(
        f_varray_3d[1],
        g_varray_3d[1],
        out=tmp_sarray_3d,
    )
    ## out += a_y b_y
    numpy.add(
        out_sarray_3d,
        tmp_sarray_3d,
        out=out_sarray_3d,
    )
    ## tmp = a_z b_z
    numpy.multiply(
        f_varray_3d[2],
        g_varray_3d[2],
        out=tmp_sarray_3d,
    )
    ## out += a_z b_z
    numpy.add(
        out_sarray_3d,
        tmp_sarray_3d,
        out=out_sarray_3d,
    )
    return out_sarray_3d


def compute_varray_directional_derivative(
    *,
    target_varray_3d: NDArray[Any],
    along_varray_3d: NDArray[Any],
    cell_widths_3d: tuple[float, float, float],
    out_varray_3d: NDArray[Any] | None = None,
    grad_order: int = 2,
) -> NDArray[Any]:
    """
    Compute the directional derivative `(direction_i d_i target_j)` for 3D vector fields.

    All vector arrays have shape `(3, num_x0_cells, num_x1_cells, num_x2_cells)`.
    """
    farray_types.ensure_3d_varray(
        varray_3d=target_varray_3d,
        param_name="<target_varray_3d>",
    )
    farray_types.ensure_3d_varray(
        varray_3d=along_varray_3d,
        param_name="<along_varray_3d>",
    )
    validate_arrays.ensure_same_shape(
        array_b=target_varray_3d,
        array_a=along_varray_3d,
        param_name_b="<target_varray_3d>",
        param_name_a="<along_varray_3d>",
    )
    farray_types.ensure_3d_cell_widths(cell_widths_3d)
    validate_types.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    nabla = difference_sarrays.get_grad_fn(grad_order)
    domain_shape = along_varray_3d.shape[1:]
    dtype = numpy.result_type(
        target_varray_3d.dtype,
        along_varray_3d.dtype,
        numpy.float64,
    )
    out_varray_3d = farray_types.ensure_farray_metadata(
        farray_shape=(3, *domain_shape),
        farray=out_varray_3d,
        dtype=dtype,
    )
    out_varray_3d.fill(0.0)
    ## accumulate the directional contraction directly so we never materialise the full d_i target_j tensor
    for grad_axis, cell_width in enumerate(cell_widths_3d):
        direction_comp_sarray_3d = along_varray_3d[grad_axis]
        for comp_index in range(3):
            grad_comp_sarray_3d = nabla(
                sarray_3d=target_varray_3d[comp_index],
                cell_width=cell_width,
                grad_axis=grad_axis,
            )
            out_varray_3d[comp_index] += direction_comp_sarray_3d * grad_comp_sarray_3d
    return out_varray_3d


def compute_varray_grad(
    varray_3d: NDArray[Any],
    *,
    cell_widths_3d: tuple[float, float, float],
    grad_f_r2tarray_3d: NDArray[Any] | None = None,
    grad_order: int = 2,
) -> NDArray[Any]:
    """
    Compute gradient of a 3D vector field.

    varray_3d has shape (3, num_x0_cells, num_x1_cells, num_x2_cells).

    Returns:
      - r2tarray_3d with shape (3, 3, num_x0_cells, num_x1_cells, num_x2_cells),
        where grad[comp_j, grad_i, ...] = d_i v_j.
    """
    farray_types.ensure_3d_varray(
        varray_3d=varray_3d,
        param_name="<varray_3d>",
    )
    farray_types.ensure_3d_cell_widths(cell_widths_3d)
    validate_types.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    nabla = difference_sarrays.get_grad_fn(grad_order)
    num_cells_x, num_cells_y, num_cells_z = varray_3d.shape[1:]
    dtype = numpy.result_type(varray_3d.dtype, numpy.float64)
    grad_f_r2tarray_3d = farray_types.ensure_farray_metadata(
        farray_shape=(3, 3, num_cells_x, num_cells_y, num_cells_z),
        farray=grad_f_r2tarray_3d,
        dtype=dtype,
    )
    cell_width_x, cell_width_y, cell_width_z = cell_widths_3d
    ## fill d_i f_j tensor: (component-j, gradient-dir-i, x0, x1, x2)
    for comp_j in range(3):
        grad_f_r2tarray_3d[comp_j, 0, ...] = nabla(
            sarray_3d=varray_3d[comp_j],
            cell_width=cell_width_x,
            grad_axis=0,
        )
        grad_f_r2tarray_3d[comp_j, 1, ...] = nabla(
            sarray_3d=varray_3d[comp_j],
            cell_width=cell_width_y,
            grad_axis=1,
        )
        grad_f_r2tarray_3d[comp_j, 2, ...] = nabla(
            sarray_3d=varray_3d[comp_j],
            cell_width=cell_width_z,
            grad_axis=2,
        )
    return grad_f_r2tarray_3d


def compute_varray_cross_product(
    *,
    f_varray_3d: NDArray[Any],
    g_varray_3d: NDArray[Any],
    out_varray_3d: NDArray[Any] | None = None,
    tmp_sarray_3d: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """
    Compute the cross product epsilon_ijk a_j b_k cellwise for two 3D vector fields.

    All arrays have shape (3, num_x0_cells, num_x1_cells, num_x2_cells).
    """
    farray_types.ensure_3d_varray(
        varray_3d=f_varray_3d,
        param_name="<f_varray_3d>",
    )
    farray_types.ensure_3d_varray(
        varray_3d=g_varray_3d,
        param_name="<g_varray_3d>",
    )
    validate_arrays.ensure_same_shape(
        array_a=f_varray_3d,
        array_b=g_varray_3d,
        param_name_a="<f_varray_3d>",
        param_name_b="<g_varray_3d>",
    )
    domain_shape = f_varray_3d.shape[1:]
    dtype = numpy.result_type(f_varray_3d.dtype, g_varray_3d.dtype)
    axb_varray_3d = farray_types.ensure_farray_metadata(
        farray_shape=f_varray_3d.shape,
        dtype=dtype,
        farray=out_varray_3d,
    )
    tmp_sarray_3d = farray_types.ensure_farray_metadata(
        farray_shape=domain_shape,
        dtype=dtype,
        farray=tmp_sarray_3d,
    )
    ## cross_x = a_y * b_z - a_z * b_y
    numpy.multiply(
        f_varray_3d[1],
        g_varray_3d[2],
        out=axb_varray_3d[0],
    )
    numpy.multiply(
        f_varray_3d[2],
        g_varray_3d[1],
        out=tmp_sarray_3d,
    )
    numpy.subtract(
        axb_varray_3d[0],
        tmp_sarray_3d,
        out=axb_varray_3d[0],
    )
    ## cross_y = -a_x * b_z + a_z * b_x
    numpy.multiply(
        f_varray_3d[2],
        g_varray_3d[0],
        out=axb_varray_3d[1],
    )
    numpy.multiply(
        f_varray_3d[0],
        g_varray_3d[2],
        out=tmp_sarray_3d,
    )
    numpy.subtract(
        axb_varray_3d[1],
        tmp_sarray_3d,
        out=axb_varray_3d[1],
    )
    ## cross_z = a_x * b_y - a_y * b_x
    numpy.multiply(
        f_varray_3d[0],
        g_varray_3d[1],
        out=axb_varray_3d[2],
    )
    numpy.multiply(
        f_varray_3d[1],
        g_varray_3d[0],
        out=tmp_sarray_3d,
    )
    numpy.subtract(
        axb_varray_3d[2],
        tmp_sarray_3d,
        out=axb_varray_3d[2],
    )
    return axb_varray_3d


def compute_varray_curl(
    varray_3d: NDArray[Any],
    *,
    cell_widths_3d: tuple[float, float, float],
    out_varray_3d: NDArray[Any] | None = None,
    grad_order: int = 2,
) -> NDArray[Any]:
    """
    Compute the curl epsilon_ijk d_j f_k of a 3D vector field.

    varray_3d and out_varray_3d have shape (3, num_x0_cells, num_x1_cells, num_x2_cells).
    """
    farray_types.ensure_3d_varray(
        varray_3d=varray_3d,
        param_name="<varray_3d>",
    )
    farray_types.ensure_3d_cell_widths(cell_widths_3d)
    validate_types.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    nabla = difference_sarrays.get_grad_fn(grad_order)
    cell_width_x, cell_width_y, cell_width_z = cell_widths_3d
    curl_varray_3d = farray_types.ensure_farray_metadata(
        farray_shape=varray_3d.shape,
        dtype=varray_3d.dtype,
        farray=out_varray_3d,
    )
    ## curl_x = d_y v_z - d_z v_y
    numpy.subtract(
        nabla(
            sarray_3d=varray_3d[2],
            cell_width=cell_width_y,
            grad_axis=1,
        ),
        nabla(
            sarray_3d=varray_3d[1],
            cell_width=cell_width_z,
            grad_axis=2,
        ),
        out=curl_varray_3d[0],
    )
    ## curl_y = d_z v_x - d_x v_z
    numpy.subtract(
        nabla(
            sarray_3d=varray_3d[0],
            cell_width=cell_width_z,
            grad_axis=2,
        ),
        nabla(
            sarray_3d=varray_3d[2],
            cell_width=cell_width_x,
            grad_axis=0,
        ),
        out=curl_varray_3d[1],
    )
    ## curl_z = d_x v_y - d_y v_x
    numpy.subtract(
        nabla(
            sarray_3d=varray_3d[1],
            cell_width=cell_width_x,
            grad_axis=0,
        ),
        nabla(
            sarray_3d=varray_3d[0],
            cell_width=cell_width_y,
            grad_axis=1,
        ),
        out=curl_varray_3d[2],
    )
    return curl_varray_3d


def compute_varray_divergence(
    varray_3d: NDArray[Any],
    *,
    cell_widths_3d: tuple[float, float, float],
    out_sarray_3d: NDArray[Any] | None = None,
    grad_order: int = 2,
) -> NDArray[Any]:
    """
    Compute the divergence d_i f_i of a 3D vector field.

    varray_3d has shape (3, num_x0_cells, num_x1_cells, num_x2_cells).
    """
    farray_types.ensure_3d_varray(
        varray_3d=varray_3d,
        param_name="<varray_3d>",
    )
    farray_types.ensure_3d_cell_widths(cell_widths_3d)
    validate_types.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    nabla = difference_sarrays.get_grad_fn(grad_order)
    cell_width_x, cell_width_y, cell_width_z = cell_widths_3d
    domain_shape = varray_3d.shape[1:]
    div_sarray_3d = farray_types.ensure_farray_metadata(
        farray_shape=domain_shape,
        dtype=varray_3d.dtype,
        farray=out_sarray_3d,
    )
    ## start with d_i f_i for i=1, then add i=2 and i=3 in-place
    div_sarray_3d[...] = nabla(
        sarray_3d=varray_3d[0],
        cell_width=cell_width_x,
        grad_axis=0,
    )
    numpy.add(
        div_sarray_3d,
        nabla(
            sarray_3d=varray_3d[1],
            cell_width=cell_width_y,
            grad_axis=1,
        ),
        out=div_sarray_3d,
    )
    numpy.add(
        div_sarray_3d,
        nabla(
            sarray_3d=varray_3d[2],
            cell_width=cell_width_z,
            grad_axis=2,
        ),
        out=div_sarray_3d,
    )
    return div_sarray_3d


def compute_varray_magnitude(
    varray_3d: NDArray[Any],
    *,
    out_sarray_3d: NDArray[Any] | None = None,
    tmp_sarray_3d: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """
    Compute |v| = sqrt(v_i v_i) per cell for a 3D vector field.

    Returns a 3D ndarray with shape (num_x0_cells, num_x1_cells, num_x2_cells).
    """
    farray_types.ensure_3d_varray(
        varray_3d=varray_3d,
        param_name="<varray_3d>",
    )
    v_magn_sq_sarray_3d = compute_sum_of_varray_comps_squared(
        varray_3d=varray_3d,
        out_sarray_3d=out_sarray_3d,
        tmp_sarray_3d=tmp_sarray_3d,
    )
    sqrt_sarray_inplace(v_magn_sq_sarray_3d)
    return v_magn_sq_sarray_3d


##
## === RANK-2 TENSOR (3D) ARRAY OPERATORS
##


def compute_r2tarray_divergence(
    r2tarray_3d: NDArray[Any],
    *,
    cell_widths_3d: tuple[float, float, float],
    out_varray_3d: NDArray[Any] | None = None,
    grad_order: int = 2,
) -> NDArray[Any]:
    """
    Compute the divergence (d_j T_ji) of a 3D rank-2 tensor field.

    r2tarray_3d has shape (3, 3, num_x0_cells, num_x1_cells, num_x2_cells),
    where index [j, i, ...] = T_ji.
    Returns a varray_3d with shape (3, num_x0_cells, num_x1_cells, num_x2_cells).
    """
    farray_types.ensure_3d_r2tarray(
        r2tarray_3d=r2tarray_3d,
        param_name="<r2tarray_3d>",
    )
    farray_types.ensure_3d_cell_widths(cell_widths_3d)
    validate_types.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    nabla = difference_sarrays.get_grad_fn(grad_order)
    domain_shape = r2tarray_3d.shape[2:]
    dtype = numpy.result_type(r2tarray_3d.dtype, numpy.float64)
    out_varray_3d = farray_types.ensure_farray_metadata(
        farray_shape=(3, *domain_shape),
        farray=out_varray_3d,
        dtype=dtype,
    )
    cell_width_x, cell_width_y, cell_width_z = cell_widths_3d
    for comp_i in range(3):
        out_varray_3d[comp_i, ...] = nabla(
            sarray_3d=r2tarray_3d[0, comp_i],
            cell_width=cell_width_x,
            grad_axis=0,
        )
        numpy.add(
            out_varray_3d[comp_i, ...],
            nabla(
                sarray_3d=r2tarray_3d[1, comp_i],
                cell_width=cell_width_y,
                grad_axis=1,
            ),
            out=out_varray_3d[comp_i, ...],
        )
        numpy.add(
            out_varray_3d[comp_i, ...],
            nabla(
                sarray_3d=r2tarray_3d[2, comp_i],
                cell_width=cell_width_z,
                grad_axis=2,
            ),
            out=out_varray_3d[comp_i, ...],
        )
    return out_varray_3d


##
## === KINETIC DISSIPATION
##


def compute_varray_kinetic_dissipation(
    v_varray_3d: NDArray[Any],
    *,
    cell_widths_3d: tuple[float, float, float],
    grad_order: int = 2,
) -> NDArray[Any]:
    """
    Compute d_j S_ji for a 3D velocity varray u_j, where

        S_ij = 0.5 * (d_i u_j + d_j u_i) - (1/3) delta_ij (d_k u_k)
    """
    farray_types.ensure_3d_varray(
        varray_3d=v_varray_3d,
        param_name="<v_varray_3d>",
    )
    dtype = v_varray_3d.dtype
    grad_v_r2tarray_3d = compute_varray_grad(
        varray_3d=v_varray_3d,
        cell_widths_3d=cell_widths_3d,
        grad_f_r2tarray_3d=None,
        grad_order=grad_order,
    )
    div_v_sarray_3d = numpy.trace(
        grad_v_r2tarray_3d,
        axis1=0,
        axis2=1,
    )
    sym_r2tarray_3d = 0.5 * grad_v_r2tarray_3d + numpy.transpose(
        grad_v_r2tarray_3d,
        axes=(1, 0, 2, 3, 4),
    )
    del grad_v_r2tarray_3d
    identity_matrix = numpy.eye(3, dtype=dtype)
    bulk_r2tarray_3d = numpy.einsum(
        "ij,xyz->jixyz",
        identity_matrix,
        div_v_sarray_3d,
        optimize=True,
    )
    del div_v_sarray_3d
    S_r2tarray_3d = sym_r2tarray_3d - (1.0 / 3.0) * bulk_r2tarray_3d
    del sym_r2tarray_3d, bulk_r2tarray_3d
    return compute_r2tarray_divergence(
        r2tarray_3d=S_r2tarray_3d,
        cell_widths_3d=cell_widths_3d,
        grad_order=grad_order,
    )


## } MODULE
