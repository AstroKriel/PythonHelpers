## { MODULE

##
## === DEPENDENCIES
##

import numpy

from jormi.ww_types import type_checks, array_checks
from jormi.ww_fields.fields_3d import (
    _finite_difference_sarrays,
    _fdata_type,
)

##
## === WORKSPACE UTILITIES
##


def _as_float_view(
    farray: numpy.ndarray,
) -> numpy.ndarray:
    """Promote from integers/low-precision to float64 for safe reductions."""
    array_checks.ensure_array(
        array=farray,
        param_name="<farray>",
    )
    dtype = numpy.result_type(farray.dtype, numpy.float64)
    return farray.astype(dtype, copy=False)


def _validate_3d_cell_widths(
    cell_widths_3d: tuple[float, float, float] | list[float],
    *,
    param_name: str = "<cell_widths_3d>",
) -> None:
    """Strictly validate `cell_widths_3d` as a length-3 sequence of finite, positive floats."""
    type_checks.ensure_sequence(
        param=cell_widths_3d,
        param_name=param_name,
        allow_none=False,
        seq_length=3,
        valid_seq_types=(tuple, list),
        valid_elem_types=type_checks.RuntimeTypes.Numerics.FloatLike,
    )
    for dim_index, cell_width in enumerate(cell_widths_3d):
        type_checks.ensure_finite_float(
            param=cell_width,
            param_name=f"{param_name}[{dim_index}]",
            allow_none=False,
            require_positive=True,
        )


##
## === SCALAR (3D) ARRAY OPERATORS
##


def compute_sarray_rms(
    sarray_3d: numpy.ndarray,
) -> float:
    """Compute the RMS of a 3D scalar array."""
    _fdata_type.ensure_3d_sarray(
        sarray_3d=sarray_3d,
        param_name="<sarray_3d>",
    )
    sarray_3d_float = _as_float_view(sarray_3d)
    return float(numpy.sqrt(numpy.mean(numpy.square(sarray_3d_float))))


def compute_sarray_volume_integral(
    *,
    sarray_3d: numpy.ndarray,
    cell_volume: float,
) -> float:
    """Compute the volume integral of a 3D scalar array."""
    _fdata_type.ensure_3d_sarray(
        sarray_3d=sarray_3d,
        param_name="<sarray_3d>",
    )
    type_checks.ensure_finite_float(
        param=cell_volume,
        param_name="<cell_volume>",
        allow_none=False,
        require_positive=True,
    )
    sarray_3d_float = _as_float_view(sarray_3d)
    return float(cell_volume * numpy.sum(sarray_3d_float))


def compute_sarray_grad(
    *,
    sarray_3d: numpy.ndarray,
    cell_widths_3d: tuple[float, float, float],
    varray_3d_out: numpy.ndarray | None = None,
    grad_order: int = 2,
) -> numpy.ndarray:
    """
    Compute the gradient of a 3D scalar array.

    Returns a 4D ndarray with shape (3, Nx, Ny, Nz).
    """
    _fdata_type.ensure_3d_sarray(
        sarray_3d=sarray_3d,
        param_name="<sarray_3d>",
    )
    _validate_3d_cell_widths(cell_widths_3d)
    type_checks.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    nabla = _finite_difference_sarrays.get_grad_fn(grad_order)
    num_cells_x, num_cells_y, num_cells_z = sarray_3d.shape
    dtype = numpy.result_type(sarray_3d.dtype, numpy.float64)
    varray_3d_gradf = _fdata_type.ensure_farray_metadata(
        farray_shape=(3, num_cells_x, num_cells_y, num_cells_z),
        farray=varray_3d_out,
        dtype=dtype,
    )
    cell_width_x, cell_width_y, cell_width_z = cell_widths_3d
    ## fill d_i f vector: (gradient-dir-i, x, y, z)
    varray_3d_gradf[0, ...] = nabla(
        sarray_3d=sarray_3d,
        cell_width=cell_width_x,
        grad_axis=0,
    )
    varray_3d_gradf[1, ...] = nabla(
        sarray_3d=sarray_3d,
        cell_width=cell_width_y,
        grad_axis=1,
    )
    varray_3d_gradf[2, ...] = nabla(
        sarray_3d=sarray_3d,
        cell_width=cell_width_z,
        grad_axis=2,
    )
    return varray_3d_gradf


def scale_sarray_inplace(
    *,
    sarray_3d: numpy.ndarray,
    scale: float,
) -> None:
    """Scale a 3D scalar array in-place by a scalar factor."""
    _fdata_type.ensure_3d_sarray(
        sarray_3d=sarray_3d,
        param_name="<sarray_3d>",
    )
    sarray_3d *= numpy.asarray(scale, dtype=sarray_3d.dtype)


def sqrt_sarray_inplace(
    sarray_3d: numpy.ndarray,
) -> None:
    """Take the square-root of a 3D scalar array in-place."""
    _fdata_type.ensure_3d_sarray(
        sarray_3d=sarray_3d,
        param_name="<sarray_3d>",
    )
    numpy.sqrt(sarray_3d, out=sarray_3d)


##
## === VECTOR (3D) ARRAY OPERATORS
##


def sum_of_varray_comps_squared(
    varray_3d: numpy.ndarray,
    *,
    sarray_3d_out: numpy.ndarray | None = None,
    sarray_3d_tmp: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """
    Compute sum_i (v_i v_i) per cell for a 3D vector field.

    varray_3d has shape (3, Nx, Ny, Nz).
    Returns a 3D ndarray with shape (Nx, Ny, Nz).
    """
    _fdata_type.ensure_3d_varray(
        varray_3d=varray_3d,
        param_name="<varray_3d>",
    )
    domain_shape = varray_3d.shape[1:]
    dtype = numpy.result_type(varray_3d.dtype, numpy.float64)
    sarray_3d_out = _fdata_type.ensure_farray_metadata(
        farray_shape=domain_shape,
        farray=sarray_3d_out,
        dtype=dtype,
    )
    sarray_3d_tmp = _fdata_type.ensure_farray_metadata(
        farray_shape=domain_shape,
        farray=sarray_3d_tmp,
        dtype=dtype,
    )
    numpy.multiply(varray_3d[0], varray_3d[0], out=sarray_3d_out)  # out = v_x^2
    numpy.multiply(varray_3d[1], varray_3d[1], out=sarray_3d_tmp)  # tmp = v_y^2
    numpy.add(sarray_3d_out, sarray_3d_tmp, out=sarray_3d_out)  # out = v_x^2 + v_y^2
    numpy.multiply(varray_3d[2], varray_3d[2], out=sarray_3d_tmp)  # tmp = v_z^2
    numpy.add(sarray_3d_out, sarray_3d_tmp, out=sarray_3d_out)  # out = v_x^2 + v_y^2 + v_z^2
    return sarray_3d_out


def dot_over_varray_comps(
    *,
    varray_3d_a: numpy.ndarray,
    varray_3d_b: numpy.ndarray,
    sarray_3d_out: numpy.ndarray | None = None,
    sarray_3d_tmp: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """
    Compute vec(a) dot vec(b) per cell for 3D vector fields.

    Each input has shape (3, Nx, Ny, Nz).
    Returns a 3D ndarray with shape (Nx, Ny, Nz).
    """
    _fdata_type.ensure_3d_varray(
        varray_3d=varray_3d_a,
        param_name="<varray_3d_a>",
    )
    _fdata_type.ensure_3d_varray(
        varray_3d=varray_3d_b,
        param_name="<varray_3d_b>",
    )
    array_checks.ensure_same_shape(
        array_a=varray_3d_a,
        array_b=varray_3d_b,
        param_name_a="<varray_3d_a>",
        param_name_b="<varray_3d_b>",
    )
    domain_shape = varray_3d_a.shape[1:]
    dtype = numpy.result_type(varray_3d_a.dtype, varray_3d_b.dtype)
    sarray_3d_out = _fdata_type.ensure_farray_metadata(
        farray_shape=domain_shape,
        farray=sarray_3d_out,
        dtype=dtype,
    )
    sarray_3d_tmp = _fdata_type.ensure_farray_metadata(
        farray_shape=domain_shape,
        farray=sarray_3d_tmp,
        dtype=dtype,
    )
    numpy.multiply(varray_3d_a[0], varray_3d_b[0], out=sarray_3d_out)  # out = a_x b_x
    numpy.multiply(varray_3d_a[1], varray_3d_b[1], out=sarray_3d_tmp)  # tmp = a_y b_y
    numpy.add(sarray_3d_out, sarray_3d_tmp, out=sarray_3d_out)  # out += a_y b_y
    numpy.multiply(varray_3d_a[2], varray_3d_b[2], out=sarray_3d_tmp)  # tmp = a_z b_z
    numpy.add(sarray_3d_out, sarray_3d_tmp, out=sarray_3d_out)  # out += a_z b_z
    return sarray_3d_out


def compute_varray_grad(
    *,
    varray_3d: numpy.ndarray,
    cell_widths_3d: tuple[float, float, float],
    r2tarray_3d_gradf: numpy.ndarray | None = None,
    grad_order: int = 2,
) -> numpy.ndarray:
    """
    Compute gradient of a 3D vector field.

    varray_3d has shape (3, Nx, Ny, Nz).

    Returns:
      - r2tarray_3d with shape (3, 3, Nx, Ny, Nz),
        where grad[comp_j, grad_i, ...] = d_i v_j.
    """
    _fdata_type.ensure_3d_varray(
        varray_3d=varray_3d,
        param_name="<varray_3d>",
    )
    _validate_3d_cell_widths(cell_widths_3d)
    type_checks.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    nabla = _finite_difference_sarrays.get_grad_fn(grad_order)
    num_cells_x, num_cells_y, num_cells_z = varray_3d.shape[1:]
    dtype = numpy.result_type(varray_3d.dtype, numpy.float64)
    r2tarray_3d_gradf = _fdata_type.ensure_farray_metadata(
        farray_shape=(3, 3, num_cells_x, num_cells_y, num_cells_z),
        farray=r2tarray_3d_gradf,
        dtype=dtype,
    )
    cell_width_x, cell_width_y, cell_width_z = cell_widths_3d
    ## fill d_i f_j tensor: (component-j, gradient-dir-i, x, y, z)
    for comp_j in range(3):
        r2tarray_3d_gradf[comp_j, 0, ...] = nabla(
            sarray_3d=varray_3d[comp_j],
            cell_width=cell_width_x,
            grad_axis=0,
        )
        r2tarray_3d_gradf[comp_j, 1, ...] = nabla(
            sarray_3d=varray_3d[comp_j],
            cell_width=cell_width_y,
            grad_axis=1,
        )
        r2tarray_3d_gradf[comp_j, 2, ...] = nabla(
            sarray_3d=varray_3d[comp_j],
            cell_width=cell_width_z,
            grad_axis=2,
        )
    return r2tarray_3d_gradf


def compute_varray_cross_product(
    *,
    varray_3d_a: numpy.ndarray,
    varray_3d_b: numpy.ndarray,
    varray_3d_out: numpy.ndarray | None = None,
    sarray_3d_tmp: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """
    Compute the cross product epsilon_ijk a_j b_k cellwise for two 3D vector fields.

    All arrays have shape (3, Nx, Ny, Nz).
    """
    _fdata_type.ensure_3d_varray(
        varray_3d=varray_3d_a,
        param_name="<varray_3d_a>",
    )
    _fdata_type.ensure_3d_varray(
        varray_3d=varray_3d_b,
        param_name="<varray_3d_b>",
    )
    array_checks.ensure_same_shape(
        array_a=varray_3d_a,
        array_b=varray_3d_b,
        param_name_a="<varray_3d_a>",
        param_name_b="<varray_3d_b>",
    )
    domain_shape = varray_3d_a.shape[1:]
    dtype = numpy.result_type(varray_3d_a.dtype, varray_3d_b.dtype)
    varray_3d_axb = _fdata_type.ensure_farray_metadata(
        farray_shape=varray_3d_a.shape,
        dtype=dtype,
        farray=varray_3d_out,
    )
    sarray_3d_tmp = _fdata_type.ensure_farray_metadata(
        farray_shape=domain_shape,
        dtype=dtype,
        farray=sarray_3d_tmp,
    )
    ## cross_x = a_y * b_z - a_z * b_y
    numpy.multiply(varray_3d_a[1], varray_3d_b[2], out=varray_3d_axb[0])
    numpy.multiply(varray_3d_a[2], varray_3d_b[1], out=sarray_3d_tmp)
    numpy.subtract(varray_3d_axb[0], sarray_3d_tmp, out=varray_3d_axb[0])
    ## cross_y = -a_x * b_z + a_z * b_x
    numpy.multiply(varray_3d_a[2], varray_3d_b[0], out=varray_3d_axb[1])
    numpy.multiply(varray_3d_a[0], varray_3d_b[2], out=sarray_3d_tmp)
    numpy.subtract(varray_3d_axb[1], sarray_3d_tmp, out=varray_3d_axb[1])
    ## cross_z = a_x * b_y - a_y * b_x
    numpy.multiply(varray_3d_a[0], varray_3d_b[1], out=varray_3d_axb[2])
    numpy.multiply(varray_3d_a[1], varray_3d_b[0], out=sarray_3d_tmp)
    numpy.subtract(varray_3d_axb[2], sarray_3d_tmp, out=varray_3d_axb[2])
    return varray_3d_axb


def compute_varray_curl(
    *,
    varray_3d: numpy.ndarray,
    cell_widths_3d: tuple[float, float, float],
    varray_3d_out: numpy.ndarray | None = None,
    grad_order: int = 2,
) -> numpy.ndarray:
    """
    Compute the curl epsilon_ijk d_j f_k of a 3D vector field.

    varray_3d and varray_3d_out have shape (3, Nx, Ny, Nz).
    """
    _fdata_type.ensure_3d_varray(
        varray_3d=varray_3d,
        param_name="<varray_3d>",
    )
    _validate_3d_cell_widths(cell_widths_3d)
    type_checks.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    nabla = _finite_difference_sarrays.get_grad_fn(grad_order)
    cell_width_x, cell_width_y, cell_width_z = cell_widths_3d
    varray_3d_curl = _fdata_type.ensure_farray_metadata(
        farray_shape=varray_3d.shape,
        dtype=varray_3d.dtype,
        farray=varray_3d_out,
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
        out=varray_3d_curl[0],
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
        out=varray_3d_curl[1],
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
        out=varray_3d_curl[2],
    )
    return varray_3d_curl


def compute_varray_divergence(
    *,
    varray_3d: numpy.ndarray,
    cell_widths_3d: tuple[float, float, float],
    sarray_3d_out: numpy.ndarray | None = None,
    grad_order: int = 2,
) -> numpy.ndarray:
    """
    Compute the divergence d_i f_i of a 3D vector field.

    varray_3d has shape (3, Nx, Ny, Nz).
    """
    _fdata_type.ensure_3d_varray(
        varray_3d=varray_3d,
        param_name="<varray_3d>",
    )
    _validate_3d_cell_widths(cell_widths_3d)
    type_checks.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    nabla = _finite_difference_sarrays.get_grad_fn(grad_order)
    cell_width_x, cell_width_y, cell_width_z = cell_widths_3d
    domain_shape = varray_3d.shape[1:]
    sarray_3d_div = _fdata_type.ensure_farray_metadata(
        farray_shape=domain_shape,
        dtype=varray_3d.dtype,
        farray=sarray_3d_out,
    )
    ## start with d_i f_i for i=1, then add i=2 and i=3 in-place
    sarray_3d_div[...] = nabla(
        sarray_3d=varray_3d[0],
        cell_width=cell_width_x,
        grad_axis=0,
    )
    numpy.add(
        sarray_3d_div,
        nabla(
            sarray_3d=varray_3d[1],
            cell_width=cell_width_y,
            grad_axis=1,
        ),
        out=sarray_3d_div,
    )
    numpy.add(
        sarray_3d_div,
        nabla(
            sarray_3d=varray_3d[2],
            cell_width=cell_width_z,
            grad_axis=2,
        ),
        out=sarray_3d_div,
    )
    return sarray_3d_div


def compute_varray_magnitude(
    *,
    varray_3d: numpy.ndarray,
    sarray_3d_out: numpy.ndarray | None = None,
    sarray_3d_tmp: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """
    Compute |v| = sqrt(v_i v_i) per cell for a 3D vector field.

    Returns a 3D ndarray with shape (Nx, Ny, Nz).
    """
    _fdata_type.ensure_3d_varray(
        varray_3d=varray_3d,
        param_name="<varray_3d>",
    )
    sarray_3d_vmagn_sq = sum_of_varray_comps_squared(
        varray_3d=varray_3d,
        sarray_3d_out=sarray_3d_out,
        sarray_3d_tmp=sarray_3d_tmp,
    )
    sqrt_sarray_inplace(sarray_3d_vmagn_sq)
    return sarray_3d_vmagn_sq


## } MODULE
