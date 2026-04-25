## { MODULE

##
## === DEPENDENCIES
##

## third-party
import numpy
from typing import Any
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


def ensure_3d_cell_widths(
    cell_widths_3d: tuple[float, float, float] | list[float],
    *,
    param_name: str = "<cell_widths_3d>",
) -> None:
    """Strictly validate `cell_widths_3d` as a length-3 sequence of finite, positive floats."""
    validate_types.ensure_sequence(
        param=cell_widths_3d,
        param_name=param_name,
        allow_none=False,
        seq_length=3,
        valid_seq_types=validate_types.RuntimeTypes.Sequences.SequenceLike,
        valid_elem_types=validate_types.RuntimeTypes.Numerics.FloatLike,
    )
    for dim_index, cell_width in enumerate(cell_widths_3d):
        validate_types.ensure_finite_float(
            param=cell_width,
            param_name=f"{param_name}[{dim_index}]",
            allow_none=False,
            allow_zero=False,
            require_positive=True,
        )


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
    sarray_3d_float = _as_float_view(sarray_3d)
    return float(numpy.sqrt(numpy.mean(numpy.square(sarray_3d_float))))


def compute_sarray_volume_integral(
    *,
    sarray_3d: NDArray[Any],
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
    sarray_3d_float = _as_float_view(sarray_3d)
    return float(cell_volume * numpy.sum(sarray_3d_float))


def compute_sarray_grad(
    *,
    sarray_3d: NDArray[Any],
    cell_widths_3d: tuple[float, float, float],
    varray_3d_out: NDArray[Any] | None = None,
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
    ensure_3d_cell_widths(cell_widths_3d)
    validate_types.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    nabla = difference_sarrays.get_grad_fn(grad_order)
    num_cells_x, num_cells_y, num_cells_z = sarray_3d.shape
    dtype = numpy.result_type(sarray_3d.dtype, numpy.float64)
    varray_3d_gradf = farray_types.ensure_farray_metadata(
        farray_shape=(3, num_cells_x, num_cells_y, num_cells_z),
        farray=varray_3d_out,
        dtype=dtype,
    )
    cell_width_x, cell_width_y, cell_width_z = cell_widths_3d
    ## fill d_i f vector: (gradient-dir-i, x0, x1, x2)
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
    sarray_3d: NDArray[Any],
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


def sum_of_varray_comps_squared(
    varray_3d: NDArray[Any],
    *,
    sarray_3d_out: NDArray[Any] | None = None,
    sarray_3d_tmp: NDArray[Any] | None = None,
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
    sarray_3d_out = farray_types.ensure_farray_metadata(
        farray_shape=domain_shape,
        farray=sarray_3d_out,
        dtype=dtype,
    )
    sarray_3d_tmp = farray_types.ensure_farray_metadata(
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
    varray_3d_a: NDArray[Any],
    varray_3d_b: NDArray[Any],
    sarray_3d_out: NDArray[Any] | None = None,
    sarray_3d_tmp: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """
    Compute vec(a) dot vec(b) per cell for 3D vector fields.

    Each input has shape (3, num_x0_cells, num_x1_cells, num_x2_cells).
    Returns a 3D ndarray with shape (num_x0_cells, num_x1_cells, num_x2_cells).
    """
    farray_types.ensure_3d_varray(
        varray_3d=varray_3d_a,
        param_name="<varray_3d_a>",
    )
    farray_types.ensure_3d_varray(
        varray_3d=varray_3d_b,
        param_name="<varray_3d_b>",
    )
    validate_arrays.ensure_same_shape(
        array_a=varray_3d_a,
        array_b=varray_3d_b,
        param_name_a="<varray_3d_a>",
        param_name_b="<varray_3d_b>",
    )
    domain_shape = varray_3d_a.shape[1:]
    dtype = numpy.result_type(varray_3d_a.dtype, varray_3d_b.dtype)
    sarray_3d_out = farray_types.ensure_farray_metadata(
        farray_shape=domain_shape,
        farray=sarray_3d_out,
        dtype=dtype,
    )
    sarray_3d_tmp = farray_types.ensure_farray_metadata(
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


def compute_varray_directional_derivative(
    *,
    varray_3d_direction: NDArray[Any],
    varray_3d_target: NDArray[Any],
    cell_widths_3d: tuple[float, float, float],
    out_varray_3d: NDArray[Any] | None = None,
    grad_order: int = 2,
) -> NDArray[Any]:
    """
    Compute the directional derivative `(direction_i d_i target_j)` for 3D vector fields.

    All vector arrays have shape `(3, num_x0_cells, num_x1_cells, num_x2_cells)`.
    """
    farray_types.ensure_3d_varray(
        varray_3d=varray_3d_direction,
        param_name="<varray_3d_direction>",
    )
    farray_types.ensure_3d_varray(
        varray_3d=varray_3d_target,
        param_name="<varray_3d_target>",
    )
    validate_arrays.ensure_same_shape(
        array_a=varray_3d_direction,
        array_b=varray_3d_target,
        param_name_a="<varray_3d_direction>",
        param_name_b="<varray_3d_target>",
    )
    ensure_3d_cell_widths(cell_widths_3d)
    validate_types.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    nabla = difference_sarrays.get_grad_fn(grad_order)
    domain_shape = varray_3d_direction.shape[1:]
    dtype = numpy.result_type(
        varray_3d_direction.dtype,
        varray_3d_target.dtype,
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
        sarray_3d_direction_comp = varray_3d_direction[grad_axis]
        for comp_index in range(3):
            sarray_3d_grad_comp = nabla(
                sarray_3d=varray_3d_target[comp_index],
                cell_width=cell_width,
                grad_axis=grad_axis,
            )
            out_varray_3d[comp_index] += sarray_3d_direction_comp * sarray_3d_grad_comp
    return out_varray_3d


def compute_varray_grad(
    *,
    varray_3d: NDArray[Any],
    cell_widths_3d: tuple[float, float, float],
    r2tarray_3d_gradf: NDArray[Any] | None = None,
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
    ensure_3d_cell_widths(cell_widths_3d)
    validate_types.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    nabla = difference_sarrays.get_grad_fn(grad_order)
    num_cells_x, num_cells_y, num_cells_z = varray_3d.shape[1:]
    dtype = numpy.result_type(varray_3d.dtype, numpy.float64)
    r2tarray_3d_gradf = farray_types.ensure_farray_metadata(
        farray_shape=(3, 3, num_cells_x, num_cells_y, num_cells_z),
        farray=r2tarray_3d_gradf,
        dtype=dtype,
    )
    cell_width_x, cell_width_y, cell_width_z = cell_widths_3d
    ## fill d_i f_j tensor: (component-j, gradient-dir-i, x0, x1, x2)
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
    varray_3d_a: NDArray[Any],
    varray_3d_b: NDArray[Any],
    varray_3d_out: NDArray[Any] | None = None,
    sarray_3d_tmp: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """
    Compute the cross product epsilon_ijk a_j b_k cellwise for two 3D vector fields.

    All arrays have shape (3, num_x0_cells, num_x1_cells, num_x2_cells).
    """
    farray_types.ensure_3d_varray(
        varray_3d=varray_3d_a,
        param_name="<varray_3d_a>",
    )
    farray_types.ensure_3d_varray(
        varray_3d=varray_3d_b,
        param_name="<varray_3d_b>",
    )
    validate_arrays.ensure_same_shape(
        array_a=varray_3d_a,
        array_b=varray_3d_b,
        param_name_a="<varray_3d_a>",
        param_name_b="<varray_3d_b>",
    )
    domain_shape = varray_3d_a.shape[1:]
    dtype = numpy.result_type(varray_3d_a.dtype, varray_3d_b.dtype)
    varray_3d_axb = farray_types.ensure_farray_metadata(
        farray_shape=varray_3d_a.shape,
        dtype=dtype,
        farray=varray_3d_out,
    )
    sarray_3d_tmp = farray_types.ensure_farray_metadata(
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
    varray_3d: NDArray[Any],
    cell_widths_3d: tuple[float, float, float],
    varray_3d_out: NDArray[Any] | None = None,
    grad_order: int = 2,
) -> NDArray[Any]:
    """
    Compute the curl epsilon_ijk d_j f_k of a 3D vector field.

    varray_3d and varray_3d_out have shape (3, num_x0_cells, num_x1_cells, num_x2_cells).
    """
    farray_types.ensure_3d_varray(
        varray_3d=varray_3d,
        param_name="<varray_3d>",
    )
    ensure_3d_cell_widths(cell_widths_3d)
    validate_types.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    nabla = difference_sarrays.get_grad_fn(grad_order)
    cell_width_x, cell_width_y, cell_width_z = cell_widths_3d
    varray_3d_curl = farray_types.ensure_farray_metadata(
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
    varray_3d: NDArray[Any],
    cell_widths_3d: tuple[float, float, float],
    sarray_3d_out: NDArray[Any] | None = None,
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
    ensure_3d_cell_widths(cell_widths_3d)
    validate_types.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    nabla = difference_sarrays.get_grad_fn(grad_order)
    cell_width_x, cell_width_y, cell_width_z = cell_widths_3d
    domain_shape = varray_3d.shape[1:]
    sarray_3d_div = farray_types.ensure_farray_metadata(
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
    varray_3d: NDArray[Any],
    sarray_3d_out: NDArray[Any] | None = None,
    sarray_3d_tmp: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """
    Compute |v| = sqrt(v_i v_i) per cell for a 3D vector field.

    Returns a 3D ndarray with shape (num_x0_cells, num_x1_cells, num_x2_cells).
    """
    farray_types.ensure_3d_varray(
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


##
## === UNIT VECTOR (3D) ARRAY OPERATORS
##


def ensure_uvarray_magnitude(
    varray_3d: NDArray[Any],
    *,
    tol: float = 1e-6,
    param_name: str = "<varray_3d>",
) -> None:
    """
    Validate that every vector in a (3, num_x0_cells, num_x1_cells, num_x2_cells) array has unit magnitude.

    Raises ValueError if any element is non-finite, or if any magnitude deviates
    from 1.0 by more than `tol`.
    """
    farray_types.ensure_3d_varray(
        varray_3d=varray_3d,
        param_name=param_name,
    )
    sarray_3d_vmagn_sq = sum_of_varray_comps_squared(varray_3d=varray_3d)
    if not numpy.all(numpy.isfinite(sarray_3d_vmagn_sq)):
        raise ValueError(
            f"`{param_name}` should not contain any NaN/Inf magnitudes.",
        )
    max_error = float(
        numpy.max(
            numpy.abs(numpy.sqrt(sarray_3d_vmagn_sq) - 1.0),
        ),
    )
    if max_error > tol:
        raise ValueError(
            f"`{param_name}` magnitude deviates from unit-magnitude=1.0 by"
            f" max(error)={max_error:.3e} (tol={tol}).",
        )


## } MODULE
