## { MODULE

##
## === DEPENDENCIES
##

import numpy

from numpy.typing import DTypeLike

from jormi.ww_types import array_checks, fdata_types
from jormi.ww_fields import finite_difference

##
## === WORKSPACE UTILITIES
##


def ensure_properties(
    *,
    farray_shape: tuple[int, ...],
    farray: numpy.ndarray | None = None,
    dtype: DTypeLike | None = None,
) -> numpy.ndarray:
    """
    Return a farray with the requested shape/dtype, reusing the provided farray
    if compatible, otherwise allocate a new farray.
    """
    if dtype is None:
        if farray is not None:
            dtype = farray.dtype
        else:
            dtype = numpy.float64
    else:
        dtype = numpy.dtype(dtype)
    if (farray is None) or (farray.shape != farray_shape) or (farray.dtype != dtype):
        return numpy.empty(farray_shape, dtype=dtype)
    return farray


##
## === OPTIMISED OPERATORS WORKING ON FDATA (AND RAW FARRAYS)
##


def _as_float_view(
    farray: numpy.ndarray,
) -> numpy.ndarray:
    """Promote from integers/low-precision to float64 for safe reductions."""
    dtype = numpy.result_type(farray.dtype, numpy.float64)
    return farray.astype(dtype, copy=False)


def compute_sdata_rms(
    sdata: fdata_types.ScalarFieldData | numpy.ndarray,
) -> float:
    """
    Compute the RMS of scalar-field array.

    Accepts:
      - ScalarFieldData (3D, (Nx, Ny, Nz)), or
      - raw 3D ndarray with shape (Nx, Ny, Nz).
    """
    sarray = fdata_types.as_3d_sarray(
        sdata=sdata,
        param_name="<sdata>",
    )
    sarray_float = _as_float_view(sarray)
    return float(numpy.sqrt(numpy.mean(numpy.square(sarray_float))))


def compute_sdata_volume_integral(
    sdata: fdata_types.ScalarFieldData | numpy.ndarray,
    cell_volume: float,
) -> float:
    """
    Compute the volume integral of scalar-field array.

    Accepts:
      - ScalarFieldData (3D, (Nx, Ny, Nz)), or
      - raw 3D ndarray with shape (Nx, Ny, Nz).
    """
    sarray = fdata_types.as_3d_sarray(
        sdata=sdata,
        param_name="<sdata>",
    )
    sarray_float = _as_float_view(sarray)
    return float(cell_volume * numpy.sum(sarray_float))


def compute_sdata_grad(
    *,
    sdata: fdata_types.ScalarFieldData | numpy.ndarray,
    cell_widths: tuple[float, float, float] | list[float],
    out_varray: numpy.ndarray | None = None,
    grad_order: int = 2,
) -> numpy.ndarray:
    """
    Compute the gradient of a scalar-field array.

    Returns a 4D ndarray with shape (3, Nx, Ny, Nz).
    """
    sarray = fdata_types.as_3d_sarray(
        sdata=sdata,
        param_name="<sdata>",
    )
    nabla = finite_difference.get_grad_fn(grad_order)
    num_cells_x, num_cells_y, num_cells_z = sarray.shape
    dtype = numpy.result_type(sarray.dtype, numpy.float64)
    grad_varray = ensure_properties(
        farray_shape=(3, num_cells_x, num_cells_y, num_cells_z),
        farray=out_varray,
        dtype=dtype,
    )
    fdata_types.ensure_3d_varray(
        varray=grad_varray,
        param_name="<grad_varray>",
    )
    cell_width_x, cell_width_y, cell_width_z = cell_widths
    ## fill d_i f vector: (gradient-dir-i, x, y, z)
    grad_varray[0, ...] = nabla(
        sarray=sarray,
        cell_width=cell_width_x,
        grad_axis=0,
    )
    grad_varray[1, ...] = nabla(
        sarray=sarray,
        cell_width=cell_width_y,
        grad_axis=1,
    )
    grad_varray[2, ...] = nabla(
        sarray=sarray,
        cell_width=cell_width_z,
        grad_axis=2,
    )
    return grad_varray


def sum_of_squared_components(
    *,
    vdata: fdata_types.VectorFieldData | numpy.ndarray,
    out_sarray: numpy.ndarray | None = None,
    tmp_sarray: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """
    Compute sum_i v_i^2 per cell.

    Accepts:
      - VectorFieldData (3, Nx, Ny, Nz), or
      - raw 4D ndarray with shape (3, Nx, Ny, Nz).

    Returns a 3D ndarray with shape (Nx, Ny, Nz).
    """
    varray = fdata_types.as_3d_varray(
        vdata=vdata,
        param_name="<vdata>",
    )
    domain_shape = varray.shape[1:]
    dtype = numpy.result_type(varray.dtype, numpy.float64)
    out_sarray = ensure_properties(
        farray_shape=domain_shape,
        farray=out_sarray,
        dtype=dtype,
    )
    fdata_types.ensure_3d_sarray(
        sarray=out_sarray,
        param_name="<out_sarray>",
    )
    tmp_sarray = ensure_properties(
        farray_shape=domain_shape,
        farray=tmp_sarray,
        dtype=dtype,
    )
    fdata_types.ensure_3d_sarray(
        sarray=tmp_sarray,
        param_name="<tmp_sarray>",
    )
    numpy.multiply(varray[0], varray[0], out=out_sarray)  # out = v_x^2
    numpy.multiply(varray[1], varray[1], out=tmp_sarray)  # tmp = v_y^2
    numpy.add(out_sarray, tmp_sarray, out=out_sarray)  # out = v_x^2 + v_y^2
    numpy.multiply(varray[2], varray[2], out=tmp_sarray)  # tmp = v_z^2
    numpy.add(out_sarray, tmp_sarray, out=out_sarray)  # out = v_x^2 + v_y^2 + v_z^2
    return out_sarray


def dot_over_components(
    *,
    vdata_a: fdata_types.VectorFieldData | numpy.ndarray,
    vdata_b: fdata_types.VectorFieldData | numpy.ndarray,
    out_sarray: numpy.ndarray | None = None,
    tmp_sarray: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """
    Compute vec(a) dot vec(b) per cell.

    Accepts:
      - VectorFieldData or ndarray for each input, each with shape (3, Nx, Ny, Nz).
    """
    varray_a = fdata_types.as_3d_varray(
        vdata=vdata_a,
        param_name="<vdata_a>",
    )
    varray_b = fdata_types.as_3d_varray(
        vdata=vdata_b,
        param_name="<vdata_b>",
    )
    array_checks.ensure_same_shape(
        array_a=varray_a,
        array_b=varray_b,
        param_name_a="<vdata_a.varray>",
        param_name_b="<vdata_b.varray>",
    )
    domain_shape = varray_a.shape[1:]
    dtype = numpy.result_type(varray_a.dtype, varray_b.dtype)
    out_sarray = ensure_properties(
        farray_shape=domain_shape,
        farray=out_sarray,
        dtype=dtype,
    )
    fdata_types.ensure_3d_sarray(
        sarray=out_sarray,
        param_name="<out_sarray>",
    )
    tmp_sarray = ensure_properties(
        farray_shape=domain_shape,
        farray=tmp_sarray,
        dtype=dtype,
    )
    fdata_types.ensure_3d_sarray(
        sarray=tmp_sarray,
        param_name="<tmp_sarray>",
    )
    numpy.multiply(varray_a[0], varray_b[0], out=out_sarray)  # out = a_x b_x
    numpy.multiply(varray_a[1], varray_b[1], out=tmp_sarray)  # tmp = a_y b_y
    numpy.add(out_sarray, tmp_sarray, out=out_sarray)  # out = a_x b_x + a_y b_y
    numpy.multiply(varray_a[2], varray_b[2], out=tmp_sarray)  # tmp = a_z b_z
    numpy.add(out_sarray, tmp_sarray, out=out_sarray)  # out = a_x b_x + a_y b_y + a_z b_z
    return out_sarray


def compute_varray_grad(
    *,
    vdata: fdata_types.VectorFieldData | numpy.ndarray,
    cell_widths: tuple[float, float, float] | list[float],
    out_r2tarray: numpy.ndarray | None = None,
    grad_order: int = 2,
) -> numpy.ndarray:
    """
    Compute gradient of a vector field.

    Accepts:
      - VectorFieldData or ndarray with shape (3, Nx, Ny, Nz).

    Returns:
      - r2tarray with shape (3, 3, Nx, Ny, Nz),
        where grad[comp_j, grad_i, ...] = d_i v_j.
    """
    varray = fdata_types.as_3d_varray(
        vdata=vdata,
        param_name="<vdata>",
    )
    nabla = finite_difference.get_grad_fn(grad_order)
    num_cells_x, num_cells_y, num_cells_z = varray.shape[1:]
    dtype = numpy.result_type(varray.dtype, numpy.float64)
    grad_r2tarray = ensure_properties(
        farray_shape=(3, 3, num_cells_x, num_cells_y, num_cells_z),
        farray=out_r2tarray,
        dtype=dtype,
    )
    fdata_types.ensure_3d_r2tarray(
        r2tarray=grad_r2tarray,
        param_name="<grad_r2tarray>",
    )
    cell_width_x, cell_width_y, cell_width_z = cell_widths
    ## fill d_i f_j tensor: (component-j, gradient-dir-i, x, y, z)
    for comp_j in range(3):
        grad_r2tarray[comp_j, 0, ...] = nabla(
            sarray=varray[comp_j],
            cell_width=cell_width_x,
            grad_axis=0,
        )
        grad_r2tarray[comp_j, 1, ...] = nabla(
            sarray=varray[comp_j],
            cell_width=cell_width_y,
            grad_axis=1,
        )
        grad_r2tarray[comp_j, 2, ...] = nabla(
            sarray=varray[comp_j],
            cell_width=cell_width_z,
            grad_axis=2,
        )
    return grad_r2tarray


## } MODULE
