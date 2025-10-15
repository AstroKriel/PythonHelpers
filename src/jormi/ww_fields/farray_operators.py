## { MODULE

##
## === DEPENDENCIES
##

import numpy
from numpy.typing import DTypeLike
from jormi.utils import type_utils, array_utils
from jormi.ww_fields import farray_types, finite_difference

##
## === WORKSPACE UTILITIES
##


def ensure_properties(
    array_shape: tuple[int, ...],
    dtype: DTypeLike | None,
    array: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """
    Return an array with the requested shape/dtype, reusing the provided array
    if compatible, otherwise allocates a new array.
    """
    if dtype is None:
        if array is not None:
            dtype = array.dtype
        else:
            dtype = numpy.float64
    else:
        dtype = numpy.dtype(dtype)
    if (array is None) or (array.shape != array_shape) or (array.dtype != dtype):
        return numpy.empty(array_shape, dtype=dtype)
    return array


##
## === OPTIMISED OPERATORS WORKING ON ARRAYS
##


def _as_float_view(
    array: numpy.ndarray,
) -> numpy.ndarray:
    ## promote integers/low-precision to float64 for safe reductions
    dtype = numpy.result_type(array.dtype, numpy.float64)
    return array.astype(dtype, copy=False)


def compute_sarray_rms(
    sarray: numpy.ndarray,
) -> float:
    farray_types.ensure_sarray(sarray)
    _sarray = _as_float_view(sarray)
    return float(numpy.sqrt(numpy.mean(numpy.square(_sarray))))


def compute_sarray_volume_integral(
    sarray: numpy.ndarray,
    cell_volume: float,
) -> float:
    farray_types.ensure_sarray(sarray)
    _sarray = _as_float_view(sarray)
    return float(cell_volume * numpy.sum(_sarray))


def compute_sarray_grad(
    sarray: numpy.ndarray,
    cell_widths: tuple[float, float, float] | list[float],
    grad_order: int = 2,
    out_varray: numpy.ndarray | None = None,
) -> numpy.ndarray:
    farray_types.ensure_sarray(sarray)
    farray_types.ensure_valid_cell_widths(cell_widths)
    nabla = finite_difference.get_grad_func(grad_order)
    num_cells_x, num_cells_y, num_cells_z = sarray.shape
    dtype = numpy.result_type(sarray.dtype, numpy.float64)
    grad_varray = ensure_properties(
        array_shape=(3, num_cells_x, num_cells_y, num_cells_z),
        dtype=dtype,
        array=out_varray,
    )
    farray_types.ensure_varray(grad_varray)
    cell_width_x, cell_width_y, cell_width_z = cell_widths
    ## fill ds/dx_i vector: (gradient-dir-i, x, y, z)
    grad_varray[0, ...] = nabla(sarray=sarray, cell_width=cell_width_x, grad_axis=0)
    grad_varray[1, ...] = nabla(sarray=sarray, cell_width=cell_width_y, grad_axis=1)
    grad_varray[2, ...] = nabla(sarray=sarray, cell_width=cell_width_z, grad_axis=2)
    return grad_varray


def sum_of_squared_components(
    varray: numpy.ndarray,
    out_sarray: numpy.ndarray | None = None,
    tmp_sarray: numpy.ndarray | None = None,
) -> numpy.ndarray:
    farray_types.ensure_varray(varray)
    domain_shape = varray.shape[1:]
    dtype = numpy.result_type(varray.dtype, numpy.float64)
    out_sarray = ensure_properties(
        array_shape=domain_shape,
        dtype=dtype,
        array=out_sarray,
    )
    farray_types.ensure_sarray(out_sarray)
    tmp_sarray = ensure_properties(
        array_shape=domain_shape,
        dtype=dtype,
        array=tmp_sarray,
    )
    farray_types.ensure_sarray(tmp_sarray)
    numpy.multiply(varray[0], varray[0], out=out_sarray)  # out = v_x^2
    numpy.multiply(varray[1], varray[1], out=tmp_sarray)  # tmp = v_y^2
    numpy.add(out_sarray, tmp_sarray, out=out_sarray)  # out = v_x^2 + v_y^2
    numpy.multiply(varray[2], varray[2], out=tmp_sarray)  # tmp = v_z^2
    numpy.add(out_sarray, tmp_sarray, out=out_sarray)  # out = v_x^2 + v_y^2 + v_z^2
    return out_sarray


def dot_over_components(
    varray_a: numpy.ndarray,
    varray_b: numpy.ndarray,
    out_sarray: numpy.ndarray | None = None,
    tmp_sarray: numpy.ndarray | None = None,
) -> numpy.ndarray:
    farray_types.ensure_varray(varray_a)
    farray_types.ensure_varray(varray_b)
    array_utils.ensure_same_shape(
        array_a=varray_a,
        array_b=varray_b,
    )
    domain_shape = varray_a.shape[1:]
    dtype = numpy.result_type(varray_a.dtype, varray_b.dtype)
    out_sarray = ensure_properties(
        array_shape=domain_shape,
        dtype=dtype,
        array=out_sarray,
    )
    farray_types.ensure_sarray(out_sarray)
    tmp_sarray = ensure_properties(
        array_shape=domain_shape,
        dtype=dtype,
        array=tmp_sarray,
    )
    farray_types.ensure_sarray(tmp_sarray)
    numpy.multiply(varray_a[0], varray_b[0], out=out_sarray)  # out = a_x b_x
    numpy.multiply(varray_a[1], varray_b[1], out=tmp_sarray)  # tmp = a_y b_y
    numpy.add(out_sarray, tmp_sarray, out=out_sarray)  # out = a_x b_x + a_y b_y
    numpy.multiply(varray_a[2], varray_b[2], out=tmp_sarray)  # tmp = a_z b_z
    numpy.add(out_sarray, tmp_sarray, out=out_sarray)  # out = a_x b_x + a_y b_y + a_z b_z
    return out_sarray


def compute_varray_grad(
    varray: numpy.ndarray,
    cell_widths: tuple[float, float, float] | list[float],
    grad_order: int = 2,
    out_r2tarray: numpy.ndarray | None = None,
) -> numpy.ndarray:
    farray_types.ensure_varray(varray)
    farray_types.ensure_valid_cell_widths(cell_widths)
    nabla = finite_difference.get_grad_func(grad_order)
    num_cells_x, num_cells_y, num_cells_z = varray.shape[1:]
    dtype = numpy.result_type(varray.dtype, numpy.float64)
    grad_r2tarray = ensure_properties(
        array_shape=(3, 3, num_cells_x, num_cells_y, num_cells_z),
        dtype=dtype,
        array=out_r2tarray,
    )
    farray_types.ensure_r2tarray(grad_r2tarray)
    cell_width_x, cell_width_y, cell_width_z = cell_widths
    ## fill du_j/dx_i tensor: (component-j, gradient-dir-i, x, y, z)
    for comp_j in range(3):
        grad_r2tarray[comp_j, 0, ...] = nabla(sarray=varray[comp_j], cell_width=cell_width_x, grad_axis=0)
        grad_r2tarray[comp_j, 1, ...] = nabla(sarray=varray[comp_j], cell_width=cell_width_y, grad_axis=1)
        grad_r2tarray[comp_j, 2, ...] = nabla(sarray=varray[comp_j], cell_width=cell_width_z, grad_axis=2)
    return grad_r2tarray


## } MODULE
