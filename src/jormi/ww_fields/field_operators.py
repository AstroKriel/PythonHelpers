## { MODULE

##
## === DEPENDENCIES
##

import numpy
from typing import Any
from numpy.typing import NDArray, DTypeLike
from jormi.ww_data import finite_difference
from jormi.ww_fields import field_types

##
## === WORKSPACE UTILITIES
##


def _ensure_array_properties(
    array_shape: tuple[int, ...],
    dtype: DTypeLike | None,
    array: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """
    Return a scratch array with the requested shape/dtype, reusing the provided array
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
    ## promote integers/low-precision to at least float64 for safe reductions
    return array.astype(numpy.result_type(array.dtype, numpy.float64), copy=False)


def _sum_of_component_squares(
    varray: numpy.ndarray,
    out_array: numpy.ndarray | None = None,
    tmp_array: numpy.ndarray | None = None,
) -> numpy.ndarray:
    field_types.ensure_varray(varray)
    domain_shape = varray.shape[1:]
    out_array = _ensure_array_properties(
        array_shape=domain_shape,
        dtype=varray.dtype,
        array=out_array,
    )
    tmp_array = _ensure_array_properties(
        array_shape=domain_shape,
        dtype=varray.dtype,
        array=tmp_array,
    )
    numpy.multiply(varray[0], varray[0], out=out_array)  # v_x^2
    numpy.multiply(varray[1], varray[1], out=tmp_array)  # v_y^2
    numpy.add(out_array, tmp_array, out=out_array)  # v_x^2 + v_y^2
    numpy.multiply(varray[2], varray[2], out=tmp_array)  # v_z^2
    numpy.add(out_array, tmp_array, out=out_array)  # v_x^2 + v_y^2 + v_z^2
    return out_array


def _dot_over_components(
    varray_a: numpy.ndarray,
    varray_b: numpy.ndarray,
    out_array: numpy.ndarray | None = None,
    tmp_array: numpy.ndarray | None = None,
) -> numpy.ndarray:
    field_types.ensure_varray(varray_a)
    field_types.ensure_varray(varray_b)
    field_types.ensure_same_grid(
        array_a=varray_a,
        array_b=varray_b,
    )
    domain_shape = varray_a.shape[1:]
    dtype = numpy.result_type(varray_a.dtype, varray_b.dtype)
    out_array = _ensure_array_properties(
        array_shape=domain_shape,
        dtype=dtype,
        array=out_array,
    )
    tmp_array = _ensure_array_properties(
        array_shape=domain_shape,
        dtype=dtype,
        array=tmp_array,
    )
    numpy.multiply(varray_a[0], varray_b[0], out=out_array)  # a_x b_x
    numpy.multiply(varray_a[1], varray_b[1], out=tmp_array)  # a_y b_y
    numpy.add(out_array, tmp_array, out=out_array)  # a_x b_x + a_y b_y
    numpy.multiply(varray_a[2], varray_b[2], out=tmp_array)  # a_z b_z
    numpy.add(out_array, tmp_array, out=out_array)  # a_x b_x + a_y b_y + a_z b_z
    return out_array


def compute_array_rms(
    sarray: numpy.ndarray,
) -> float:
    field_types.ensure_sarray(sarray)
    _sarray = _as_float_view(sarray)
    return float(numpy.sqrt(numpy.mean(numpy.square(_sarray))))


def compute_array_volume_integral(
    sarray: numpy.ndarray,
    cell_volume: float,
) -> float:
    field_types.ensure_sarray(sarray)
    _sarray = _as_float_view(sarray)
    return float(cell_volume * numpy.sum(_sarray))


##
## === OPTIMISED OPERATORS WORKING ON FIELDS
##


def compute_sfield_rms(
    sfield: field_types.ScalarField,
) -> float:
    field_types.ensure_sfield(sfield)
    return compute_array_rms(sfield.data)


def compute_sfield_volume_integral(
    sfield: field_types.ScalarField,
    domain_details: field_types.UniformDomain,
) -> float:
    field_types.ensure_sfield(sfield)
    field_types.ensure_uniform_domain(domain_details)
    field_types.ensure_domain_matches_sfield(
        domain_details=domain_details,
        sfield=sfield,
    )
    return compute_array_volume_integral(
        sarray=sfield.data,
        cell_volume=domain_details.cell_volume,
    )


def compute_vfield_magnitude(
    vfield: field_types.VectorField,
    label: str = "|vec(f)|",
) -> field_types.ScalarField:
    field_types.ensure_vfield(vfield)
    varray = vfield.data
    field_types.ensure_varray(varray)
    field_magn = _sum_of_component_squares(varray)  # allocates output (reused below)
    numpy.sqrt(field_magn, out=field_magn)  # in-place transform
    return field_types.ScalarField(
        sim_time=vfield.sim_time,
        data=field_magn,
        label=label,
    )


def compute_vfield_dot_product(
    vfield_a: field_types.VectorField,
    vfield_b: field_types.VectorField,
    label: str = "dot-product",
) -> field_types.ScalarField:
    field_types.ensure_vfield(vfield_a)
    field_types.ensure_vfield(vfield_b)
    varray_a = vfield_a.data
    varray_b = vfield_b.data
    field_types.ensure_varray(varray_a)
    field_types.ensure_varray(varray_b)
    field_types.ensure_same_grid(
        array_a=varray_a,
        array_b=varray_b,
    )
    dot_array = _dot_over_components(
        varray_a=varray_a,
        varray_b=varray_b,
    )
    return field_types.ScalarField(
        sim_time=vfield_a.sim_time,
        data=dot_array,
        label=label,
    )


def compute_vfield_cross_product(
    vfield_a: field_types.VectorField,
    vfield_b: field_types.VectorField,
    labels: tuple[str, str, str] = ("cross-x", "cross-y", "cross-z"),
) -> field_types.VectorField:
    field_types.ensure_vfield(vfield_a)
    field_types.ensure_vfield(vfield_b)
    varray_a = vfield_a.data
    varray_b = vfield_b.data
    field_types.ensure_varray(varray_a)
    field_types.ensure_varray(varray_b)
    field_types.ensure_same_grid(
        array_a=varray_a,
        array_b=varray_b,
    )
    dtype = numpy.result_type(varray_a.dtype, varray_b.dtype)
    cross_array = numpy.empty(varray_a.shape, dtype=dtype)
    cross_array[0] = varray_a[1] * varray_b[2] - varray_a[2] * varray_b[1]
    cross_array[1] = -varray_a[0] * varray_b[2] + varray_a[2] * varray_b[0]
    cross_array[2] = varray_a[0] * varray_b[1] - varray_a[1] * varray_b[0]
    return field_types.VectorField(
        sim_time=vfield_a.sim_time,
        data=cross_array,
        labels=labels,
    )


def compute_vfield_curl(
    vfield: field_types.VectorField,
    domain_details: field_types.UniformDomain,
    labels: tuple[str, str, str] = ("curl-x", "curl-y", "curl-z"),
    grad_order: int = 2,
) -> field_types.VectorField:
    field_types.ensure_vfield(vfield)
    field_types.ensure_uniform_domain(domain_details)
    field_types.ensure_domain_matches_vfield(
        domain_details=domain_details,
        vfield=vfield,
    )
    varray = vfield.data
    field_types.ensure_varray(varray)
    nabla = finite_difference.get_grad_func(grad_order)
    cell_width_x, cell_width_y, cell_width_z = domain_details.cell_widths
    curl_array = numpy.empty(varray.shape, dtype=varray.dtype)
    curl_array[0] = nabla(varray[2], cell_width_y, grad_axis=1) - nabla(varray[1], cell_width_z, grad_axis=2)
    curl_array[1] = nabla(varray[0], cell_width_z, grad_axis=2) - nabla(varray[2], cell_width_x, grad_axis=0)
    curl_array[2] = nabla(varray[1], cell_width_x, grad_axis=0) - nabla(varray[0], cell_width_y, grad_axis=1)
    return field_types.VectorField(
        sim_time=vfield.sim_time,
        data=curl_array,
        labels=labels,
    )


def compute_sfield_gradient(
    sfield: field_types.ScalarField,
    domain_details: field_types.UniformDomain,
    labels: tuple[str, str, str] = ("df/dx", "df/dy", "df/dz"),
    grad_order: int = 2,
) -> field_types.VectorField:
    field_types.ensure_sfield(sfield)
    field_types.ensure_uniform_domain(domain_details)
    field_types.ensure_domain_matches_sfield(
        domain_details=domain_details,
        sfield=sfield,
    )
    sarray = sfield.data
    field_types.ensure_sarray(sarray)
    nabla = finite_difference.get_grad_func(grad_order)
    cell_width_x, cell_width_y, cell_width_z = domain_details.cell_widths
    grad_array = numpy.empty((3, *sarray.shape), dtype=sarray.dtype)
    grad_array[0] = nabla(sarray, cell_width_x, grad_axis=0)
    grad_array[1] = nabla(sarray, cell_width_y, grad_axis=1)
    grad_array[2] = nabla(sarray, cell_width_z, grad_axis=2)
    return field_types.VectorField(
        sim_time=sfield.sim_time,
        data=grad_array,
        labels=labels,
    )


def compute_vfield_divergence(
    vfield: field_types.VectorField,
    domain_details: field_types.UniformDomain,
    label: str = "div(f)",
    grad_order: int = 2,
) -> field_types.ScalarField:
    field_types.ensure_vfield(vfield)
    field_types.ensure_uniform_domain(domain_details)
    field_types.ensure_domain_matches_vfield(
        domain_details=domain_details,
        vfield=vfield,
    )
    varray = vfield.data
    field_types.ensure_varray(varray)
    nabla = finite_difference.get_grad_func(grad_order)
    cell_width_x, cell_width_y, cell_width_z = domain_details.cell_widths
    dfx_dx = nabla(varray[0], cell_width_x, grad_axis=0)
    dfy_dy = nabla(varray[1], cell_width_y, grad_axis=1)
    dfz_dz = nabla(varray[2], cell_width_z, grad_axis=2)
    div_array = dfx_dx + dfy_dy + dfz_dz
    return field_types.ScalarField(
        sim_time=vfield.sim_time,
        data=div_array,
        label=label,
    )


##
## === COMPUTING FIELD QUANTITIES
##


def compute_magnetic_energy_density(
    vfield: field_types.VectorField,
    energy_prefactor: float = 0.5,
    label: str = "Emag",
) -> field_types.ScalarField:
    field_types.ensure_vfield(vfield)
    varray = vfield.data
    field_types.ensure_varray(varray)
    Emag_array = _sum_of_component_squares(varray)  # allocates output (reused below)
    Emag_array *= numpy.asarray(energy_prefactor, dtype=Emag_array.dtype)  # in-place transform
    return field_types.ScalarField(
        sim_time=vfield.sim_time,
        data=Emag_array,
        label=label,
    )


def compute_total_magnetic_energy(
    vfield: field_types.VectorField,
    domain_details: field_types.UniformDomain,
    energy_prefactor: float = 0.5,
) -> float:
    field_types.ensure_vfield(vfield)
    field_types.ensure_uniform_domain(domain_details)
    field_types.ensure_domain_matches_vfield(
        domain_details=domain_details,
        vfield=vfield,
    )
    Emag_sfield = compute_magnetic_energy_density(
        vfield=vfield,
        energy_prefactor=energy_prefactor,
    )
    return compute_sfield_volume_integral(
        sfield=Emag_sfield,
        domain_details=domain_details,
    )


## } MODULE
