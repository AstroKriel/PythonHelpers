## { MODULE

##
## === DEPENDENCIES
##

import numpy
from jormi.ww_data import finite_difference
from jormi.ww_fields import field_types

##
## === VALIDATION HELPERS
##


def _ensure_sarray_shape(
    sarray: numpy.ndarray,
) -> None:
    if sarray.ndim != 3:
        raise ValueError("Scalar field arrays must have shape (num_cells_x, num_cells_y, num_cells_z).")


def _ensure_varray_shape(
    varray: numpy.ndarray,
) -> None:
    if (varray.ndim != 4) or (varray.shape[0] != 3):
        raise ValueError("Vector field arrays must have shape (3, num_cells_x, num_cells_y, num_cells_z).")


def _ensure_sfield_type(
    sfield,
) -> None:
    if not isinstance(sfield, field_types.ScalarField):
        raise TypeError(f"Expected ScalarField, got {type(sfield).__name__}")


def _ensure_vfield_type(
    vfield,
) -> None:
    if not isinstance(vfield, field_types.VectorField):
        raise TypeError(f"Expected VectorField, got {type(vfield).__name__}")


def _ensure_uniform_domain_type(
    domain,
) -> None:
    if not isinstance(domain, field_types.UniformDomain):
        raise TypeError(f"Expected UniformDomain, got {type(domain).__name__}")


def _ensure_same_grid_size(
    array_a: numpy.ndarray,
    array_b: numpy.ndarray,
) -> None:
    if array_a.shape != array_b.shape:
        raise ValueError(f"Grid mismatch: {array_a.shape} vs {array_b.shape}")


def _ensure_domain_matches_sfield(
    domain: field_types.UniformDomain,
    sfield: field_types.ScalarField,
) -> None:
    if domain.resolution != sfield.data.shape:
        raise ValueError(
            f"Domain resolution {domain.resolution} does not match scalar grid {sfield.data.shape}",
        )


def _ensure_domain_matches_vfield(
    domain: field_types.UniformDomain,
    vfield: field_types.VectorField,
) -> None:
    if domain.resolution != vfield.data.shape[1:]:
        raise ValueError(
            f"Domain resolution {domain.resolution} does not match vector grid {vfield.data.shape[1:]}",
        )


##
## === WORKSPACE UTILITIES
##


def _ensure_array_properties(
    array_shape: tuple[int, ...],
    dtype: numpy.dtype = field_types.DEFAULT_FLOAT_TYPE,
    array: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """
    Return a scratch array with the requested shape/dtype, reusing the provided array
    if compatible, otherwise allocates a new array.
    """
    dtype = numpy.dtype(dtype)  # use numpy types
    if (array is None) or (array.shape != array_shape) or (array.dtype != dtype):
        return numpy.empty(array_shape, dtype=dtype)
    return array


def _get_grad_func(
    grad_order: int,
):
    valid_grad_orders = {
        2: finite_difference.second_order_centered_difference,
        4: finite_difference.fourth_order_centered_difference,
        6: finite_difference.sixth_order_centered_difference,
    }
    if grad_order not in valid_grad_orders:
        raise ValueError(f"Gradient order `{grad_order}` is invalid.")
    return valid_grad_orders[grad_order]


##
## === OPTIMISED OPERATORS WORKING ON ARRAYS
##


def _sum_of_component_squares(
    varray: numpy.ndarray,
    out_array: numpy.ndarray | None = None,
    tmp_array: numpy.ndarray | None = None,
) -> numpy.ndarray:
    _ensure_varray_shape(varray)
    domain_shape = varray.shape[1:]
    out_array = _ensure_array_properties(
        array_shape=domain_shape,
        dtype=field_types.DEFAULT_FLOAT_TYPE,
        array=out_array,
    )
    tmp_array = _ensure_array_properties(
        array_shape=domain_shape,
        dtype=field_types.DEFAULT_FLOAT_TYPE,
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
    _ensure_varray_shape(varray_a)
    _ensure_varray_shape(varray_b)
    _ensure_same_grid_size(
        array_a=varray_a,
        array_b=varray_b,
    )
    domain_shape = varray_a.shape[1:]
    out_array = _ensure_array_properties(
        array_shape=domain_shape,
        dtype=field_types.DEFAULT_FLOAT_TYPE,
        array=out_array,
    )
    tmp_array = _ensure_array_properties(
        array_shape=domain_shape,
        dtype=field_types.DEFAULT_FLOAT_TYPE,
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
    sarray = numpy.asarray(sarray, dtype=field_types.DEFAULT_FLOAT_TYPE)
    _ensure_sarray_shape(sarray)
    return float(numpy.sqrt(numpy.mean(numpy.square(sarray))))


def compute_array_volume_integral(
    sarray: numpy.ndarray,
    cell_volume: float,
) -> float:
    sarray = numpy.asarray(sarray, dtype=field_types.DEFAULT_FLOAT_TYPE)
    _ensure_sarray_shape(sarray)
    return float(cell_volume * numpy.sum(sarray, dtype=field_types.DEFAULT_FLOAT_TYPE))


##
## === OPTIMISED OPERATORS WORKING ON FIELDS
##


def compute_sfield_rms(
    sfield: field_types.ScalarField,
) -> float:
    _ensure_sfield_type(sfield)
    return compute_array_rms(sfield.data)


def compute_sfield_volume_integral(
    sfield: field_types.ScalarField,
    domain: field_types.UniformDomain,
) -> float:
    _ensure_sfield_type(sfield)
    _ensure_uniform_domain_type(domain)
    _ensure_domain_matches_sfield(
        domain=domain,
        sfield=sfield,
    )
    return compute_array_volume_integral(
        sarray=sfield.data,
        cell_volume=domain.cell_volume,
    )


def compute_vfield_magnitude(
    vfield: field_types.VectorField,
    label: str = "|vec(f)|",
) -> field_types.ScalarField:
    _ensure_vfield_type(vfield)
    varray = numpy.asarray(vfield.data, dtype=field_types.DEFAULT_FLOAT_TYPE)
    _ensure_varray_shape(varray)
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
    _ensure_vfield_type(vfield_a)
    _ensure_vfield_type(vfield_b)
    varray_a = numpy.asarray(vfield_a.data, dtype=field_types.DEFAULT_FLOAT_TYPE)
    varray_b = numpy.asarray(vfield_b.data, dtype=field_types.DEFAULT_FLOAT_TYPE)
    _ensure_varray_shape(varray_a)
    _ensure_varray_shape(varray_b)
    _ensure_same_grid_size(
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
    _ensure_vfield_type(vfield_a)
    _ensure_vfield_type(vfield_b)
    varray_a = numpy.asarray(vfield_a.data, dtype=field_types.DEFAULT_FLOAT_TYPE)
    varray_b = numpy.asarray(vfield_b.data, dtype=field_types.DEFAULT_FLOAT_TYPE)
    _ensure_varray_shape(varray_a)
    _ensure_varray_shape(varray_b)
    _ensure_same_grid_size(
        array_a=varray_a,
        array_b=varray_b,
    )
    cross_array = numpy.empty(varray_a.shape, dtype=field_types.DEFAULT_FLOAT_TYPE)
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
    domain: field_types.UniformDomain,
    labels: tuple[str, str, str] = ("curl-x", "curl-y", "curl-z"),
    grad_order: int = 2,
) -> field_types.VectorField:
    _ensure_vfield_type(vfield)
    _ensure_uniform_domain_type(domain)
    _ensure_domain_matches_vfield(
        domain=domain,
        vfield=vfield,
    )
    varray = numpy.asarray(vfield.data, dtype=field_types.DEFAULT_FLOAT_TYPE)
    _ensure_varray_shape(varray)
    nabla = _get_grad_func(grad_order)
    cell_width_x, cell_width_y, cell_width_z = domain.cell_widths
    curl_array = numpy.empty(varray.shape, dtype=field_types.DEFAULT_FLOAT_TYPE)
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
    domain: field_types.UniformDomain,
    labels: tuple[str, str, str] = ("df/dx", "df/dy", "df/dz"),
    grad_order: int = 2,
) -> field_types.VectorField:
    _ensure_sfield_type(sfield)
    _ensure_uniform_domain_type(domain)
    _ensure_domain_matches_sfield(
        domain=domain,
        sfield=sfield,
    )
    sarray = numpy.asarray(sfield.data, dtype=field_types.DEFAULT_FLOAT_TYPE)
    _ensure_sarray_shape(sarray)
    nabla = _get_grad_func(grad_order)
    cell_width_x, cell_width_y, cell_width_z = domain.cell_widths
    grad_array = numpy.empty((3, *sarray.shape), dtype=field_types.DEFAULT_FLOAT_TYPE)
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
    domain: field_types.UniformDomain,
    label: str = "div(f)",
    grad_order: int = 2,
) -> field_types.ScalarField:
    _ensure_vfield_type(vfield)
    _ensure_uniform_domain_type(domain)
    _ensure_domain_matches_vfield(
        domain=domain,
        vfield=vfield
    )
    varray = numpy.asarray(vfield.data, dtype=field_types.DEFAULT_FLOAT_TYPE)
    _ensure_varray_shape(varray)
    nabla = _get_grad_func(grad_order)
    cell_width_x, cell_width_y, cell_width_z = domain.cell_widths
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
    _ensure_vfield_type(vfield)
    varray = numpy.asarray(vfield.data, dtype=field_types.DEFAULT_FLOAT_TYPE)
    _ensure_varray_shape(varray)
    Emag_array = _sum_of_component_squares(varray)  # allocates output (reused below)
    Emag_array *= energy_prefactor  # in-place transform
    return field_types.ScalarField(
        sim_time=vfield.sim_time,
        data=Emag_array,
        label=label,
    )


def compute_total_magnetic_energy(
    vfield: field_types.VectorField,
    domain: field_types.UniformDomain,
    energy_prefactor: float = 0.5,
) -> float:
    _ensure_vfield_type(vfield)
    _ensure_uniform_domain_type(domain)
    _ensure_domain_matches_vfield(
        domain=domain,
        vfield=vfield,
    )
    Emag_sfield = compute_magnetic_energy_density(
        vfield=vfield,
        energy_prefactor=energy_prefactor,
    )
    return compute_sfield_volume_integral(
        sfield=Emag_sfield,
        domain=domain,
    )


## } MODULE
