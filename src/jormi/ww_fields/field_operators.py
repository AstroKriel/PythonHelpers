## { MODULE

##
## === DEPENDENCIES
##

import numpy
from jormi.ww_data import finite_difference
from jormi.ww_fields import field_types

##
## === HELPERS
##


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


def _ensure_vfield_array_shape(
    varray: numpy.ndarray,
) -> None:
    if (varray.ndim != 4) or (varray.shape[0] != 3):
        raise ValueError("Vector field arrays must have shape (3, num_cells_x, num_cells_y, num_cells_z).")


def _validate_sfield(
    sfield,
) -> None:
    if not isinstance(sfield, field_types.ScalarField):
        raise TypeError(f"Expected ScalarField, got {type(sfield).__name__}")


def _validate_vfield(
    vfield,
) -> None:
    if not isinstance(vfield, field_types.VectorField):
        raise TypeError(f"Expected VectorField, got {type(vfield).__name__}")


def _validate_domain(
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
## === OPERATORS
##


def compute_array_rms(
    sarray: numpy.ndarray,
) -> float:
    return float(numpy.sqrt(numpy.mean(numpy.square(sarray))))


def compute_sfield_rms(
    sfield: field_types.ScalarField,
) -> float:
    _validate_sfield(sfield)
    sarray = numpy.asarray(sfield.data, dtype=numpy.float64)
    return compute_array_rms(sarray)


def compute_vfield_magnitude(
    vfield: field_types.VectorField,
    label: str = "|vec(f)|",
) -> field_types.ScalarField:
    _validate_vfield(vfield)
    varray = numpy.asarray(vfield.data, dtype=numpy.float64)
    _ensure_vfield_array_shape(varray)
    field_magn = numpy.sqrt(numpy.sum(varray * varray, axis=0))
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
    _validate_vfield(vfield_a)
    _validate_vfield(vfield_b)
    varray_a = numpy.asarray(vfield_a.data, dtype=numpy.float64)
    varray_b = numpy.asarray(vfield_b.data, dtype=numpy.float64)
    _ensure_vfield_array_shape(varray_a)
    _ensure_same_grid_size(varray_a, varray_b)
    dot_array = numpy.einsum("ixyz,ixyz->xyz", varray_a, varray_b, optimize=True)
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
    _validate_vfield(vfield_a)
    _validate_vfield(vfield_b)
    varray_a = numpy.asarray(vfield_a.data, dtype=numpy.float64)
    varray_b = numpy.asarray(vfield_b.data, dtype=numpy.float64)
    _ensure_vfield_array_shape(varray_a)
    _ensure_same_grid_size(varray_a, varray_b)
    cross_array = numpy.stack(
        [
            varray_a[1] * varray_b[2] - varray_a[2] * varray_b[1],
            varray_a[2] * varray_b[0] - varray_a[0] * varray_b[2],
            varray_a[0] * varray_b[1] - varray_a[1] * varray_b[0],
        ],
        axis=0,
    )
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
    _validate_vfield(vfield)
    _validate_domain(domain)
    _ensure_domain_matches_vfield(domain=domain, vfield=vfield)
    varray = numpy.asarray(vfield.data, dtype=numpy.float64)
    _ensure_vfield_array_shape(varray)
    grad_func = _get_grad_func(grad_order)
    cell_width_x, cell_width_y, cell_width_z = domain.cell_widths
    curl_array = numpy.stack(
        [
            grad_func(varray[2], cell_width_y, grad_axis=1) -
            grad_func(varray[1], cell_width_z, grad_axis=2),
            grad_func(varray[0], cell_width_z, grad_axis=2) -
            grad_func(varray[2], cell_width_x, grad_axis=0),
            grad_func(varray[1], cell_width_x, grad_axis=0) -
            grad_func(varray[0], cell_width_y, grad_axis=1),
        ],
        axis=0,
    )
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
    _validate_sfield(sfield)
    _validate_domain(domain)
    _ensure_domain_matches_sfield(domain=domain, sfield=sfield)
    sarray = numpy.asarray(sfield.data, dtype=numpy.float64)
    if sarray.ndim != 3:
        raise ValueError("`sfield.data` must have shape (num_cells_x, num_cells_y, num_cells_z).")
    grad_func = _get_grad_func(grad_order)
    cell_width_x, cell_width_y, cell_width_z = domain.cell_widths
    grad_array = numpy.array(
        [
            grad_func(sarray, cell_width_x, grad_axis=0),
            grad_func(sarray, cell_width_y, grad_axis=1),
            grad_func(sarray, cell_width_z, grad_axis=2),
        ],
    )
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
    _validate_vfield(vfield)
    _validate_domain(domain)
    _ensure_domain_matches_vfield(domain=domain, vfield=vfield)
    varray = numpy.asarray(vfield.data, dtype=numpy.float64)
    _ensure_vfield_array_shape(varray)
    grad_func = _get_grad_func(grad_order)
    cell_width_x, cell_width_y, cell_width_z = domain.cell_widths
    ## build df_i/dx_j stack (3, 3, nx, ny, nz)
    grad_stack = numpy.stack(
        [
            numpy.array(
                [
                    grad_func(varray[0], cell_width_x, grad_axis=0),
                    grad_func(varray[0], cell_width_y, grad_axis=1),
                    grad_func(varray[0], cell_width_z, grad_axis=2),
                ],
            ),
            numpy.array(
                [
                    grad_func(varray[1], cell_width_x, grad_axis=0),
                    grad_func(varray[1], cell_width_y, grad_axis=1),
                    grad_func(varray[1], cell_width_z, grad_axis=2),
                ],
            ),
            numpy.array(
                [
                    grad_func(varray[2], cell_width_x, grad_axis=0),
                    grad_func(varray[2], cell_width_y, grad_axis=1),
                    grad_func(varray[2], cell_width_z, grad_axis=2),
                ],
            ),
        ],
        axis=0,
    )
    div_array = numpy.einsum("iixyz->xyz", grad_stack, optimize=True)
    return field_types.ScalarField(
        sim_time=vfield.sim_time,
        data=div_array,
        label=label,
    )


def compute_magnetic_energy(
    vfield: field_types.VectorField,
    energy_prefactor: float = 0.5,
    label: str = "E_mag",
) -> field_types.ScalarField:
    _validate_vfield(vfield)
    varray = numpy.asarray(vfield.data, dtype=numpy.float64)
    _ensure_vfield_array_shape(varray)
    Emag_array = energy_prefactor * numpy.add.reduce(varray * varray, axis=0)
    return field_types.ScalarField(
        sim_time=vfield.sim_time,
        data=Emag_array,
        label=label,
    )


## } MODULE
