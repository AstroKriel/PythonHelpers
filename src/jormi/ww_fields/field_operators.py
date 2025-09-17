## { MODULE

##
## === DEPENDENCIES
##

import numpy
from jormi.ww_data import finite_difference

##
## === HELPERS
##


def _validate_domain_length(
    domain_lengths: tuple[float, float, float],
) -> tuple[float, float, float]:
    if not (isinstance(domain_lengths, (tuple, list)) and len(domain_lengths) == 3):
        raise ValueError(
            "`domain_lengths` must be a 3-tuple (domain_length_x, domain_length_y, domain_length_z).",
        )
    domain_length_x = float(domain_lengths[0])
    domain_length_y = float(domain_lengths[1])
    domain_length_z = float(domain_lengths[2])
    if not (domain_length_x > 0 and domain_length_y > 0 and domain_length_z > 0):
        raise ValueError("All entries of `domain_lengths` must be positive.")
    return (domain_length_x, domain_length_y, domain_length_z)


def _get_cell_widths(
    domain_lengths: tuple[float, float, float],
    n_cells_xyz: tuple[int, int, int],
) -> tuple[float, float, float]:
    domain_length_x, domain_length_y, domain_length_z = _validate_domain_length(domain_lengths)
    n_cells_x, n_cells_y, n_cells_z = n_cells_xyz
    return (
        domain_length_x / n_cells_x,
        domain_length_y / n_cells_y,
        domain_length_z / n_cells_z,
    )


def _get_grad_func(grad_order: int):
    implemented_grad_funcs = {
        2: finite_difference.second_order_centered_difference,
        4: finite_difference.fourth_order_centered_difference,
        6: finite_difference.sixth_order_centered_difference,
    }
    if grad_order not in implemented_grad_funcs:
        raise ValueError(f"Gradient order `{grad_order}` is invalid.")
    return implemented_grad_funcs[grad_order]


##
## === OPERATORS
##


def compute_sfield_rms(
    sfield: numpy.ndarray,
) -> float:
    sfield = numpy.asarray(sfield, dtype=numpy.float64)
    return float(numpy.sqrt(numpy.mean(numpy.square(sfield))))


def compute_vfield_dot_product(
    vfield_1: numpy.ndarray,
    vfield_2: numpy.ndarray,
) -> numpy.ndarray:
    vfield_1 = numpy.asarray(vfield_1, dtype=numpy.float64)
    vfield_2 = numpy.asarray(vfield_2, dtype=numpy.float64)
    if (vfield_1.shape != vfield_2.shape) or (vfield_1.ndim != 4) or (vfield_1.shape[0] != 3):
        raise ValueError("Both inputs must have shape (3, n_cells_x, n_cells_y, n_cells_z).")
    return numpy.einsum("ixyz,ixyz->xyz", vfield_1, vfield_2, optimize=True)


def compute_vfield_magnitude(
    vfield: numpy.ndarray,
) -> numpy.ndarray:
    vfield = numpy.asarray(vfield, dtype=numpy.float64)
    if (vfield.ndim != 4) or (vfield.shape[0] != 3):
        raise ValueError("`vfield` must have shape (3, n_cells_x, n_cells_y, n_cells_z).")
    return numpy.sqrt(numpy.sum(vfield * vfield, axis=0))


def compute_vfield_cross_product(
    vfield_1: numpy.ndarray,
    vfield_2: numpy.ndarray,
) -> numpy.ndarray:
    vfield_1 = numpy.asarray(vfield_1, dtype=numpy.float64)
    vfield_2 = numpy.asarray(vfield_2, dtype=numpy.float64)
    if (vfield_1.shape != vfield_2.shape) or (vfield_1.ndim != 4) or (vfield_1.shape[0] != 3):
        raise ValueError("Both inputs must have shape (3, n_cells_x, n_cells_y, n_cells_z).")
    return numpy.stack(
        [
            vfield_1[1] * vfield_2[2] - vfield_1[2] * vfield_2[1],
            vfield_1[2] * vfield_2[0] - vfield_1[0] * vfield_2[2],
            vfield_1[0] * vfield_2[1] - vfield_1[1] * vfield_2[0],
        ],
        axis=0,
    )


def compute_vfield_curl(
    vfield: numpy.ndarray,
    domain_lengths: tuple[float, float, float],
    grad_order: int = 2,
) -> numpy.ndarray:
    ## input format: (vector-component, x, y, z), assuming uniform grid
    ## output format: (curl-component, x, y, z)
    vfield = numpy.asarray(vfield, dtype=numpy.float64)
    if vfield.ndim != 4 or vfield.shape[0] != 3:
        raise ValueError("`vfield` must have shape (3, n_cells_x, n_cells_y, n_cells_z).")
    grad_func = _get_grad_func(grad_order)
    n_cells_x, n_cells_y, n_cells_z = vfield.shape[1:]
    cell_width_x, cell_width_y, cell_width_z = _get_cell_widths(
        domain_lengths,
        (n_cells_x, n_cells_y, n_cells_z),
    )
    # curl components
    return numpy.stack(
        [
            grad_func(vfield[2], cell_width_y, grad_axis=1) - grad_func(vfield[1], cell_width_z, grad_axis=2),
            grad_func(vfield[0], cell_width_z, grad_axis=2) - grad_func(vfield[2], cell_width_x, grad_axis=0),
            grad_func(vfield[1], cell_width_x, grad_axis=0) - grad_func(vfield[0], cell_width_y, grad_axis=1),
        ],
        axis=0,
    )


def compute_sfield_gradient(
    sfield: numpy.ndarray,
    domain_lengths: tuple[float, float, float],
    grad_order: int = 2,
) -> numpy.ndarray:
    ## input format: (x, y, z), assuming uniform grid
    ## output format: (gradient-direction, x, y, z)
    sfield = numpy.asarray(sfield, dtype=numpy.float64)
    if sfield.ndim != 3:
        raise ValueError("`sfield` must have shape (n_cells_x, n_cells_y, n_cells_z).")
    grad_func = _get_grad_func(grad_order)
    n_cells_x, n_cells_y, n_cells_z = sfield.shape
    cell_width_x, cell_width_y, cell_width_z = _get_cell_widths(
        domain_lengths,
        (n_cells_x, n_cells_y, n_cells_z),
    )
    return numpy.array(
        [
            grad_func(sfield, cell_width_x, grad_axis=0),
            grad_func(sfield, cell_width_y, grad_axis=1),
            grad_func(sfield, cell_width_z, grad_axis=2),
        ],
    )


def compute_vfield_gradient(
    vfield: numpy.ndarray,
    domain_lengths: tuple[float, float, float],
    grad_order: int = 2,
) -> numpy.ndarray:
    ## df_i/dx_j: (component-i, gradient-direction-j, x, y, z)
    vfield = numpy.asarray(vfield, dtype=numpy.float64)
    if vfield.ndim != 4 or vfield.shape[0] != 3:
        raise ValueError("`vfield` must have shape (3, n_cells_x, n_cells_y, n_cells_z).")
    return numpy.stack(
        [compute_sfield_gradient(vfield[dim_index], domain_lengths, grad_order) for dim_index in range(3)],
        axis=0,
    )


def compute_vfield_divergence(
    vfield: numpy.ndarray,
    domain_lengths: tuple[float, float, float],
    grad_order: int = 2,
) -> numpy.ndarray:
    r2tensor_grad_q = compute_vfield_gradient(vfield, domain_lengths, grad_order)
    return numpy.einsum("iixyz->xyz", r2tensor_grad_q, optimize=True)


## } MODULE
