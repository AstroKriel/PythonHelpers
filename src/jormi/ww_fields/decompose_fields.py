## { MODULE

##
## === DEPENDENCIES
##

import numpy
from jormi.utils import func_utils
from jormi.ww_fields import field_operators

##
## === FUNCTIONS
##


@func_utils.time_function
def compute_helmholtz_decomposition(
    vfield: numpy.ndarray,
    domain_lengths: tuple[float, float, float],
) -> tuple[numpy.ndarray, numpy.ndarray]:
    if vfield.shape[0] != 3:
        raise ValueError(
            "Input vector field must have shape: (3, n_cells_x, n_cells_y, n_cells_z)",
        )
    if len(domain_lengths) != 3:
        raise ValueError("Input domain size must have shape: (length_x, length_y, length_z)")
    vfield = numpy.asarray(vfield, dtype=numpy.float64)
    length_x, length_y, length_z = map(float, domain_lengths)
    n_cells_x, n_cells_y, n_cells_z = vfield.shape[1:]
    cell_width_x = length_x / n_cells_x
    cell_width_y = length_y / n_cells_y
    cell_width_z = length_z / n_cells_z
    kx_values = 2 * numpy.pi * numpy.fft.fftfreq(n_cells_x, d=cell_width_x)
    ky_values = 2 * numpy.pi * numpy.fft.fftfreq(n_cells_y, d=cell_width_y)
    kz_values = 2 * numpy.pi * numpy.fft.fftfreq(n_cells_z, d=cell_width_z)
    grid_kx, grid_ky, grid_kz = numpy.meshgrid(kx_values, ky_values, kz_values, indexing="ij")
    grid_k_magn = grid_kx**2 + grid_ky**2 + grid_kz**2
    ## avoid division by zero
    ## note: numpy.fft.fftn will assume the zero frequency is at index 0
    grid_k_magn[0, 0, 0] = 1
    vfield_fft_q = numpy.fft.fftn(vfield, axes=(1, 2, 3), norm="forward")
    ## \vec{k} cdot \vec{F}(\vec{k})
    sfield_k_dot_fft_q = grid_kx * vfield_fft_q[0] + grid_ky * vfield_fft_q[1] + grid_kz * vfield_fft_q[2]
    ## divergence (curl-free) component: (\vec{k} / k^2) (\vec{k} \cdot \vec{F}(\vec{k}))
    vfield_fft_div = numpy.stack(
        [
            (grid_kx / grid_k_magn) * sfield_k_dot_fft_q,
            (grid_ky / grid_k_magn) * sfield_k_dot_fft_q,
            (grid_kz / grid_k_magn) * sfield_k_dot_fft_q,
        ],
        axis=0,
    )
    ## solenoidal (divergence-free) component: \vec{F}(\vec{k}) - (\vec{k} / k^2) (\vec{k} \cdot \vec{F}(\vec{k}))
    vfield_fft_sol = vfield_fft_q - vfield_fft_div
    ## transform back to real space
    vfield_div = numpy.fft.ifftn(vfield_fft_div, axes=(1, 2, 3), norm="forward").real
    vfield_sol = numpy.fft.ifftn(vfield_fft_sol, axes=(1, 2, 3), norm="forward").real
    del kx_values, ky_values, kz_values, grid_kx, grid_ky, grid_kz, grid_k_magn
    del vfield_fft_q, sfield_k_dot_fft_q, vfield_fft_div, vfield_fft_sol
    return vfield_div, vfield_sol


@func_utils.time_function
def compute_tnb_terms(
    vfield: numpy.ndarray,
    domain_lengths: tuple[float, float, float],
    grad_order: int = 2,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    ## format: (vector-component, x, y, z)
    vfield = numpy.array(vfield)
    ##
    ## --- COMPUTE TANGENT BASIS
    ##
    ## (f_k f_k)^(1/2)
    sfield_magn = field_operators.compute_vfield_magnitude(vfield)
    ## f_i / (f_k f_k)^(1/2)
    vbasis_tangent = numpy.divide(
        vfield,
        sfield_magn,
        out=numpy.zeros_like(vfield),
        where=(sfield_magn > 0),
    )
    ##
    ## --- COMPUTE NORMAL BASIS
    ##
    ## df_j/dx_i: (component-j, gradient-direction-i, x, y, z)
    r2tensor_grad_b = field_operators.compute_vfield_gradient(
        vfield=vfield,
        domain_lengths=domain_lengths,
        grad_order=grad_order,
    )
    ## f_i df_j/dx_i
    vbasis_normal_term1 = numpy.einsum(
        "ixyz,jixyz->jxyz",
        vfield,
        r2tensor_grad_b,
        optimize=True,
    )
    ## f_i f_j f_m df_m/dx_i
    vbasis_normal_term2 = numpy.einsum(
        "ixyz,jxyz,mxyz,mixyz->jxyz",
        vfield,
        vfield,
        vfield,
        r2tensor_grad_b,
        optimize=True,
    )
    ## (f_i df_j/dx_i) / (f_k f_k) - (f_i f_j f_m df_m/dx_i) / (f_k f_k)^2
    vfield_kappa = vbasis_normal_term1 * sfield_magn**(-2) - vbasis_normal_term2 * sfield_magn**(-4)
    ## field curvature
    sfield_curvature = field_operators.compute_vfield_magnitude(vfield_kappa)
    ## normal basis
    vbasis_normal = numpy.divide(
        vfield_kappa,
        sfield_curvature,
        out=numpy.zeros_like(vfield_kappa),
        where=(sfield_curvature > 0),
    )
    ##
    ## --- COMPUTE BINORMAL BASIS
    ##
    ## by definition b-basis is orthogonal to both t- and n-basis
    vbasis_binormal = field_operators.compute_vfield_cross_product(vbasis_tangent, vbasis_normal)
    ## clean up temporary quantities
    del vbasis_normal_term1, vbasis_normal_term2, vfield_kappa
    return vbasis_tangent, vbasis_normal, vbasis_binormal, sfield_curvature


## } MODULE
