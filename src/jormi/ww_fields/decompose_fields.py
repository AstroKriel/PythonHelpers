## { MODULE

##
## === DEPENDENCIES
##

import numpy
from dataclasses import dataclass
from jormi.utils import func_utils
from jormi.ww_data import finite_difference
from jormi.ww_fields import field_types, field_operators

##
## === FUNCTIONS
##


@dataclass(frozen=True)
class HelmholtzDecomposition:
    div_vfield: field_types.VectorField
    sol_vfield: field_types.VectorField


@func_utils.time_function
def compute_helmholtz_decomposition(
    vfield: field_types.VectorField,
    domain_details: field_types.UniformDomain,
    field_label: str = "f",
) -> HelmholtzDecomposition:
    """
    Compute the Helmholtz decomposition of a three-dimensional vector field into
    its divergence-free (solenoidal) and curl-free (irrotational) components.
    """
    field_types.ensure_vfield(vfield)
    field_types.ensure_uniform_domain(domain_details)
    field_types.ensure_domain_matches_vfield(domain_details, vfield)
    if not all(domain_details.periodicity):
        raise ValueError("Helmholtz (FFT) assumes periodic BCs in all directions.")
    dtype = vfield.data.dtype
    num_cells_x, num_cells_y, num_cells_z = domain_details.resolution
    cell_width_x, cell_width_y, cell_width_z = domain_details.cell_widths
    kx_values = 2.0 * numpy.pi * numpy.fft.fftfreq(num_cells_x, d=cell_width_x)
    ky_values = 2.0 * numpy.pi * numpy.fft.fftfreq(num_cells_y, d=cell_width_y)
    kz_values = 2.0 * numpy.pi * numpy.fft.fftfreq(num_cells_z, d=cell_width_z)
    grid_kx, grid_ky, grid_kz = numpy.meshgrid(kx_values, ky_values, kz_values, indexing="ij")
    grid_k_magn = grid_kx**2 + grid_ky**2 + grid_kz**2
    ## avoid division by zero
    ## note: numpy.fft.fftn will assume the zero frequency is at index 0
    grid_k_magn[0, 0, 0] = 1.0
    q_fft_varray = numpy.fft.fftn(vfield.data, axes=(1, 2, 3), norm="forward")
    ## \vec{k} cdot \vec{F}(\vec{k})
    k_dot_fft_q_sfield = grid_kx * q_fft_varray[0] + grid_ky * q_fft_varray[1] + grid_kz * q_fft_varray[2]
    ## divergence (curl-free) component: (\vec{k} / k^2) (\vec{k} \cdot \vec{F}(\vec{k}))
    with numpy.errstate(divide="ignore", invalid="ignore"):
        div_fft_varray = numpy.stack(
            [
                (grid_kx / grid_k_magn) * k_dot_fft_q_sfield,
                (grid_ky / grid_k_magn) * k_dot_fft_q_sfield,
                (grid_kz / grid_k_magn) * k_dot_fft_q_sfield,
            ],
            axis=0,
        )
    ## solenoidal (divergence-free) component: \vec{F}(\vec{k}) - (\vec{k} / k^2) (\vec{k} \cdot \vec{F}(\vec{k}))
    sol_fft_varray = q_fft_varray - div_fft_varray
    ## transform back to real space
    div_array = numpy.fft.ifftn(
        div_fft_varray,
        axes=(1, 2, 3),
        norm="forward",
    ).real.astype(
        dtype,
        copy=False,
    )
    sol_array = numpy.fft.ifftn(
        sol_fft_varray,
        axes=(1, 2, 3),
        norm="forward",
    ).real.astype(
        dtype,
        copy=False,
    )
    div_labels = (
        r"$" + field_label + r"_{\parallel,x}$",
        r"$" + field_label + r"_{\parallel,y}$",
        r"$" + field_label + r"_{\parallel,z}$",
    )
    sol_labels = (
        r"$" + field_label + r"_{\perp,x}$",
        r"$" + field_label + r"_{\perp,y}$",
        r"$" + field_label + r"_{\perp,z}$",
    )
    div_vfield = field_types.VectorField(sim_time=vfield.sim_time, data=div_array, labels=div_labels)
    sol_vfield = field_types.VectorField(sim_time=vfield.sim_time, data=sol_array, labels=sol_labels)
    del kx_values, ky_values, kz_values, grid_kx, grid_ky, grid_kz, grid_k_magn
    del q_fft_varray, k_dot_fft_q_sfield, div_fft_varray, sol_fft_varray
    return HelmholtzDecomposition(
        div_vfield=div_vfield,
        sol_vfield=sol_vfield,
    )


@dataclass(frozen=True)
class TNBTerms:
    tangent_nbasis: field_types.VectorField
    normal_nbasis: field_types.VectorField
    binormal_nbasis: field_types.VectorField
    curvature_sfield: field_types.ScalarField


@func_utils.time_function
def compute_tnb_terms(
    vfield: field_types.VectorField,
    domain_details: field_types.UniformDomain,
    grad_order: int = 2,
    field_label: str | None = "f",
) -> TNBTerms:
    """
    Compute the Frenet-Serret-like tangent (T), normal (N), and binormal (B) bases
    for a three-dimensional vector field on a uniform grid.
    """
    field_types.ensure_vfield(vfield)
    field_types.ensure_uniform_domain(domain_details)
    field_types.ensure_domain_matches_vfield(domain_details, vfield)
    varray = vfield.data
    dtype = varray.dtype
    nabla = finite_difference.get_grad_func(grad_order)
    num_cells_x, num_cells_y, num_cells_z = domain_details.resolution
    cell_width_x, cell_width_y, cell_width_z = domain_details.cell_widths
    ## --- COMPUTE TANGENT BASIS
    ## field magnitude: |f| = (f_k f_k)^(1/2)
    field_magn_sarray = field_operators.compute_vfield_magnitude(vfield).data
    ## T_i = f_i / |f|
    tangent_narray = numpy.zeros_like(varray)
    numpy.divide(
        varray,
        field_magn_sarray,
        out=tangent_narray,
        where=(field_magn_sarray > 0),  # guard from zero magnitude
    )
    tangent_labels = (
        fr"${field_label}_T,x$" if field_label else f"{vfield.labels[0]}_T",
        fr"${field_label}_T,y$" if field_label else f"{vfield.labels[1]}_T",
        fr"${field_label}_T,z$" if field_label else f"{vfield.labels[2]}_T",
    )
    tangent_nbasis = field_types.VectorField(
        sim_time=vfield.sim_time,
        data=tangent_narray,
        labels=tangent_labels,
    )
    ## --- COMPUTE NORMAL BASIS
    ## gradient tensor: df_j/dx_i with layout (j, i, x, y, z)
    r2tensor_grad_array = numpy.empty((3, 3, num_cells_x, num_cells_y, num_cells_z), dtype=dtype)
    for comp_j in range(3):
        r2tensor_grad_array[comp_j, 0] = nabla(varray[comp_j], cell_width_x, grad_axis=0)  # df_j/dx
        r2tensor_grad_array[comp_j, 1] = nabla(varray[comp_j], cell_width_y, grad_axis=1)  # df_j/dy
        r2tensor_grad_array[comp_j, 2] = nabla(varray[comp_j], cell_width_z, grad_axis=2)  # df_j/dz
    ## term1_j = f_i * (df_j/dx_i) = (f dot grad) f
    normal_term1_narray = numpy.einsum(
        "ixyz,jixyz->jxyz",
        varray,
        r2tensor_grad_array,
        optimize=True,
    )
    ## term2_j = f_i * f_j * f_m * (df_m/dx_i)
    normal_term2_narray = numpy.einsum(
        "ixyz,jxyz,mxyz,mixyz->jxyz",
        varray,
        varray,
        varray,
        r2tensor_grad_array,
        optimize=True,
    )
    ## curvature vector: kappa_j = term1_j / |f|^2  -  term2_j / |f|^4
    inv_magn2_sarray = numpy.zeros_like(field_magn_sarray)
    numpy.divide(1.0, field_magn_sarray, out=inv_magn2_sarray, where=(field_magn_sarray > 0))
    inv_magn2_sarray **= 2  # 1/|f|^2
    inv_magn4_array = inv_magn2_sarray**2  # 1/|f|^4
    curvature_varray = normal_term1_narray * inv_magn2_sarray - normal_term2_narray * inv_magn4_array
    vfield_kappa = field_types.VectorField(
        sim_time=vfield.sim_time,
        data=curvature_varray,
        labels=(r"$\kappa_x$", r"$\kappa_y$", r"$\kappa_z$"),
    )
    curvature_sfield = field_operators.compute_vfield_magnitude(vfield_kappa, label="kappa")
    curvature_sarray = curvature_sfield.data
    ## N_i = kappa_i / |kappa|
    normal_narray = numpy.zeros_like(curvature_varray)
    numpy.divide(
        curvature_varray,
        curvature_sarray,
        out=normal_narray,
        where=(curvature_sarray > 0),  # guard zero curvature
    )
    normal_labels = (
        fr"${field_label}_N,x$" if field_label else f"{vfield.labels[0]}_N",
        fr"${field_label}_N,y$" if field_label else f"{vfield.labels[1]}_N",
        fr"${field_label}_N,z$" if field_label else f"{vfield.labels[2]}_N",
    )
    normal_nbasis = field_types.VectorField(
        sim_time=vfield.sim_time,
        data=normal_narray,
        labels=normal_labels,
    )
    ## --- COMPUTE BINORMAL BASIS
    ## B = T x N  (orthogonal to both T and N)
    binormal_nbasis = field_operators.compute_vfield_cross_product(
        vfield_a=tangent_nbasis,
        vfield_b=normal_nbasis,
        labels=(r"$\hat{e}_{b,1}$", r"$\hat{e}_{b,2}$", r"$\hat{e}_{b,3}$"),
    )
    del normal_term1_narray, normal_term2_narray, inv_magn2_sarray, inv_magn4_array
    return TNBTerms(
        tangent_nbasis=tangent_nbasis,
        normal_nbasis=normal_nbasis,
        binormal_nbasis=binormal_nbasis,
        curvature_sfield=curvature_sfield,
    )


## } MODULE
