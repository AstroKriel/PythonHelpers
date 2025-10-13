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
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class HelmholtzDecomposition:
    div_vfield: field_types.VectorField
    sol_vfield: field_types.VectorField

    def __post_init__(self):
        field_types.ensure_vfield(self.div_vfield)
        field_types.ensure_vfield(self.sol_vfield)


@dataclass(frozen=True)
class TNBTerms:
    tangent_uvfield: field_types.UnitVectorField
    normal_uvfield: field_types.UnitVectorField
    binormal_uvfield: field_types.UnitVectorField
    curvature_sfield: field_types.ScalarField

    def __post_init__(self):
        field_types.ensure_uvfield(self.tangent_uvfield)
        field_types.ensure_uvfield(self.normal_uvfield)
        field_types.ensure_uvfield(self.binormal_uvfield)
        field_types.ensure_sfield(self.curvature_sfield)


##
## === FUNCTIONS
##


@func_utils.time_function
def compute_helmholtz_decomposition(
    vfield: field_types.VectorField,
    domain_details: field_types.UniformDomain,
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
    kx_grid, ky_grid, kz_grid = numpy.meshgrid(kx_values, ky_values, kz_values, indexing="ij")
    k_magn_grid = kx_grid**2 + ky_grid**2 + kz_grid**2
    ## avoid division by zero
    ## note: numpy.fft.fftn will assume the zero frequency is at index 0
    k_magn_grid[0, 0, 0] = 1.0
    fft_varray = numpy.fft.fftn(vfield.data, axes=(1, 2, 3), norm="forward")
    ## \vec{k} cdot \vec{F}(\vec{k})
    k_dot_fft_sfield = kx_grid * fft_varray[0] + ky_grid * fft_varray[1] + kz_grid * fft_varray[2]
    ## divergence (curl-free) component: (\vec{k} / k^2) (\vec{k} \cdot \vec{F}(\vec{k}))
    with numpy.errstate(divide="ignore", invalid="ignore"):
        div_fft_varray = numpy.stack(
            [
                (kx_grid / k_magn_grid) * k_dot_fft_sfield,
                (ky_grid / k_magn_grid) * k_dot_fft_sfield,
                (kz_grid / k_magn_grid) * k_dot_fft_sfield,
            ],
            axis=0,
        )
    ## solenoidal (divergence-free) component: \vec{F}(\vec{k}) - (\vec{k} / k^2) (\vec{k} \cdot \vec{F}(\vec{k}))
    sol_fft_varray = fft_varray - div_fft_varray
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
    div_vfield = field_types.VectorField(
        sim_time=vfield.sim_time,
        data=div_array,
        labels=(
            r"$f_{\parallel,x}$",
            r"$f_{\parallel,y}$",
            r"$f_{\parallel,z}$",
        ),
    )
    sol_vfield = field_types.VectorField(
        sim_time=vfield.sim_time,
        data=sol_array,
        labels=(
            r"$f_{\perp,x}$",
            r"$f_{\perp,y}$",
            r"$f_{\perp,z}$",
        ),
    )
    del kx_values, ky_values, kz_values, kx_grid, ky_grid, kz_grid, k_magn_grid
    del fft_varray, k_dot_fft_sfield, div_fft_varray, sol_fft_varray
    return HelmholtzDecomposition(
        div_vfield=div_vfield,
        sol_vfield=sol_vfield,
    )


@func_utils.time_function
def compute_tnb_terms(
    vfield: field_types.VectorField,
    domain_details: field_types.UniformDomain,
    grad_order: int = 2,
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
    f_magn_sarray = field_operators.compute_vfield_magnitude(vfield).data
    ## T_i = f_i / |f|
    tangent_uvarray = numpy.zeros_like(varray)
    numpy.divide(
        varray,
        f_magn_sarray,
        out=tangent_uvarray,
        where=(f_magn_sarray > 0),  # guard from zero magnitude
    )
    tangent_uvfield = field_types.UnitVectorField(
        sim_time=vfield.sim_time,
        data=tangent_uvarray,
        labels=(
            r"$\hat{t}_x$",
            r"$\hat{t}_y$",
            r"$\hat{t}_z$",
        ),
    )
    ## --- COMPUTE NORMAL BASIS
    ## gradient tensor: df_j/dx_i with layout (j, i, x, y, z)
    grad_array = numpy.empty((3, 3, num_cells_x, num_cells_y, num_cells_z), dtype=dtype)
    for comp_j in range(3):
        grad_array[comp_j, 0] = nabla(varray[comp_j], cell_width_x, grad_axis=0)  # df_j/dx
        grad_array[comp_j, 1] = nabla(varray[comp_j], cell_width_y, grad_axis=1)  # df_j/dy
        grad_array[comp_j, 2] = nabla(varray[comp_j], cell_width_z, grad_axis=2)  # df_j/dz
    ## term1_j = f_i * (df_j/dx_i) = (f dot grad) f
    normal_term1_varray = numpy.einsum(
        "ixyz,jixyz->jxyz",
        varray,
        grad_array,
        optimize=True,
    )
    ## term2_j = f_i * f_j * f_m * (df_m/dx_i)
    normal_term2_varray = numpy.einsum(
        "ixyz,jxyz,mxyz,mixyz->jxyz",
        varray,
        varray,
        varray,
        grad_array,
        optimize=True,
    )
    ## curvature vector: kappa_j = term1_j / |f|^2  -  term2_j / |f|^4
    inv_magn2_sarray = numpy.zeros_like(f_magn_sarray)
    numpy.divide(1.0, f_magn_sarray, out=inv_magn2_sarray, where=(f_magn_sarray > 0))
    inv_magn2_sarray **= 2  # 1/|f|^2
    inv_magn4_array = inv_magn2_sarray**2  # 1/|f|^4
    kappa_varray = normal_term1_varray * inv_magn2_sarray - normal_term2_varray * inv_magn4_array
    kappa_vfield = field_types.VectorField(
        sim_time=vfield.sim_time,
        data=kappa_varray,
        labels=(
            r"$\kappa_x$",
            r"$\kappa_y$",
            r"$\kappa_z$",
        ),
    )
    curvature_sfield = field_operators.compute_vfield_magnitude(kappa_vfield, label="kappa")
    curvature_sarray = curvature_sfield.data
    ## N_i = kappa_i / |kappa|
    normal_uvarray = numpy.zeros_like(kappa_varray)
    numpy.divide(
        kappa_varray,
        curvature_sarray,
        out=normal_uvarray,
        where=(curvature_sarray > 0),  # guard zero curvature
    )
    normal_uvfield = field_types.UnitVectorField(
        sim_time=vfield.sim_time,
        data=normal_uvarray,
        labels=(
            r"$\hat{n}_x$",
            r"$\hat{n}_y$",
            r"$\hat{n}_z$",
        ),
    )
    ## --- COMPUTE BINORMAL BASIS
    ## B = T x N  (orthogonal to both T and N)
    binormal_vfield = field_operators.compute_vfield_cross_product(
        vfield_a=tangent_uvfield,
        vfield_b=normal_uvfield,
        labels=(
            r"$\hat{b}_x$",
            r"$\hat{b}_y$",
            r"$\hat{b}_z$",
        ),
    )
    binormal_uvfield = field_types.as_unit_vfield(vfield=binormal_vfield)
    del normal_term1_varray, normal_term2_varray, inv_magn2_sarray, inv_magn4_array
    return TNBTerms(
        tangent_uvfield=tangent_uvfield,
        normal_uvfield=normal_uvfield,
        binormal_uvfield=binormal_uvfield,
        curvature_sfield=curvature_sfield,
    )


## } MODULE
