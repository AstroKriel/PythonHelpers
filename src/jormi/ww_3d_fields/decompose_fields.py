## { MODULE

##
## === DEPENDENCIES
##

import numpy

from dataclasses import dataclass

from jormi.ww_types import fdata_types, domain_types, field_types
from jormi.ww_fields import fdata_operators, field_operators

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class HelmholtzDecomposition:
    div_vfield: field_types.VectorField
    sol_vfield: field_types.VectorField
    bulk_vfield: field_types.VectorField

    def __post_init__(
        self,
    ):
        field_types.ensure_vfield(
            vfield=self.div_vfield,
            param_name="<div_vfield>",
        )
        field_types.ensure_vfield(
            vfield=self.sol_vfield,
            param_name="<sol_vfield>",
        )
        field_types.ensure_vfield(
            vfield=self.bulk_vfield,
            param_name="<bulk_vfield>",
        )
        field_types.ensure_same_field_shape(
            field_a=self.div_vfield,
            field_b=self.sol_vfield,
            field_name_a="<div_vfield>",
            field_name_b="<sol_vfield>",
        )
        field_types.ensure_same_field_shape(
            field_a=self.div_vfield,
            field_b=self.bulk_vfield,
            field_name_a="<div_vfield>",
            field_name_b="<bulk_vfield>",
        )
        if any([
                self.div_vfield.udomain != self.sol_vfield.udomain,
                self.div_vfield.udomain != self.bulk_vfield.udomain,
        ]):
            raise ValueError("HelmholtzDecomposition fields must share the same UniformDomain.")


@dataclass(frozen=True)
class TNBTerms:
    tangent_uvfield: field_types.UnitVectorField
    normal_uvfield: field_types.UnitVectorField
    binormal_uvfield: field_types.UnitVectorField
    curvature_sfield: field_types.ScalarField

    def __post_init__(
        self,
    ):
        field_types.ensure_uvfield(
            uvfield=self.tangent_uvfield,
            param_name="<tangent_uvfield>",
        )
        field_types.ensure_uvfield(
            uvfield=self.normal_uvfield,
            param_name="<normal_uvfield>",
        )
        field_types.ensure_uvfield(
            uvfield=self.binormal_uvfield,
            param_name="<binormal_uvfield>",
        )
        field_types.ensure_sfield(
            sfield=self.curvature_sfield,
            param_name="<curvature_sfield>",
        )
        field_types.ensure_same_field_shape(
            field_a=self.tangent_uvfield,
            field_b=self.normal_uvfield,
            field_name_a="<tangent_uvfield>",
            field_name_b="<normal_uvfield>",
        )
        field_types.ensure_same_field_shape(
            field_a=self.tangent_uvfield,
            field_b=self.binormal_uvfield,
            field_name_a="<tangent_uvfield>",
            field_name_b="<binormal_uvfield>",
        )
        if any([
                self.tangent_uvfield.udomain != self.normal_uvfield.udomain,
                self.tangent_uvfield.udomain != self.binormal_uvfield.udomain,
                self.tangent_uvfield.udomain != self.curvature_sfield.udomain,
        ]):
            raise ValueError("TNBTerms fields must share the same UniformDomain.")


##
## === FUNCTIONS
##


# @fn_utils.time_fn
def compute_helmholtz_decomposition(
    vfield: field_types.VectorField,
    udomain: domain_types.UniformDomain,
) -> HelmholtzDecomposition:
    """
    Compute the Helmholtz decomposition of a three-dimensional vector field into
    its divergence-free (solenoidal), curl-free (irrotational), and bulk (k=0) components.
    """
    field_types.ensure_vfield(vfield)
    domain_types.ensure_udomain(
        udomain=udomain,
        param_name="<udomain>",
    )
    field_types.ensure_udomain_matches_vfield(
        vfield=vfield,
        udomain=udomain,
    )
    if not all(udomain.periodicity):
        raise ValueError("Helmholtz (FFT) assumes periodic BCs in all directions.")
    sim_time = vfield.sim_time
    dtype = vfield.fdata.farray.dtype
    ## --- Build Fourier wavenumbers on the uniform grid
    num_cells_x, num_cells_y, num_cells_z = udomain.resolution
    cell_width_x, cell_width_y, cell_width_z = udomain.cell_widths
    kx_values = 2.0 * numpy.pi * numpy.fft.fftfreq(num_cells_x, d=cell_width_x)
    ky_values = 2.0 * numpy.pi * numpy.fft.fftfreq(num_cells_y, d=cell_width_y)
    kz_values = 2.0 * numpy.pi * numpy.fft.fftfreq(num_cells_z, d=cell_width_z)
    kx_grid, ky_grid, kz_grid = numpy.meshgrid(kx_values, ky_values, kz_values, indexing="ij")
    k_magn_grid = kx_grid**2 + ky_grid**2 + kz_grid**2
    ## avoid division by zero at k=0 (we zero out k=0 in the fft_varray anyway)
    k_magn_grid[0, 0, 0] = 1.0
    ## --- Transform into Fourier space
    ## with norm="forward", the k=0 coefficient is the spatial mean of the field
    fft_varray = numpy.fft.fftn(
        vfield.fdata.farray,
        axes=(1, 2, 3),
        norm="forward",
    )
    ## --- Compute bulk term
    ## only keep the k=0 coefficient (spatial mean; constant in space)
    bulk_fft_varray = numpy.zeros_like(fft_varray)
    bulk_fft_varray[:, 0, 0, 0] = fft_varray[:, 0, 0, 0]
    ## remove the k=0 coefficient from the working spectrum so projections do not steal the mean
    fft_varray[:, 0, 0, 0] = 0.0
    ## --- Compute projections in Fourier space
    ## with fft_varray[i] = F_i(k) and k = (kx, ky, kz)
    ## the divergence (curl-free) part is: div_fft_varray[i] = (k_i / k^2) * (k_j * F_j(k))
    ## the solenoidal (div-free) part is: sol_fft_varray[i] = F_i(k) - div_fft_varray[i]
    k_dot_fft_sfield = (kx_grid * fft_varray[0] + ky_grid * fft_varray[1] + kz_grid * fft_varray[2])
    with numpy.errstate(divide="ignore", invalid="ignore"):
        div_fft_varray = numpy.stack(
            [
                (kx_grid / k_magn_grid) * k_dot_fft_sfield,
                (ky_grid / k_magn_grid) * k_dot_fft_sfield,
                (kz_grid / k_magn_grid) * k_dot_fft_sfield,
            ],
            axis=0,
        )
    ## solenoidal (divergence-free) component in Fourier space
    sol_fft_varray = fft_varray - div_fft_varray
    ## transform back to real space
    div_varray = numpy.fft.ifftn(
        div_fft_varray,
        axes=(1, 2, 3),
        norm="forward",
    ).real.astype(
        dtype,
        copy=False,
    )
    sol_varray = numpy.fft.ifftn(
        sol_fft_varray,
        axes=(1, 2, 3),
        norm="forward",
    ).real.astype(
        dtype,
        copy=False,
    )
    bulk_varray = numpy.fft.ifftn(
        bulk_fft_varray,
        axes=(1, 2, 3),
        norm="forward",
    ).real.astype(
        dtype,
        copy=False,
    )
    ## free-up large temporary quantities before constructing new fields
    del (
        kx_values,
        ky_values,
        kz_values,
        kx_grid,
        ky_grid,
        kz_grid,
        k_magn_grid,
        fft_varray,
        k_dot_fft_sfield,
        div_fft_varray,
        sol_fft_varray,
        bulk_fft_varray,
    )
    ## package-up fields
    div_vdata = fdata_types.VectorFieldData(
        farray=div_varray,
        param_name="<helmholtz.div_vfield.fdata>",
    )
    sol_vdata = fdata_types.VectorFieldData(
        farray=sol_varray,
        param_name="<helmholtz.sol_vfield.fdata>",
    )
    bulk_vdata = fdata_types.VectorFieldData(
        farray=bulk_varray,
        param_name="<helmholtz.bulk_vfield.fdata>",
    )
    div_vfield = field_types.VectorField(
        fdata=div_vdata,
        udomain=udomain,
        field_label=r"$\vec{f}_\parallel$",
        sim_time=sim_time,
    )
    sol_vfield = field_types.VectorField(
        fdata=sol_vdata,
        udomain=udomain,
        field_label=r"$\vec{f}_\perp$",
        sim_time=sim_time,
    )
    bulk_vfield = field_types.VectorField(
        fdata=bulk_vdata,
        udomain=udomain,
        field_label=r"$\vec{f}_\mathrm{bulk}$",
        sim_time=sim_time,
    )
    return HelmholtzDecomposition(
        div_vfield=div_vfield,
        sol_vfield=sol_vfield,
        bulk_vfield=bulk_vfield,
    )


# @fn_utils.time_fn
def compute_tnb_terms(
    vfield: field_types.VectorField,
    udomain: domain_types.UniformDomain,
    grad_order: int = 2,
) -> TNBTerms:
    """
    Compute the Frenet-Serret-like tangent (T), normal (N), and binormal (B) bases
    for a three-dimensional vector field on a uniform grid.
    """
    field_types.ensure_vfield(vfield)
    domain_types.ensure_udomain(
        udomain=udomain,
        param_name="<udomain>",
    )
    field_types.ensure_udomain_matches_vfield(
        vfield=vfield,
        udomain=udomain,
    )
    sim_time = vfield.sim_time
    varray = vfield.fdata.farray
    ## --- COMPUTE TANGENT BASIS
    ## field magnitude: |f| = (f_k f_k)^(1/2)
    f_magn_sfield = field_operators.compute_vfield_magnitude(vfield=vfield)
    f_magn_sarray = f_magn_sfield.fdata.farray
    ## T_i = f_i / |f|
    tangent_uvarray = numpy.zeros_like(varray)
    numpy.divide(
        varray,
        f_magn_sarray,
        out=tangent_uvarray,
        where=(f_magn_sarray > 0),  # guard from zero magnitude
    )
    tangent_uvdata = fdata_types.VectorFieldData(
        farray=tangent_uvarray,
        param_name="<tnb.tangent.fdata>",
    )
    tangent_uvfield = field_types.UnitVectorField(
        fdata=tangent_uvdata,
        udomain=udomain,
        field_label=r"$\hat{t}$",
        sim_time=sim_time,
    )
    ## --- COMPUTE NORMAL BASIS
    ## gradient tensor: d_i f_j with layout (j, i, x, y, z)
    grad_r2tarray = fdata_operators.compute_varray_grad(
        vdata=vfield.fdata,
        cell_widths=udomain.cell_widths,
        grad_order=grad_order,
    )
    ## term1_j = f_i * (d_i f_j) = (f dot grad) f
    normal_term1_varray = numpy.einsum(
        "ixyz,jixyz->jxyz",
        varray,
        grad_r2tarray,
        optimize=True,
    )
    ## term2_j = f_i * f_j * f_m * (d_i f_m)
    normal_term2_varray = numpy.einsum(
        "ixyz,jxyz,mxyz,mixyz->jxyz",
        varray,
        varray,
        varray,
        grad_r2tarray,
        optimize=True,
    )
    ## curvature vector: kappa_j = term1_j / |f|^2 - term2_j / |f|^4
    inv_magn2_sarray = numpy.zeros_like(f_magn_sarray)
    numpy.divide(
        1.0,
        f_magn_sarray,
        out=inv_magn2_sarray,
        where=(f_magn_sarray > 0),
    )
    inv_magn2_sarray **= 2  # 1/|f|^2
    inv_magn4_sarray = inv_magn2_sarray**2  # 1/|f|^4
    kappa_varray = normal_term1_varray * inv_magn2_sarray - normal_term2_varray * inv_magn4_sarray
    curvature_sarray = fdata_operators.sum_of_squared_components(
        vdata=kappa_varray,
    )
    numpy.sqrt(curvature_sarray, out=curvature_sarray)
    curvature_sdata = fdata_types.ScalarFieldData(
        farray=curvature_sarray,
        param_name="<tnb.curvature.fdata>",
    )
    curvature_sfield = field_types.ScalarField(
        fdata=curvature_sdata,
        udomain=udomain,
        field_label=r"$|\vec{\kappa}|$",
        sim_time=sim_time,
    )
    ## N_i = kappa_i / |kappa|
    normal_uvarray = numpy.zeros_like(kappa_varray)
    numpy.divide(
        kappa_varray,
        curvature_sarray,
        out=normal_uvarray,
        where=(curvature_sarray > 0.0),  # guard against zero curvature
    )
    normal_uvdata = fdata_types.VectorFieldData(
        farray=normal_uvarray,
        param_name="<tnb.normal.fdata>",
    )
    normal_uvfield = field_types.UnitVectorField(
        fdata=normal_uvdata,
        udomain=udomain,
        field_label=r"$\hat{n}$",
        sim_time=sim_time,
    )
    ## --- COMPUTE BINORMAL BASIS
    ## B = T x N  (orthogonal to both T and N)
    binormal_vfield = field_operators.compute_vfield_cross_product(
        vfield_a=tangent_uvfield,
        vfield_b=normal_uvfield,
        field_label=r"$\hat{b}$",
    )
    binormal_uvfield = field_types.as_uvfield(
        vfield=binormal_vfield,
    )
    del normal_term1_varray, normal_term2_varray, inv_magn2_sarray, inv_magn4_sarray
    return TNBTerms(
        tangent_uvfield=tangent_uvfield,
        normal_uvfield=normal_uvfield,
        binormal_uvfield=binormal_uvfield,
        curvature_sfield=curvature_sfield,
    )


## } MODULE
