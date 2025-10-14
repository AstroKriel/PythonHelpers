## { MODULE

##
## === DEPENDENCIES
##

import numpy
from dataclasses import dataclass
from jormi.utils import func_utils
from jormi.ww_fields import array_types, array_operators, finite_difference, field_types, decompose_fields

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class MagneticCurvatureTerms:
    curvature_sfield: field_types.ScalarField
    stretching_sfield: field_types.ScalarField
    compression_sfield: field_types.ScalarField

    def __post_init__(self):
        field_types.ensure_sfield(self.curvature_sfield)
        field_types.ensure_sfield(self.stretching_sfield)
        field_types.ensure_sfield(self.compression_sfield)


@dataclass(frozen=True)
class LorentzForceTerms:
    lorentz_vfield: field_types.VectorField
    tension_vfield: field_types.VectorField
    gradP_perp_vfield: field_types.VectorField

    def __post_init__(self):
        field_types.ensure_vfield(self.lorentz_vfield)
        field_types.ensure_vfield(self.tension_vfield)
        field_types.ensure_vfield(self.gradP_perp_vfield)


##
## === FUNCTIONS
##


@func_utils.time_function
def compute_magnetic_curvature_terms(
    u_vfield: field_types.VectorField,
    tangent_uvfield: field_types.UnitVectorField,
    normal_uvfield: field_types.UnitVectorField,
    uniform_domain: field_types.UniformDomain,
    out_grad_r2tarray: numpy.ndarray | None = None,
    grad_order: int = 2,
) -> MagneticCurvatureTerms:
    """
    Compute curvature, stretching, and compression terms for a velocity field.

    Index notation:
        curvature   = n_i n_j (du_j/dx_i)
        stretching  = t_i t_j (du_j/dx_i)
        compression = du_i/dx_i
    """
    field_types.ensure_vfield(u_vfield)
    field_types.ensure_uvfield(tangent_uvfield)
    field_types.ensure_uvfield(normal_uvfield)
    field_types.ensure_uniform_domain(uniform_domain)
    field_types.ensure_domain_matches_vfield(uniform_domain, u_vfield)
    field_types.ensure_domain_matches_vfield(uniform_domain, tangent_uvfield)
    field_types.ensure_domain_matches_vfield(uniform_domain, normal_uvfield)
    sim_time = u_vfield.sim_time
    tangent_uvarray = tangent_uvfield.data
    normal_uvarray = normal_uvfield.data
    ## du_j/dx_i with layout (j, i, x, y, z)
    gradu_r2tarray = array_operators.compute_varray_grad(
        varray=u_vfield.data,
        cell_widths=uniform_domain.cell_widths,
        grad_order=grad_order,
        out_r2tarray=out_grad_r2tarray,
    )
    array_types.ensure_r2tarray(gradu_r2tarray)
    curvature_sarray = numpy.einsum(
        "ixyz,jxyz,jixyz->xyz",
        normal_uvarray,
        normal_uvarray,
        gradu_r2tarray,
        optimize=True,
    )
    curvature_sfield = field_types.ScalarField(
        sim_time=sim_time,
        data=curvature_sarray,
        field_label=r"$n_i n_j \,\partial_i u_j$",
    )
    stretching_sarray = numpy.einsum(
        "ixyz,jxyz,jixyz->xyz",
        tangent_uvarray,
        tangent_uvarray,
        gradu_r2tarray,
        optimize=True,
    )
    stretching_sfield = field_types.ScalarField(
        sim_time=sim_time,
        data=stretching_sarray,
        field_label=r"$t_i t_j \,\partial_i u_j$",
    )
    compression_sarray = numpy.trace(gradu_r2tarray, axis1=0, axis2=1)
    del gradu_r2tarray
    compression_sfield = field_types.ScalarField(
        sim_time=sim_time,
        data=compression_sarray,
        field_label=r"$\partial_i u_i$",
    )
    return MagneticCurvatureTerms(
        curvature_sfield=curvature_sfield,
        stretching_sfield=stretching_sfield,
        compression_sfield=compression_sfield,
    )


@func_utils.time_function
def compute_lorentz_force_terms(
    b_vfield: field_types.VectorField,
    uniform_domain: field_types.UniformDomain,
    grad_order: int = 2,
) -> LorentzForceTerms:
    """
    Lorentz force decomposition:
        tension_i    = |b|^2 * kappa_i
        gradP_perp_i = 0.5 * d_i |b|^2 - 0.5 * t_i t_j d_j |b|^2
        lorentz_i    = tension_i - gradP_perp_i
    """
    field_types.ensure_vfield(b_vfield)
    field_types.ensure_uniform_domain(uniform_domain)
    field_types.ensure_domain_matches_vfield(uniform_domain, b_vfield)
    sim_time = b_vfield.sim_time
    tnb_terms = decompose_fields.compute_tnb_terms(
        vfield=b_vfield,
        uniform_domain=uniform_domain,
        grad_order=grad_order,
    )
    tangent_uvarray = tnb_terms.tangent_uvfield.data
    normal_uvarray = tnb_terms.normal_uvfield.data
    curvature_sarray = tnb_terms.curvature_sfield.data
    del tnb_terms
    b_magn_sq_sarray = array_operators.sum_of_component_squares(b_vfield.data)
    ## d_i P where P = 0.5 * |b|^2; first compute d_i |b|^2 then scale
    gradP_varray = array_operators.compute_sarray_grad(
        sarray=b_magn_sq_sarray,
        cell_widths=uniform_domain.cell_widths,
        grad_order=grad_order,
    )
    gradP_varray *= 0.5  # scale in-place
    ## pressure aligned with vec(b): t_i t_j d_j P
    gradP_aligned_varray = numpy.einsum(
        "ixyz,jxyz,jxyz->ixyz",
        tangent_uvarray,
        tangent_uvarray,
        gradP_varray,
        optimize=True,
    )
    ## |b|^2 * kappa_i
    tension_varray = b_magn_sq_sarray[numpy.newaxis, ...] * curvature_sarray[numpy.newaxis, ...] * normal_uvarray
    ## d_i P - t_i t_j d_j P
    gradP_perp_varray = gradP_varray - gradP_aligned_varray
    ## tension - gradP_perp
    lorentz_varray = tension_varray - gradP_perp_varray
    ## output fields
    gradP_perp_vfield = field_types.VectorField(
        sim_time=sim_time,
        data=gradP_perp_varray,
        field_label=r"$(\nabla (b^2/2))_\perp$",
    )
    tension_vfield = field_types.VectorField(
        sim_time=sim_time,
        data=tension_varray,
        field_label=r"$b^2 \,\vec{\kappa}$",
    )
    lorentz_vfield = field_types.VectorField(
        sim_time=sim_time,
        data=lorentz_varray,
        field_label=r"$(\nabla\times\vec{b})\times\vec{b}$",
    )
    return LorentzForceTerms(
        lorentz_vfield=lorentz_vfield,
        tension_vfield=tension_vfield,
        gradP_perp_vfield=gradP_perp_vfield,
    )


@func_utils.time_function
def compute_dissipation_function(
    u_vfield: field_types.VectorField,
    uniform_domain: field_types.UniformDomain,
    grad_order: int = 2,
) -> field_types.VectorField:
    """
    Compute d_j S_ji, where
        S_ij = 0.5 * (d_i u_j + d_j u_i) - (1/3) * delta_ij * (d_k u_k)
    """
    field_types.ensure_vfield(u_vfield)
    field_types.ensure_uniform_domain(uniform_domain)
    field_types.ensure_domain_matches_vfield(uniform_domain, u_vfield)
    sim_time = u_vfield.sim_time
    u_varray = u_vfield.data
    dtype = u_vfield.data.dtype
    cell_width_x, cell_width_y, cell_width_z = uniform_domain.cell_widths
    num_cells_x, num_cells_y, num_cells_z = uniform_domain.resolution
    ## d_i u_j
    gradu_r2tarray = array_operators.compute_varray_grad(
        varray=u_varray,
        cell_widths=uniform_domain.cell_widths,
        grad_order=grad_order,
    )
    divu_sarray = numpy.trace(gradu_r2tarray, axis1=0, axis2=1)
    ## S_ij = 0.5 * (d_i u_j + d_j u_i) - (1/3) delta_ij * (d_k u_k)
    sym_term_r2tarray = 0.5 * gradu_r2tarray + numpy.transpose(gradu_r2tarray, axes=(1, 0, 2, 3, 4))
    identity_matrix = numpy.eye(3, dtype=dtype)
    bulk_term_r2tarray = numpy.einsum(
        "ij,xyz->jixyz",
        identity_matrix,
        divu_sarray,
        optimize=True,
    )
    sr_r2tarray = sym_term_r2tarray - (1.0 / 3.0) * bulk_term_r2tarray
    ## d_j S_ji = d_x S_xi + d_y S_yi + d_z S_zi
    nabla = finite_difference.get_grad_func(grad_order)
    df_varray = array_operators.ensure_array_properties(
        array_shape=(3, num_cells_x, num_cells_y, num_cells_z),
        dtype=dtype,
    )
    array_types.ensure_varray(df_varray)
    for comp_i in range(3):
        ## d_x S_xi
        df_varray[comp_i, ...] = nabla(sarray=sr_r2tarray[0, comp_i], cell_width=cell_width_x, grad_axis=0)
        ## + d_y S_yi
        numpy.add(
            df_varray[comp_i, ...],
            nabla(sarray=sr_r2tarray[1, comp_i], cell_width=cell_width_y, grad_axis=1),
            out=df_varray[comp_i, ...],
        )
        ## + d_z S_zi
        numpy.add(
            df_varray[comp_i, ...],
            nabla(sarray=sr_r2tarray[2, comp_i], cell_width=cell_width_z, grad_axis=2),
            out=df_varray[comp_i, ...],
        )
    return field_types.VectorField(
        sim_time=sim_time,
        data=df_varray,
        field_label=r"$\partial_j \mathcal{S}_{j i}$",
    )


## } MODULE
