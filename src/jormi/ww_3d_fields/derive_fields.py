## { MODULE

##
## === DEPENDENCIES
##

import numpy

from dataclasses import dataclass

from jormi.ww_3d_fields import (
    fdata_types,
    finite_difference,
    fdata_operators,
    domain_types,
    field_types,
    decompose_fields,
)


##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class MagneticCurvatureTerms:
    curvature_sfield: field_types.ScalarField
    stretching_sfield: field_types.ScalarField
    compression_sfield: field_types.ScalarField

    def __post_init__(
        self,
    ) -> None:
        field_types.ensure_sfield(
            sfield=self.curvature_sfield,
            param_name="<curvature_sfield>",
        )
        field_types.ensure_sfield(
            sfield=self.stretching_sfield,
            param_name="<stretching_sfield>",
        )
        field_types.ensure_sfield(
            sfield=self.compression_sfield,
            param_name="<compression_sfield>",
        )
        field_types.ensure_same_field_shape(
            field_a=self.curvature_sfield,
            field_b=self.stretching_sfield,
            field_name_a="<curvature_sfield>",
            field_name_b="<stretching_sfield>",
        )
        field_types.ensure_same_field_shape(
            field_a=self.curvature_sfield,
            field_b=self.compression_sfield,
            field_name_a="<curvature_sfield>",
            field_name_b="<compression_sfield>",
        )
        if any(
            [
                self.curvature_sfield.udomain != self.stretching_sfield.udomain,
                self.curvature_sfield.udomain != self.compression_sfield.udomain,
            ],
        ):
            raise ValueError("MagneticCurvatureTerms fields must share the same UniformDomain.")


@dataclass(frozen=True)
class LorentzForceTerms:
    lorentz_vfield: field_types.VectorField
    tension_vfield: field_types.VectorField
    gradP_perp_vfield: field_types.VectorField

    def __post_init__(
        self,
    ) -> None:
        field_types.ensure_vfield(
            vfield=self.lorentz_vfield,
            param_name="<lorentz_vfield>",
        )
        field_types.ensure_vfield(
            vfield=self.tension_vfield,
            param_name="<tension_vfield>",
        )
        field_types.ensure_vfield(
            vfield=self.gradP_perp_vfield,
            param_name="<gradP_perp_vfield>",
        )
        field_types.ensure_same_field_shape(
            field_a=self.lorentz_vfield,
            field_b=self.tension_vfield,
            field_name_a="<lorentz_vfield>",
            field_name_b="<tension_vfield>",
        )
        field_types.ensure_same_field_shape(
            field_a=self.lorentz_vfield,
            field_b=self.gradP_perp_vfield,
            field_name_a="<lorentz_vfield>",
            field_name_b="<gradP_perp_vfield>",
        )
        if any(
            [
                self.lorentz_vfield.udomain != self.tension_vfield.udomain,
                self.lorentz_vfield.udomain != self.gradP_perp_vfield.udomain,
            ],
        ):
            raise ValueError("LorentzForceTerms fields must share the same UniformDomain.")


##
## === FUNCTIONS
##


# @fn_utils.time_fn
def compute_magnetic_curvature_terms(
    u_vfield: field_types.VectorField,
    tangent_uvfield: field_types.UnitVectorField,
    normal_uvfield: field_types.UnitVectorField,
    udomain: domain_types.UniformDomain,
    out_grad_r2tarray: numpy.ndarray | None = None,
    grad_order: int = 2,
) -> MagneticCurvatureTerms:
    """
    Compute curvature, stretching, and compression terms for a 3D velocity field.

    Index notation (Einstein summation):
        curvature   = n_i n_j d_i u_j
        stretching  = t_i t_j d_i u_j
        compression = d_i u_i
    """
    field_types.ensure_vfield(
        vfield=u_vfield,
        param_name="<u_vfield>",
    )
    field_types.ensure_uvfield(
        uvfield=tangent_uvfield,
        param_name="<tangent_uvfield>",
    )
    field_types.ensure_uvfield(
        uvfield=normal_uvfield,
        param_name="<normal_uvfield>",
    )
    domain_types.ensure_udomain(
        udomain=udomain,
        param_name="<udomain>",
    )
    field_types.ensure_udomain_matches_vfield(
        udomain=udomain,
        vfield=u_vfield,
        vfield_name="<u_vfield>",
    )
    field_types.ensure_udomain_matches_vfield(
        udomain=udomain,
        vfield=tangent_uvfield,
        vfield_name="<tangent_uvfield>",
    )
    field_types.ensure_udomain_matches_vfield(
        udomain=udomain,
        vfield=normal_uvfield,
        vfield_name="<normal_uvfield>",
    )
    sim_time = u_vfield.sim_time
    tangent_uvarray = tangent_uvfield.fdata.farray
    normal_uvarray = normal_uvfield.fdata.farray
    ## d_i u_j: (j, i, x, y, z)
    gradu_r2tarray = fdata_operators.compute_varray_grad(
        vdata=u_vfield.fdata,
        cell_widths=udomain.cell_widths,
        out_r2tarray=out_grad_r2tarray,
        grad_order=grad_order,
    )
    fdata_types.ensure_r2tarray(
        r2tarray=gradu_r2tarray,
        param_name="<gradu_r2tarray>",
    )
    curvature_sarray = numpy.einsum(
        "ixyz,jxyz,jixyz->xyz",
        normal_uvarray,
        normal_uvarray,
        gradu_r2tarray,
        optimize=True,
    )
    curvature_sdata = fdata_types.ScalarFieldData(
        farray=curvature_sarray,
        param_name="<curvature.fdata>",
    )
    curvature_sfield = field_types.ScalarField(
        fdata=curvature_sdata,
        field_label=r"$n_i n_j d_i u_j$",
        udomain=udomain,
        sim_time=sim_time,
    )
    stretching_sarray = numpy.einsum(
        "ixyz,jxyz,jixyz->xyz",
        tangent_uvarray,
        tangent_uvarray,
        gradu_r2tarray,
        optimize=True,
    )
    stretching_sdata = fdata_types.ScalarFieldData(
        farray=stretching_sarray,
        param_name="<stretching.fdata>",
    )
    stretching_sfield = field_types.ScalarField(
        fdata=stretching_sdata,
        field_label=r"$t_i t_j d_i u_j$",
        udomain=udomain,
        sim_time=sim_time,
    )
    compression_sarray = numpy.trace(
        gradu_r2tarray,
        axis1=0,
        axis2=1,
    )
    del gradu_r2tarray
    compression_sdata = fdata_types.ScalarFieldData(
        farray=compression_sarray,
        param_name="<compression.fdata>",
    )
    compression_sfield = field_types.ScalarField(
        fdata=compression_sdata,
        field_label=r"$d_i u_i$",
        udomain=udomain,
        sim_time=sim_time,
    )
    return MagneticCurvatureTerms(
        curvature_sfield=curvature_sfield,
        stretching_sfield=stretching_sfield,
        compression_sfield=compression_sfield,
    )


# @fn_utils.time_fn
def compute_lorentz_force_terms(
    b_vfield: field_types.VectorField,
    udomain: domain_types.UniformDomain,
    grad_order: int = 2,
) -> LorentzForceTerms:
    """
    Lorentz force decomposition in index notation:

        tension_i    = (b_k b_k) kappa_i
        gradP_perp_i = d_i (b_k b_k / 2) - t_i t_j d_j (b_k b_k / 2)
        lorentz_i    = tension_i - gradP_perp_i
    """
    field_types.ensure_vfield(
        vfield=b_vfield,
        param_name="<b_vfield>",
    )
    domain_types.ensure_udomain(
        udomain=udomain,
        param_name="<udomain>",
    )
    field_types.ensure_udomain_matches_vfield(
        udomain=udomain,
        vfield=b_vfield,
        domain_name="<udomain>",
        vfield_name="<b_vfield>",
    )
    sim_time = b_vfield.sim_time
    tnb_terms = decompose_fields.compute_tnb_terms(
        vfield=b_vfield,
        udomain=udomain,
        grad_order=grad_order,
    )
    tangent_uvarray = tnb_terms.tangent_uvfield.fdata.farray
    normal_uvarray = tnb_terms.normal_uvfield.fdata.farray
    curvature_sarray = tnb_terms.curvature_sfield.fdata.farray
    del tnb_terms
    b_magn_sq_sarray = fdata_operators.sum_of_varray_comps_squared(
        vdata=b_vfield.fdata,
    )
    ## d_i P where P = 0.5 * |b|^2; first compute d_i |b|^2 then scale
    gradP_varray = fdata_operators.compute_sdata_grad(
        sdata=b_magn_sq_sarray,
        cell_widths=udomain.cell_widths,
        grad_order=grad_order,
    )
    gradP_varray *= 0.5  # scale in-place
    ## pressure aligned with b: t_i t_j d_j P
    gradP_aligned_varray = numpy.einsum(
        "ixyz,jxyz,jxyz->ixyz",
        tangent_uvarray,
        tangent_uvarray,
        gradP_varray,
        optimize=True,
    )
    ## |b|^2 * kappa_i
    tension_varray = (
        b_magn_sq_sarray[numpy.newaxis, ...]
        * curvature_sarray[numpy.newaxis, ...]
        * normal_uvarray
    )
    ## d_i P - t_i t_j d_j P
    gradP_perp_varray = gradP_varray - gradP_aligned_varray
    ## tension - gradP_perp
    lorentz_varray = tension_varray - gradP_perp_varray
    ## output fields
    gradP_perp_vdata = fdata_types.VectorFieldData(
        farray=gradP_perp_varray,
        param_name="<gradP_perp.fdata>",
    )
    gradP_perp_vfield = field_types.VectorField(
        fdata=gradP_perp_vdata,
        field_label=r"$[d_i (b_k b_k / 2)]_\perp$",
        udomain=udomain,
        sim_time=sim_time,
    )
    tension_vdata = fdata_types.VectorFieldData(
        farray=tension_varray,
        param_name="<tension.fdata>",
    )
    tension_vfield = field_types.VectorField(
        fdata=tension_vdata,
        field_label=r"$b_k b_k \kappa_i$",
        udomain=udomain,
        sim_time=sim_time,
    )
    lorentz_vdata = fdata_types.VectorFieldData(
        farray=lorentz_varray,
        param_name="<lorentz.fdata>",
    )
    lorentz_vfield = field_types.VectorField(
        fdata=lorentz_vdata,
        field_label=r"$L_i$",
        udomain=udomain,
        sim_time=sim_time,
    )
    return LorentzForceTerms(
        lorentz_vfield=lorentz_vfield,
        tension_vfield=tension_vfield,
        gradP_perp_vfield=gradP_perp_vfield,
    )


# @fn_utils.time_fn
def compute_dissipation_fntion(  # sic: keep name as-is
    u_vfield: field_types.VectorField,
    udomain: domain_types.UniformDomain,
    grad_order: int = 2,
) -> field_types.VectorField:
    """
    Compute d_j S_ji, where

        S_ij = 0.5 * (d_i u_j + d_j u_i) - (1/3) delta_ij (d_k u_k)
    """
    field_types.ensure_vfield(
        vfield=u_vfield,
        param_name="<u_vfield>",
    )
    domain_types.ensure_udomain(udomain=udomain)
    field_types.ensure_udomain_matches_vfield(
        udomain=udomain,
        vfield=u_vfield,
        vfield_name="<u_vfield>",
    )
    sim_time = u_vfield.sim_time
    u_varray = u_vfield.fdata.farray
    dtype = u_varray.dtype
    cell_width_x, cell_width_y, cell_width_z = udomain.cell_widths
    num_cells_x, num_cells_y, num_cells_z = udomain.resolution
    ## d_i u_j
    gradu_r2tarray = fdata_operators.compute_varray_grad(
        vdata=u_vfield.fdata,
        cell_widths=udomain.cell_widths,
        grad_order=grad_order,
    )
    divu_sarray = numpy.trace(
        gradu_r2tarray,
        axis1=0,
        axis2=1,
    )
    ## S_ij = 0.5 * (d_i u_j + d_j u_i) - (1/3) delta_ij * (d_k u_k)
    sym_term_r2tarray = 0.5 * gradu_r2tarray + numpy.transpose(
        gradu_r2tarray,
        axes=(1, 0, 2, 3, 4),
    )
    identity_matrix = numpy.eye(3, dtype=dtype)
    bulk_term_r2tarray = numpy.einsum(
        "ij,xyz->jixyz",
        identity_matrix,
        divu_sarray,
        optimize=True,
    )
    sr_r2tarray = sym_term_r2tarray - (1.0 / 3.0) * bulk_term_r2tarray
    ## d_j S_ji = d_x S_xi + d_y S_yi + d_z S_zi
    nabla = finite_difference.get_grad_fn(grad_order)
    df_varray = fdata_operators.ensure_properties(
        farray_shape=(3, num_cells_x, num_cells_y, num_cells_z),
        farray=None,
        dtype=dtype,
    )
    fdata_types.ensure_varray(
        varray=df_varray,
        param_name="<df_varray>",
    )
    for comp_i in range(3):
        ## d_x S_xi
        df_varray[comp_i, ...] = nabla(
            sarray=sr_r2tarray[0, comp_i],
            cell_width=cell_width_x,
            grad_axis=0,
        )
        ## + d_y S_yi
        numpy.add(
            df_varray[comp_i, ...],
            nabla(
                sarray=sr_r2tarray[1, comp_i],
                cell_width=cell_width_y,
                grad_axis=1,
            ),
            out=df_varray[comp_i, ...],
        )
        ## + d_z S_zi
        numpy.add(
            df_varray[comp_i, ...],
            nabla(
                sarray=sr_r2tarray[2, comp_i],
                cell_width=cell_width_z,
                grad_axis=2,
            ),
            out=df_varray[comp_i, ...],
        )
    df_vdata = fdata_types.VectorFieldData(
        farray=df_varray,
        param_name="<dissipation.df.fdata>",
    )
    return field_types.VectorField(
        fdata=df_vdata,
        field_label=r"$d_j \mathcal{S}_{j i}$",
        udomain=udomain,
        sim_time=sim_time,
    )


## } MODULE
