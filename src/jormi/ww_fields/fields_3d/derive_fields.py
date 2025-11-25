## { MODULE

##
## === DEPENDENCIES
##

from dataclasses import dataclass

from jormi.ww_fields.fields_3d import (
    domain_types,
    field_types,
    derive_fdata,
)


##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class MagneticCurvatureFields_3D:
    sfield_3d_curvature: field_types.ScalarField_3D
    sfield_3d_stretching: field_types.ScalarField_3D
    sfield_3d_compression: field_types.ScalarField_3D

    def __post_init__(
        self,
    ) -> None:
        field_types.ensure_3d_sfield(
            sfield_3d=self.sfield_3d_curvature,
            param_name="<sfield_3d_curvature>",
        )
        field_types.ensure_3d_sfield(
            sfield_3d=self.sfield_3d_stretching,
            param_name="<sfield_3d_stretching>",
        )
        field_types.ensure_3d_sfield(
            sfield_3d=self.sfield_3d_compression,
            param_name="<sfield_3d_compression>",
        )
        field_types.ensure_same_3d_field_shape(
            field_3d_a=self.sfield_3d_curvature,
            field_3d_b=self.sfield_3d_stretching,
            field_name_a="<sfield_3d_curvature>",
            field_name_b="<sfield_3d_stretching>",
        )
        field_types.ensure_same_3d_field_shape(
            field_3d_a=self.sfield_3d_curvature,
            field_3d_b=self.sfield_3d_compression,
            field_name_a="<sfield_3d_curvature>",
            field_name_b="<sfield_3d_compression>",
        )
        field_types.ensure_same_3d_field_udomains(
            field_3d_a=self.sfield_3d_curvature,
            field_3d_b=self.sfield_3d_stretching,
            field_name_a="<sfield_3d_curvature>",
            field_name_b="<sfield_3d_stretching>",
        )
        field_types.ensure_same_3d_field_udomains(
            field_3d_a=self.sfield_3d_curvature,
            field_3d_b=self.sfield_3d_compression,
            field_name_a="<sfield_3d_curvature>",
            field_name_b="<sfield_3d_compression>",
        )


@dataclass(frozen=True)
class LorentzForceFields_3D:
    vfield_3d_lorentz: field_types.VectorField_3D
    vfield_3d_tension: field_types.VectorField_3D
    vfield_3d_gradP_perp: field_types.VectorField_3D

    def __post_init__(
        self,
    ) -> None:
        field_types.ensure_3d_vfield(
            vfield_3d=self.vfield_3d_lorentz,
            param_name="<vfield_3d_lorentz>",
        )
        field_types.ensure_3d_vfield(
            vfield_3d=self.vfield_3d_tension,
            param_name="<vfield_3d_tension>",
        )
        field_types.ensure_3d_vfield(
            vfield_3d=self.vfield_3d_gradP_perp,
            param_name="<vfield_3d_gradP_perp>",
        )
        field_types.ensure_same_3d_field_shape(
            field_3d_a=self.vfield_3d_lorentz,
            field_3d_b=self.vfield_3d_tension,
            field_name_a="<vfield_3d_lorentz>",
            field_name_b="<vfield_3d_tension>",
        )
        field_types.ensure_same_3d_field_shape(
            field_3d_a=self.vfield_3d_lorentz,
            field_3d_b=self.vfield_3d_gradP_perp,
            field_name_a="<vfield_3d_lorentz>",
            field_name_b="<vfield_3d_gradP_perp>",
        )
        field_types.ensure_same_3d_field_udomains(
            field_3d_a=self.vfield_3d_lorentz,
            field_3d_b=self.vfield_3d_tension,
            field_name_a="<vfield_3d_lorentz>",
            field_name_b="<vfield_3d_tension>",
        )
        field_types.ensure_same_3d_field_udomains(
            field_3d_a=self.vfield_3d_lorentz,
            field_3d_b=self.vfield_3d_gradP_perp,
            field_name_a="<vfield_3d_lorentz>",
            field_name_b="<vfield_3d_gradP_perp>",
        )


##
## === FUNCTIONS
##


def compute_magnetic_curvature_terms(
    *,
    vfield_3d_u: field_types.VectorField_3D,
    uvfield_3d_tangent: field_types.UnitVectorField_3D,
    uvfield_3d_normal: field_types.UnitVectorField_3D,
    grad_order: int = 2,
) -> MagneticCurvatureFields_3D:
    """
    Field-level wrapper for magnetic curvature, stretching, compression.

    Uses:
        curvature   = n_i n_j d_i u_j
        stretching  = t_i t_j d_i u_j
        compression = d_i u_i
    """
    field_types.ensure_3d_vfield(
        vfield_3d=vfield_3d_u,
        param_name="<vfield_3d_u>",
    )
    field_types.ensure_3d_uvfield(
        uvfield_3d=uvfield_3d_tangent,
        param_name="<uvfield_3d_tangent>",
    )
    field_types.ensure_3d_uvfield(
        uvfield_3d=uvfield_3d_normal,
        param_name="<uvfield_3d_normal>",
    )
    field_types.ensure_same_3d_field_udomains(
        field_3d_a=vfield_3d_u,
        field_3d_b=uvfield_3d_tangent,
        field_name_a="<vfield_3d_u>",
        field_name_b="<uvfield_3d_tangent>",
    )
    field_types.ensure_same_3d_field_udomains(
        field_3d_a=vfield_3d_u,
        field_3d_b=uvfield_3d_normal,
        field_name_a="<vfield_3d_u>",
        field_name_b="<uvfield_3d_normal>",
    )
    udomain_3d = vfield_3d_u.udomain
    sim_time = vfield_3d_u.sim_time
    mct_3d_fdata = derive_fdata.compute_magnetic_curvature_terms(
        vdata_3d_u=vfield_3d_u.fdata,
        uvdata_3d_tangent=uvfield_3d_tangent.fdata,
        uvdata_3d_normal=uvfield_3d_normal.fdata,
        cell_widths=udomain_3d.cell_widths,
        grad_order=grad_order,
    )
    sfield_3d_curvature = field_types.ScalarField_3D(
        fdata=mct_3d_fdata.sdata_3d_curvature,
        udomain=udomain_3d,
        field_label=r"$n_i n_j d_i u_j$",
        sim_time=sim_time,
    )
    sfield_3d_stretching = field_types.ScalarField_3D(
        fdata=mct_3d_fdata.sdata_3d_stretching,
        udomain=udomain_3d,
        field_label=r"$t_i t_j d_i u_j$",
        sim_time=sim_time,
    )
    sfield_3d_compression = field_types.ScalarField_3D(
        fdata=mct_3d_fdata.sdata_3d_compression,
        udomain=udomain_3d,
        field_label=r"$d_i u_i$",
        sim_time=sim_time,
    )
    return MagneticCurvatureFields_3D(
        sfield_3d_curvature=sfield_3d_curvature,
        sfield_3d_stretching=sfield_3d_stretching,
        sfield_3d_compression=sfield_3d_compression,
    )


def compute_lorentz_force_terms(
    vfield_3d_b: field_types.VectorField_3D,
    *,
    grad_order: int = 2,
) -> LorentzForceFields_3D:
    """
    Field-level Lorentz force decomposition wrapper.
    """
    field_types.ensure_3d_vfield(
        vfield_3d=vfield_3d_b,
        param_name="<vfield_3d_b>",
    )
    udomain_3d = vfield_3d_b.udomain
    domain_types.ensure_3d_udomain(
        udomain_3d=udomain_3d,
        param_name="<udomain_3d>",
    )
    sim_time = vfield_3d_b.sim_time
    lft_3d_fdata = derive_fdata.compute_lorentz_force_terms(
        vdata_3d_b=vfield_3d_b.fdata,
        cell_widths=udomain_3d.cell_widths,
        grad_order=grad_order,
    )
    vfield_3d_lorentz = field_types.VectorField_3D(
        fdata=lft_3d_fdata.vdata_3d_lorentz,
        udomain=udomain_3d,
        field_label=r"$L_i$",
        sim_time=sim_time,
    )
    vfield_3d_tension = field_types.VectorField_3D(
        fdata=lft_3d_fdata.vdata_3d_tension,
        udomain=udomain_3d,
        field_label=r"$b_k b_k \kappa_i$",
        sim_time=sim_time,
    )
    vfield_3d_gradP_perp = field_types.VectorField_3D(
        fdata=lft_3d_fdata.vdata_3d_gradP_perp,
        udomain=udomain_3d,
        field_label=r"$[d_i (b_k b_k / 2)]_\perp$",
        sim_time=sim_time,
    )
    return LorentzForceFields_3D(
        vfield_3d_lorentz=vfield_3d_lorentz,
        vfield_3d_tension=vfield_3d_tension,
        vfield_3d_gradP_perp=vfield_3d_gradP_perp,
    )


def compute_kinetic_dissipation(
    vfield_3d_u: field_types.VectorField_3D,
    *,
    grad_order: int = 2,
) -> field_types.VectorField_3D:
    """
    Compute d_j S_ji for a 3D velocity field u_i, where

        S_ij = 0.5 * (d_i u_j + d_j u_i) - (1/3) delta_ij (d_k u_k)
    """
    field_types.ensure_3d_vfield(
        vfield_3d=vfield_3d_u,
        param_name="<vfield_3d_u>",
    )
    udomain_3d = vfield_3d_u.udomain
    sim_time = vfield_3d_u.sim_time
    vdata_3d_df = derive_fdata.compute_kinetic_dissipation(
        vdata_3d_u=vfield_3d_u.fdata,
        cell_widths=udomain_3d.cell_widths,
        grad_order=grad_order,
    )
    return field_types.VectorField_3D(
        fdata=vdata_3d_df,
        udomain=udomain_3d,
        field_label=r"$d_j \mathcal{S}_{j i}$",
        sim_time=sim_time,
    )


## } MODULE
