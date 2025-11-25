## { MODULE

##
## === DEPENDENCIES
##

from dataclasses import dataclass

from jormi.ww_fields.fields_3d import (
    domain_types,
    field_types,
    decompose_fdata,
)


##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class HelmholtzDecomposedFields_3D:
    vfield_3d_div: field_types.VectorField_3D
    vfield_3d_sol: field_types.VectorField_3D
    vfield_3d_bulk: field_types.VectorField_3D

    def __post_init__(
        self,
    ) -> None:
        field_types.ensure_3d_vfield(
            vfield_3d=self.vfield_3d_div,
            param_name="<vfield_3d_div>",
        )
        field_types.ensure_3d_vfield(
            vfield_3d=self.vfield_3d_sol,
            param_name="<vfield_3d_sol>",
        )
        field_types.ensure_3d_vfield(
            vfield_3d=self.vfield_3d_bulk,
            param_name="<vfield_3d_bulk>",
        )
        field_types.ensure_same_3d_field_shape(
            field_3d_a=self.vfield_3d_div,
            field_3d_b=self.vfield_3d_sol,
            field_name_a="<vfield_3d_div>",
            field_name_b="<vfield_3d_sol>",
        )
        field_types.ensure_same_3d_field_shape(
            field_3d_a=self.vfield_3d_div,
            field_3d_b=self.vfield_3d_bulk,
            field_name_a="<vfield_3d_div>",
            field_name_b="<vfield_3d_bulk>",
        )
        field_types.ensure_same_3d_field_udomains(
            field_3d_a=self.vfield_3d_div,
            field_3d_b=self.vfield_3d_sol,
            field_name_a="<vfield_3d_div>",
            field_name_b="<vfield_3d_sol>",
        )
        field_types.ensure_same_3d_field_udomains(
            field_3d_a=self.vfield_3d_div,
            field_3d_b=self.vfield_3d_bulk,
            field_name_a="<vfield_3d_div>",
            field_name_b="<vfield_3d_bulk>",
        )


@dataclass(frozen=True)
class TNBDecomposedFields_3D:
    uvfield_3d_tangent: field_types.UnitVectorField_3D
    uvfield_3d_normal: field_types.UnitVectorField_3D
    uvfield_3d_binormal: field_types.UnitVectorField_3D
    sfield_3d_curvature: field_types.ScalarField_3D

    def __post_init__(
        self,
    ) -> None:
        field_types.ensure_3d_uvfield(
            uvfield_3d=self.uvfield_3d_tangent,
            param_name="<uvfield_3d_tangent>",
        )
        field_types.ensure_3d_uvfield(
            uvfield_3d=self.uvfield_3d_normal,
            param_name="<uvfield_3d_normal>",
        )
        field_types.ensure_3d_uvfield(
            uvfield_3d=self.uvfield_3d_binormal,
            param_name="<uvfield_3d_binormal>",
        )
        field_types.ensure_3d_sfield(
            sfield_3d=self.sfield_3d_curvature,
            param_name="<sfield_3d_curvature>",
        )
        field_types.ensure_same_3d_field_shape(
            field_3d_a=self.uvfield_3d_tangent,
            field_3d_b=self.uvfield_3d_normal,
            field_name_a="<uvfield_3d_tangent>",
            field_name_b="<uvfield_3d_normal>",
        )
        field_types.ensure_same_3d_field_shape(
            field_3d_a=self.uvfield_3d_tangent,
            field_3d_b=self.uvfield_3d_binormal,
            field_name_a="<uvfield_3d_tangent>",
            field_name_b="<uvfield_3d_binormal>",
        )
        field_types.ensure_same_3d_field_udomains(
            field_3d_a=self.uvfield_3d_tangent,
            field_3d_b=self.uvfield_3d_normal,
            field_name_a="<uvfield_3d_tangent>",
            field_name_b="<uvfield_3d_normal>",
        )
        field_types.ensure_same_3d_field_udomains(
            field_3d_a=self.uvfield_3d_tangent,
            field_3d_b=self.uvfield_3d_binormal,
            field_name_a="<uvfield_3d_tangent>",
            field_name_b="<uvfield_3d_binormal>",
        )
        field_types.ensure_same_3d_field_udomains(
            field_3d_a=self.uvfield_3d_tangent,
            field_3d_b=self.sfield_3d_curvature,
            field_name_a="<uvfield_3d_tangent>",
            field_name_b="<sfield_3d_curvature>",
        )


##
## === FUNCTIONS
##


def compute_helmholtz_decomposition(
    vfield_3d_q: field_types.VectorField_3D,
) -> HelmholtzDecomposedFields_3D:
    """Field-level Helmholtz decomposition wrapper."""
    field_types.ensure_3d_vfield(
        vfield_3d=vfield_3d_q,
        param_name="<vfield_3d_q>",
    )
    udomain_3d = vfield_3d_q.udomain
    domain_types.ensure_3d_udomain(
        udomain_3d=udomain_3d,
        param_name="<udomain_3d>",
    )
    if not all(udomain_3d.periodicity):
        raise ValueError("Helmholtz (FFT) assumes periodic BCs in all directions.")
    sim_time = vfield_3d_q.sim_time
    helmholtz_3d_fdata = decompose_fdata.compute_helmholtz_decomposition(
        vdata_3d_q=vfield_3d_q.fdata,
        resolution=udomain_3d.resolution,
        cell_widths=udomain_3d.cell_widths,
    )
    vfield_3d_div = field_types.VectorField_3D(
        fdata=helmholtz_3d_fdata.vdata_3d_div,
        udomain=udomain_3d,
        field_label="f_{i,div}",
        sim_time=sim_time,
    )
    vfield_3d_sol = field_types.VectorField_3D(
        fdata=helmholtz_3d_fdata.vdata_3d_sol,
        udomain=udomain_3d,
        field_label="f_{i,sol}",
        sim_time=sim_time,
    )
    vfield_3d_bulk = field_types.VectorField_3D(
        fdata=helmholtz_3d_fdata.vdata_3d_bulk,
        udomain=udomain_3d,
        field_label="f_{i,bulk}",
        sim_time=sim_time,
    )
    return HelmholtzDecomposedFields_3D(
        vfield_3d_div=vfield_3d_div,
        vfield_3d_sol=vfield_3d_sol,
        vfield_3d_bulk=vfield_3d_bulk,
    )


def compute_tnb_terms(
    vfield_3d: field_types.VectorField_3D,
    *,
    grad_order: int = 2,
) -> TNBDecomposedFields_3D:
    """Field-level TNB decomposition wrapper."""
    field_types.ensure_3d_vfield(
        vfield_3d=vfield_3d,
        param_name="<vfield_3d>",
    )
    udomain_3d = vfield_3d.udomain
    sim_time = vfield_3d.sim_time
    tnb_3d_fdata = decompose_fdata.compute_tnb_terms(
        vdata_3d=vfield_3d.fdata,
        cell_widths=udomain_3d.cell_widths,
        grad_order=grad_order,
    )
    uvfield_3d_tangent = field_types.UnitVectorField_3D(
        fdata=tnb_3d_fdata.vdata_3d_tangent,
        udomain=udomain_3d,
        field_label="t_i",
        sim_time=sim_time,
    )
    uvfield_3d_normal = field_types.UnitVectorField_3D(
        fdata=tnb_3d_fdata.vdata_3d_normal,
        udomain=udomain_3d,
        field_label="n_i",
        sim_time=sim_time,
    )
    uvfield_3d_binormal = field_types.UnitVectorField_3D(
        fdata=tnb_3d_fdata.vdata_3d_binormal,
        udomain=udomain_3d,
        field_label="b_i",
        sim_time=sim_time,
    )
    sfield_3d_curvature = field_types.ScalarField_3D(
        fdata=tnb_3d_fdata.sdata_3d_curvature,
        udomain=udomain_3d,
        field_label="sqrt(kappa_i kappa_i)",
        sim_time=sim_time,
    )
    return TNBDecomposedFields_3D(
        uvfield_3d_tangent=uvfield_3d_tangent,
        uvfield_3d_normal=uvfield_3d_normal,
        uvfield_3d_binormal=uvfield_3d_binormal,
        sfield_3d_curvature=sfield_3d_curvature,
    )


## } MODULE
