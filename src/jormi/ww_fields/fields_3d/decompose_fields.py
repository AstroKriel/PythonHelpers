## { MODULE

##
## === DEPENDENCIES
##

from dataclasses import dataclass

from jormi.ww_types import type_manager
from jormi.ww_fields.fields_3d import (
    _decompose_farrays,
    domains,
    fields,
)

##
## === HELMHOLTZ DECOMPOSITION
##


@dataclass(frozen=True)
class HelmholtzDecomposedFields_3D:
    vfield_3d_div: fields.VectorField_3D
    vfield_3d_sol: fields.VectorField_3D
    vfield_3d_bulk: fields.VectorField_3D

    def __post_init__(
        self,
    ) -> None:
        fields.ensure_3d_vfield(
            vfield_3d=self.vfield_3d_div,
            param_name="<vfield_3d_div>",
        )
        fields.ensure_3d_vfield(
            vfield_3d=self.vfield_3d_sol,
            param_name="<vfield_3d_sol>",
        )
        fields.ensure_3d_vfield(
            vfield_3d=self.vfield_3d_bulk,
            param_name="<vfield_3d_bulk>",
        )
        fields.ensure_same_3d_field_shape_and_udomains(
            field_3d_a=self.vfield_3d_div,
            field_3d_b=self.vfield_3d_sol,
            field_name_a="<vfield_3d_div>",
            field_name_b="<vfield_3d_sol>",
        )
        fields.ensure_same_3d_field_shape_and_udomains(
            field_3d_a=self.vfield_3d_div,
            field_3d_b=self.vfield_3d_bulk,
            field_name_a="<vfield_3d_div>",
            field_name_b="<vfield_3d_bulk>",
        )


def compute_helmholtz_decomposed_fields(
    vfield_3d_q: fields.VectorField_3D,
) -> HelmholtzDecomposedFields_3D:
    """Field-level Helmholtz decomposition wrapper."""
    fields.ensure_3d_vfield(
        vfield_3d=vfield_3d_q,
        param_name="<vfield_3d_q>",
    )
    udomain_3d = vfield_3d_q.udomain
    domains.ensure_3d_periodic_udomain(
        udomain_3d=udomain_3d,
        param_name="<udomain_3d>",
    )
    varray_3d_q = fields.extract_3d_varray(
        vfield_3d=vfield_3d_q,
        param_name="<vfield_3d_q>",
    )
    sim_time = vfield_3d_q.sim_time
    hd_3d_farrays = _decompose_farrays.compute_helmholtz_decomposed_farrays(
        varray_3d_q=varray_3d_q,
        resolution=udomain_3d.resolution,
        cell_widths_3d=udomain_3d.cell_widths,
    )
    vfield_3d_div = fields.VectorField_3D.from_3d_varray(
        varray_3d=hd_3d_farrays.varray_3d_div,
        udomain_3d=udomain_3d,
        field_label="f_{i,div}",
        sim_time=sim_time,
    )
    vfield_3d_sol = fields.VectorField_3D.from_3d_varray(
        varray_3d=hd_3d_farrays.varray_3d_sol,
        udomain_3d=udomain_3d,
        field_label="f_{i,sol}",
        sim_time=sim_time,
    )
    vfield_3d_bulk = fields.VectorField_3D.from_3d_varray(
        varray_3d=hd_3d_farrays.varray_3d_bulk,
        udomain_3d=udomain_3d,
        field_label="f_{i,bulk}",
        sim_time=sim_time,
    )
    return HelmholtzDecomposedFields_3D(
        vfield_3d_div=vfield_3d_div,
        vfield_3d_sol=vfield_3d_sol,
        vfield_3d_bulk=vfield_3d_bulk,
    )


##
## --- TANGENT NORMAL BINORMAL BASIS
##


@dataclass(frozen=True)
class TNBDecomposedFields_3D:
    uvfield_3d_tangent: fields.UnitVectorField_3D
    uvfield_3d_normal: fields.UnitVectorField_3D
    uvfield_3d_binormal: fields.UnitVectorField_3D
    sfield_3d_curvature: fields.ScalarField_3D

    def __post_init__(
        self,
    ) -> None:
        fields.ensure_3d_uvfield(
            uvfield_3d=self.uvfield_3d_tangent,
            param_name="<uvfield_3d_tangent>",
        )
        fields.ensure_3d_uvfield(
            uvfield_3d=self.uvfield_3d_normal,
            param_name="<uvfield_3d_normal>",
        )
        fields.ensure_3d_uvfield(
            uvfield_3d=self.uvfield_3d_binormal,
            param_name="<uvfield_3d_binormal>",
        )
        fields.ensure_3d_sfield(
            sfield_3d=self.sfield_3d_curvature,
            param_name="<sfield_3d_curvature>",
        )
        fields.ensure_same_3d_field_shape_and_udomains(
            field_3d_a=self.uvfield_3d_tangent,
            field_3d_b=self.uvfield_3d_normal,
            field_name_a="<uvfield_3d_tangent>",
            field_name_b="<uvfield_3d_normal>",
        )
        fields.ensure_same_3d_field_shape_and_udomains(
            field_3d_a=self.uvfield_3d_tangent,
            field_3d_b=self.uvfield_3d_binormal,
            field_name_a="<uvfield_3d_tangent>",
            field_name_b="<uvfield_3d_binormal>",
        )
        fields.ensure_same_3d_field_udomains(
            field_3d_a=self.uvfield_3d_tangent,
            field_3d_b=self.sfield_3d_curvature,
            field_name_a="<uvfield_3d_tangent>",
            field_name_b="<sfield_3d_curvature>",
        )


def compute_tnb_decomposed_fields(
    vfield_3d: fields.VectorField_3D,
    *,
    grad_order: int = 2,
) -> TNBDecomposedFields_3D:
    """Field-level TNB decomposition wrapper."""
    fields.ensure_3d_vfield(
        vfield_3d=vfield_3d,
        param_name="<vfield_3d>",
    )
    type_manager.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    udomain_3d = vfield_3d.udomain
    varray_3d = fields.extract_3d_varray(
        vfield_3d=vfield_3d,
        param_name="<vfield_3d>",
    )
    sim_time = vfield_3d.sim_time
    tnb_3d_farrays = _decompose_farrays.compute_tnb_farrays(
        varray_3d=varray_3d,
        cell_widths_3d=udomain_3d.cell_widths,
        grad_order=grad_order,
    )
    vfield_3d_tangent = fields.VectorField_3D.from_3d_varray(
        varray_3d=tnb_3d_farrays.uvarray_3d_tangent,
        udomain_3d=udomain_3d,
        field_label="t_i",
        sim_time=sim_time,
    )
    vfield_3d_normal = fields.VectorField_3D.from_3d_varray(
        varray_3d=tnb_3d_farrays.uvarray_3d_normal,
        udomain_3d=udomain_3d,
        field_label="n_i",
        sim_time=sim_time,
    )
    vfield_3d_binormal = fields.VectorField_3D.from_3d_varray(
        varray_3d=tnb_3d_farrays.uvarray_3d_binormal,
        udomain_3d=udomain_3d,
        field_label="b_i",
        sim_time=sim_time,
    )
    sfield_3d_curvature = fields.ScalarField_3D.from_3d_sarray(
        sarray_3d=tnb_3d_farrays.sarray_3d_curvature,
        udomain_3d=udomain_3d,
        field_label="sqrt(kappa_i kappa_i)",
        sim_time=sim_time,
    )
    uvfield_3d_tangent = fields.as_3d_uvfield(
        vfield_3d=vfield_3d_tangent,
    )
    uvfield_3d_normal = fields.as_3d_uvfield(
        vfield_3d=vfield_3d_normal,
    )
    uvfield_3d_binormal = fields.as_3d_uvfield(
        vfield_3d=vfield_3d_binormal,
    )
    return TNBDecomposedFields_3D(
        uvfield_3d_tangent=uvfield_3d_tangent,
        uvfield_3d_normal=uvfield_3d_normal,
        uvfield_3d_binormal=uvfield_3d_binormal,
        sfield_3d_curvature=sfield_3d_curvature,
    )


##
## --- MAGNETIC DECOMPOSITION
##


@dataclass(frozen=True)
class MagneticCurvatureFields_3D:
    sfield_3d_curvature: fields.ScalarField_3D
    sfield_3d_stretching: fields.ScalarField_3D
    sfield_3d_compression: fields.ScalarField_3D

    def __post_init__(
        self,
    ) -> None:
        fields.ensure_3d_sfield(
            sfield_3d=self.sfield_3d_curvature,
            param_name="<sfield_3d_curvature>",
        )
        fields.ensure_3d_sfield(
            sfield_3d=self.sfield_3d_stretching,
            param_name="<sfield_3d_stretching>",
        )
        fields.ensure_3d_sfield(
            sfield_3d=self.sfield_3d_compression,
            param_name="<sfield_3d_compression>",
        )
        fields.ensure_same_3d_field_shape_and_udomains(
            field_3d_a=self.sfield_3d_curvature,
            field_3d_b=self.sfield_3d_stretching,
            field_name_a="<sfield_3d_curvature>",
            field_name_b="<sfield_3d_stretching>",
        )
        fields.ensure_same_3d_field_shape_and_udomains(
            field_3d_a=self.sfield_3d_curvature,
            field_3d_b=self.sfield_3d_compression,
            field_name_a="<sfield_3d_curvature>",
            field_name_b="<sfield_3d_compression>",
        )


def compute_magnetic_curvature_decomposed_fields(
    *,
    vfield_3d_u: fields.VectorField_3D,
    uvfield_3d_tangent: fields.UnitVectorField_3D,
    uvfield_3d_normal: fields.UnitVectorField_3D,
    grad_order: int = 2,
) -> MagneticCurvatureFields_3D:
    """
    Field-level wrapper for magnetic curvature, stretching, compression.

    Uses:
        curvature   = n_i n_j d_i u_j
        stretching  = t_i t_j d_i u_j
        compression = d_i u_i
    """
    fields.ensure_3d_vfield(
        vfield_3d=vfield_3d_u,
        param_name="<vfield_3d_u>",
    )
    fields.ensure_3d_uvfield(
        uvfield_3d=uvfield_3d_tangent,
        param_name="<uvfield_3d_tangent>",
    )
    fields.ensure_3d_uvfield(
        uvfield_3d=uvfield_3d_normal,
        param_name="<uvfield_3d_normal>",
    )
    fields.ensure_same_3d_field_udomains(
        field_3d_a=vfield_3d_u,
        field_3d_b=uvfield_3d_tangent,
        field_name_a="<vfield_3d_u>",
        field_name_b="<uvfield_3d_tangent>",
    )
    fields.ensure_same_3d_field_udomains(
        field_3d_a=vfield_3d_u,
        field_3d_b=uvfield_3d_normal,
        field_name_a="<vfield_3d_u>",
        field_name_b="<uvfield_3d_normal>",
    )
    type_manager.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    varray_3d_u = fields.extract_3d_varray(
        vfield_3d=vfield_3d_u,
        param_name="<vfield_3d_u>",
    )
    uvarray_3d_tangent = fields.extract_3d_varray(
        vfield_3d=uvfield_3d_tangent,
        param_name="<uvfield_3d_tangent>",
    )
    uvarray_3d_normal = fields.extract_3d_varray(
        vfield_3d=uvfield_3d_normal,
        param_name="<uvfield_3d_normal>",
    )
    udomain_3d = vfield_3d_u.udomain
    sim_time = vfield_3d_u.sim_time
    mc_3d_farrays = _decompose_farrays.compute_magnetic_curvature_farrays(
        varray_3d_u=varray_3d_u,
        uvarray_3d_tangent=uvarray_3d_tangent,
        uvarray_3d_normal=uvarray_3d_normal,
        cell_widths_3d=udomain_3d.cell_widths,
        grad_order=grad_order,
    )
    sfield_3d_curvature = fields.ScalarField_3D.from_3d_sarray(
        sarray_3d=mc_3d_farrays.sarray_3d_curvature,
        udomain_3d=udomain_3d,
        field_label=r"$n_i n_j d_i u_j$",
        sim_time=sim_time,
    )
    sfield_3d_stretching = fields.ScalarField_3D.from_3d_sarray(
        sarray_3d=mc_3d_farrays.sarray_3d_stretching,
        udomain_3d=udomain_3d,
        field_label=r"$t_i t_j d_i u_j$",
        sim_time=sim_time,
    )
    sfield_3d_compression = fields.ScalarField_3D.from_3d_sarray(
        sarray_3d=mc_3d_farrays.sarray_3d_compression,
        udomain_3d=udomain_3d,
        field_label=r"$d_i u_i$",
        sim_time=sim_time,
    )
    return MagneticCurvatureFields_3D(
        sfield_3d_curvature=sfield_3d_curvature,
        sfield_3d_stretching=sfield_3d_stretching,
        sfield_3d_compression=sfield_3d_compression,
    )


##
## --- LORENTZ FORCE TERMS
##


@dataclass(frozen=True)
class LorentzForceFields_3D:
    vfield_3d_lorentz: fields.VectorField_3D
    vfield_3d_tension: fields.VectorField_3D
    vfield_3d_gradP_perp: fields.VectorField_3D

    def __post_init__(
        self,
    ) -> None:
        fields.ensure_3d_vfield(
            vfield_3d=self.vfield_3d_lorentz,
            param_name="<vfield_3d_lorentz>",
        )
        fields.ensure_3d_vfield(
            vfield_3d=self.vfield_3d_tension,
            param_name="<vfield_3d_tension>",
        )
        fields.ensure_3d_vfield(
            vfield_3d=self.vfield_3d_gradP_perp,
            param_name="<vfield_3d_gradP_perp>",
        )
        fields.ensure_same_3d_field_shape_and_udomains(
            field_3d_a=self.vfield_3d_lorentz,
            field_3d_b=self.vfield_3d_tension,
            field_name_a="<vfield_3d_lorentz>",
            field_name_b="<vfield_3d_tension>",
        )
        fields.ensure_same_3d_field_shape_and_udomains(
            field_3d_a=self.vfield_3d_lorentz,
            field_3d_b=self.vfield_3d_gradP_perp,
            field_name_a="<vfield_3d_lorentz>",
            field_name_b="<vfield_3d_gradP_perp>",
        )


def compute_lorentz_force_decomposed_fields(
    vfield_3d_b: fields.VectorField_3D,
    *,
    grad_order: int = 2,
) -> LorentzForceFields_3D:
    """Field-level Lorentz force decomposition wrapper."""
    fields.ensure_3d_vfield(
        vfield_3d=vfield_3d_b,
        param_name="<vfield_3d_b>",
    )
    type_manager.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    udomain_3d = vfield_3d_b.udomain
    domains.ensure_3d_udomain(
        udomain_3d=udomain_3d,
        param_name="<udomain_3d>",
    )
    varray_3d_b = fields.extract_3d_varray(
        vfield_3d=vfield_3d_b,
        param_name="<vfield_3d_b>",
    )
    sim_time = vfield_3d_b.sim_time
    lf_3d_farrays = _decompose_farrays.compute_lorentz_force_farrays(
        varray_3d_b=varray_3d_b,
        cell_widths_3d=udomain_3d.cell_widths,
        grad_order=grad_order,
    )
    vfield_3d_lorentz = fields.VectorField_3D.from_3d_varray(
        varray_3d=lf_3d_farrays.varray_3d_lorentz,
        udomain_3d=udomain_3d,
        field_label=r"$L_i$",
        sim_time=sim_time,
    )
    vfield_3d_tension = fields.VectorField_3D.from_3d_varray(
        varray_3d=lf_3d_farrays.varray_3d_tension,
        udomain_3d=udomain_3d,
        field_label=r"$b_k b_k \kappa_i$",
        sim_time=sim_time,
    )
    vfield_3d_gradP_perp = fields.VectorField_3D.from_3d_varray(
        varray_3d=lf_3d_farrays.varray_3d_gradP_perp,
        udomain_3d=udomain_3d,
        field_label=r"$[d_i (b_k b_k / 2)]_\perp$",
        sim_time=sim_time,
    )
    return LorentzForceFields_3D(
        vfield_3d_lorentz=vfield_3d_lorentz,
        vfield_3d_tension=vfield_3d_tension,
        vfield_3d_gradP_perp=vfield_3d_gradP_perp,
    )


## } MODULE
