## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass

## local
from jormi.ww_arrays.farrays_3d import decompose_farrays as _decompose_farrays
from jormi.ww_fields.fields_3d import (
    domain_models,
    field_models,
)
from jormi.ww_validation import validate_types

##
## === HELMHOLTZ DECOMPOSITION
##


@dataclass(frozen=True)
class HelmholtzDecomposedFields_3D:
    div_vfield_3d: field_models.VectorField_3D
    sol_vfield_3d: field_models.VectorField_3D
    bulk_vfield_3d: field_models.VectorField_3D

    def __post_init__(
        self,
    ) -> None:
        ## validate each field independently
        field_models.ensure_3d_vfield(
            vfield_3d=self.div_vfield_3d,
            param_name="<div_vfield_3d>",
        )
        field_models.ensure_3d_vfield(
            vfield_3d=self.sol_vfield_3d,
            param_name="<sol_vfield_3d>",
        )
        field_models.ensure_3d_vfield(
            vfield_3d=self.bulk_vfield_3d,
            param_name="<bulk_vfield_3d>",
        )
        ## validate shared field geometry and domains
        field_models.ensure_same_3d_field_shape_and_udomains(
            field_3d_a=self.div_vfield_3d,
            field_3d_b=self.sol_vfield_3d,
            field_name_a="<div_vfield_3d>",
            field_name_b="<sol_vfield_3d>",
        )
        field_models.ensure_same_3d_field_shape_and_udomains(
            field_3d_a=self.div_vfield_3d,
            field_3d_b=self.bulk_vfield_3d,
            field_name_a="<div_vfield_3d>",
            field_name_b="<bulk_vfield_3d>",
        )


def compute_helmholtz_decomposed_fields(
    vfield_3d: field_models.VectorField_3D,
) -> HelmholtzDecomposedFields_3D:
    """Field-level Helmholtz decomposition wrapper."""
    field_models.ensure_3d_vfield(
        vfield_3d=vfield_3d,
        param_name="<vfield_3d>",
    )
    udomain_3d = vfield_3d.udomain
    domain_models.ensure_3d_periodic_udomain(
        udomain_3d=udomain_3d,
        param_name="<udomain_3d>",
    )
    varray_3d = field_models.extract_3d_varray(
        vfield_3d=vfield_3d,
        param_name="<vfield_3d>",
    )
    sim_time = vfield_3d.sim_time
    helmholtz_farrays_3d = _decompose_farrays.compute_helmholtz_decomposed_farrays(
        varray_3d=varray_3d,
        resolution=udomain_3d.resolution,
        cell_widths_3d=udomain_3d.cell_widths,
    )
    div_vfield_3d = field_models.VectorField_3D.from_3d_varray(
        varray_3d=helmholtz_farrays_3d.div_varray_3d,
        udomain_3d=udomain_3d,
        field_name="div_component",
        latex_label=r"\vec{f}_\mathrm{div}",
        sim_time=sim_time,
    )
    sol_vfield_3d = field_models.VectorField_3D.from_3d_varray(
        varray_3d=helmholtz_farrays_3d.sol_varray_3d,
        udomain_3d=udomain_3d,
        field_name="sol_component",
        latex_label=r"\vec{f}_\mathrm{sol}",
        sim_time=sim_time,
    )
    bulk_vfield_3d = field_models.VectorField_3D.from_3d_varray(
        varray_3d=helmholtz_farrays_3d.bulk_varray_3d,
        udomain_3d=udomain_3d,
        field_name="bulk_component",
        latex_label=r"\vec{f}_\mathrm{bulk}",
        sim_time=sim_time,
    )
    return HelmholtzDecomposedFields_3D(
        div_vfield_3d=div_vfield_3d,
        sol_vfield_3d=sol_vfield_3d,
        bulk_vfield_3d=bulk_vfield_3d,
    )


##
## --- TANGENT NORMAL BINORMAL BASIS
##


@dataclass(frozen=True)
class TNBDecomposedFields_3D:
    tangent_uvfield_3d: field_models.UnitVectorField_3D
    normal_uvfield_3d: field_models.UnitVectorField_3D
    binormal_uvfield_3d: field_models.UnitVectorField_3D
    curvature_sfield_3d: field_models.ScalarField_3D

    def __post_init__(
        self,
    ) -> None:
        ## validate each TNB field independently
        field_models.ensure_3d_uvfield(
            uvfield_3d=self.tangent_uvfield_3d,
            param_name="<tangent_uvfield_3d>",
        )
        field_models.ensure_3d_uvfield(
            uvfield_3d=self.normal_uvfield_3d,
            param_name="<normal_uvfield_3d>",
        )
        field_models.ensure_3d_uvfield(
            uvfield_3d=self.binormal_uvfield_3d,
            param_name="<binormal_uvfield_3d>",
        )
        field_models.ensure_3d_sfield(
            sfield_3d=self.curvature_sfield_3d,
            param_name="<curvature_sfield_3d>",
        )
        ## validate shared field geometry within the decomposition
        field_models.ensure_same_3d_field_shape_and_udomains(
            field_3d_a=self.tangent_uvfield_3d,
            field_3d_b=self.normal_uvfield_3d,
            field_name_a="<tangent_uvfield_3d>",
            field_name_b="<normal_uvfield_3d>",
        )
        field_models.ensure_same_3d_field_shape_and_udomains(
            field_3d_a=self.tangent_uvfield_3d,
            field_3d_b=self.binormal_uvfield_3d,
            field_name_a="<tangent_uvfield_3d>",
            field_name_b="<binormal_uvfield_3d>",
        )
        field_models.ensure_same_3d_field_udomains(
            field_3d_a=self.tangent_uvfield_3d,
            field_3d_b=self.curvature_sfield_3d,
            field_name_a="<tangent_uvfield_3d>",
            field_name_b="<curvature_sfield_3d>",
        )


def compute_tnb_decomposed_fields(
    vfield_3d: field_models.VectorField_3D,
    *,
    grad_order: int = 2,
) -> TNBDecomposedFields_3D:
    """Field-level TNB decomposition wrapper."""
    field_models.ensure_3d_vfield(
        vfield_3d=vfield_3d,
        param_name="<vfield_3d>",
    )
    validate_types.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    udomain_3d = vfield_3d.udomain
    varray_3d = field_models.extract_3d_varray(
        vfield_3d=vfield_3d,
        param_name="<vfield_3d>",
    )
    sim_time = vfield_3d.sim_time
    tnb_farrays_3d = _decompose_farrays.compute_tnb_farrays_3d(
        varray_3d=varray_3d,
        cell_widths_3d=udomain_3d.cell_widths,
        grad_order=grad_order,
    )
    tangent_vfield_3d = field_models.VectorField_3D.from_3d_varray(
        varray_3d=tnb_farrays_3d.tangent_uvarray_3d,
        udomain_3d=udomain_3d,
        field_name="tangent",
        latex_label=r"\vec{t}",
        sim_time=sim_time,
    )
    normal_vfield_3d = field_models.VectorField_3D.from_3d_varray(
        varray_3d=tnb_farrays_3d.normal_uvarray_3d,
        udomain_3d=udomain_3d,
        field_name="normal",
        latex_label=r"\vec{n}",
        sim_time=sim_time,
    )
    binormal_vfield_3d = field_models.VectorField_3D.from_3d_varray(
        varray_3d=tnb_farrays_3d.binormal_uvarray_3d,
        udomain_3d=udomain_3d,
        field_name="binormal",
        latex_label=r"\hat{b}",
        sim_time=sim_time,
    )
    curvature_sfield_3d = field_models.ScalarField_3D.from_3d_sarray(
        sarray_3d=tnb_farrays_3d.curvature_sarray_3d,
        udomain_3d=udomain_3d,
        field_name="curvature_magnitude",
        latex_label=r"|\vec{\kappa}|",
        sim_time=sim_time,
    )
    tangent_uvfield_3d = field_models.as_3d_uvfield(
        vfield_3d=tangent_vfield_3d,
    )
    normal_uvfield_3d = field_models.as_3d_uvfield(
        vfield_3d=normal_vfield_3d,
    )
    binormal_uvfield_3d = field_models.as_3d_uvfield(
        vfield_3d=binormal_vfield_3d,
    )
    return TNBDecomposedFields_3D(
        tangent_uvfield_3d=tangent_uvfield_3d,
        normal_uvfield_3d=normal_uvfield_3d,
        binormal_uvfield_3d=binormal_uvfield_3d,
        curvature_sfield_3d=curvature_sfield_3d,
    )


##
## --- MAGNETIC DECOMPOSITION
##


@dataclass(frozen=True)
class MagneticCurvatureFields_3D:
    curvature_sfield_3d: field_models.ScalarField_3D
    stretching_sfield_3d: field_models.ScalarField_3D
    compression_sfield_3d: field_models.ScalarField_3D

    def __post_init__(
        self,
    ) -> None:
        ## validate each scalar field independently
        field_models.ensure_3d_sfield(
            sfield_3d=self.curvature_sfield_3d,
            param_name="<curvature_sfield_3d>",
        )
        field_models.ensure_3d_sfield(
            sfield_3d=self.stretching_sfield_3d,
            param_name="<stretching_sfield_3d>",
        )
        field_models.ensure_3d_sfield(
            sfield_3d=self.compression_sfield_3d,
            param_name="<compression_sfield_3d>",
        )
        ## validate shared field geometry and domains
        field_models.ensure_same_3d_field_shape_and_udomains(
            field_3d_a=self.curvature_sfield_3d,
            field_3d_b=self.stretching_sfield_3d,
            field_name_a="<curvature_sfield_3d>",
            field_name_b="<stretching_sfield_3d>",
        )
        field_models.ensure_same_3d_field_shape_and_udomains(
            field_3d_a=self.curvature_sfield_3d,
            field_3d_b=self.compression_sfield_3d,
            field_name_a="<curvature_sfield_3d>",
            field_name_b="<compression_sfield_3d>",
        )


def compute_magnetic_curvature_decomposed_fields(
    *,
    velocity_vfield_3d: field_models.VectorField_3D,
    tangent_uvfield_3d: field_models.UnitVectorField_3D,
    normal_uvfield_3d: field_models.UnitVectorField_3D,
    grad_order: int = 2,
) -> MagneticCurvatureFields_3D:
    """
    Field-level wrapper for magnetic curvature, stretching, compression.

    Uses:
        curvature   = n_i n_j d_i v_j
        stretching  = t_i t_j d_i v_j
        compression = d_i v_i
    """
    field_models.ensure_3d_vfield(
        vfield_3d=velocity_vfield_3d,
        param_name="<velocity_vfield_3d>",
    )
    field_models.ensure_3d_uvfield(
        uvfield_3d=tangent_uvfield_3d,
        param_name="<tangent_uvfield_3d>",
    )
    field_models.ensure_3d_uvfield(
        uvfield_3d=normal_uvfield_3d,
        param_name="<normal_uvfield_3d>",
    )
    field_models.ensure_same_3d_field_udomains(
        field_3d_a=velocity_vfield_3d,
        field_3d_b=tangent_uvfield_3d,
        field_name_a="<velocity_vfield_3d>",
        field_name_b="<tangent_uvfield_3d>",
    )
    field_models.ensure_same_3d_field_udomains(
        field_3d_a=velocity_vfield_3d,
        field_3d_b=normal_uvfield_3d,
        field_name_a="<velocity_vfield_3d>",
        field_name_b="<normal_uvfield_3d>",
    )
    validate_types.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    v_varray_3d = field_models.extract_3d_varray(
        vfield_3d=velocity_vfield_3d,
        param_name="<velocity_vfield_3d>",
    )
    tangent_uvarray_3d = field_models.extract_3d_varray(
        vfield_3d=tangent_uvfield_3d,
        param_name="<tangent_uvfield_3d>",
    )
    normal_uvarray_3d = field_models.extract_3d_varray(
        vfield_3d=normal_uvfield_3d,
        param_name="<normal_uvfield_3d>",
    )
    udomain_3d = velocity_vfield_3d.udomain
    sim_time = velocity_vfield_3d.sim_time
    magnetic_curvature_farrays_3d = _decompose_farrays.compute_magnetic_curvature_farrays_3d(
        v_varray_3d=v_varray_3d,
        tangent_uvarray_3d=tangent_uvarray_3d,
        normal_uvarray_3d=normal_uvarray_3d,
        cell_widths_3d=udomain_3d.cell_widths,
        grad_order=grad_order,
    )
    curvature_sfield_3d = field_models.ScalarField_3D.from_3d_sarray(
        sarray_3d=magnetic_curvature_farrays_3d.curvature_sarray_3d,
        udomain_3d=udomain_3d,
        field_name="magnetic_curvature",
        latex_label=r"n_i n_j \partial_i v_j",
        sim_time=sim_time,
    )
    stretching_sfield_3d = field_models.ScalarField_3D.from_3d_sarray(
        sarray_3d=magnetic_curvature_farrays_3d.stretching_sarray_3d,
        udomain_3d=udomain_3d,
        field_name="magnetic_stretching",
        latex_label=r"t_i t_j \partial_i v_j",
        sim_time=sim_time,
    )
    compression_sfield_3d = field_models.ScalarField_3D.from_3d_sarray(
        sarray_3d=magnetic_curvature_farrays_3d.compression_sarray_3d,
        udomain_3d=udomain_3d,
        field_name="magnetic_compression",
        latex_label=r"\partial_i v_i",
        sim_time=sim_time,
    )
    return MagneticCurvatureFields_3D(
        curvature_sfield_3d=curvature_sfield_3d,
        stretching_sfield_3d=stretching_sfield_3d,
        compression_sfield_3d=compression_sfield_3d,
    )


##
## --- LORENTZ FORCE TERMS
##


@dataclass(frozen=True)
class LorentzForceFields_3D:
    lorentz_vfield_3d: field_models.VectorField_3D
    tension_vfield_3d: field_models.VectorField_3D
    grad_p_perp_vfield_3d: field_models.VectorField_3D

    def __post_init__(
        self,
    ) -> None:
        ## validate each force field independently
        field_models.ensure_3d_vfield(
            vfield_3d=self.lorentz_vfield_3d,
            param_name="<lorentz_vfield_3d>",
        )
        field_models.ensure_3d_vfield(
            vfield_3d=self.tension_vfield_3d,
            param_name="<tension_vfield_3d>",
        )
        field_models.ensure_3d_vfield(
            vfield_3d=self.grad_p_perp_vfield_3d,
            param_name="<grad_p_perp_vfield_3d>",
        )
        ## validate shared field geometry and domains
        field_models.ensure_same_3d_field_shape_and_udomains(
            field_3d_a=self.lorentz_vfield_3d,
            field_3d_b=self.tension_vfield_3d,
            field_name_a="<lorentz_vfield_3d>",
            field_name_b="<tension_vfield_3d>",
        )
        field_models.ensure_same_3d_field_shape_and_udomains(
            field_3d_a=self.lorentz_vfield_3d,
            field_3d_b=self.grad_p_perp_vfield_3d,
            field_name_a="<lorentz_vfield_3d>",
            field_name_b="<grad_p_perp_vfield_3d>",
        )


def compute_lorentz_force_decomposed_fields(
    magnetic_vfield_3d: field_models.VectorField_3D,
    *,
    grad_order: int = 2,
) -> LorentzForceFields_3D:
    """Field-level Lorentz force decomposition wrapper."""
    field_models.ensure_3d_vfield(
        vfield_3d=magnetic_vfield_3d,
        param_name="<magnetic_vfield_3d>",
    )
    validate_types.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    udomain_3d = magnetic_vfield_3d.udomain
    domain_models.ensure_3d_udomain(
        udomain_3d=udomain_3d,
        param_name="<udomain_3d>",
    )
    b_varray_3d = field_models.extract_3d_varray(
        vfield_3d=magnetic_vfield_3d,
        param_name="<magnetic_vfield_3d>",
    )
    sim_time = magnetic_vfield_3d.sim_time
    lorentz_force_farrays_3d = _decompose_farrays.compute_lorentz_force_farrays_3d(
        b_varray_3d=b_varray_3d,
        cell_widths_3d=udomain_3d.cell_widths,
        grad_order=grad_order,
    )
    lorentz_vfield_3d = field_models.VectorField_3D.from_3d_varray(
        varray_3d=lorentz_force_farrays_3d.lorentz_varray_3d,
        udomain_3d=udomain_3d,
        field_name="lorentz_force",
        latex_label=r"(\nabla\times\vec{b})\times\vec{b}",
        sim_time=sim_time,
    )
    tension_vfield_3d = field_models.VectorField_3D.from_3d_varray(
        varray_3d=lorentz_force_farrays_3d.tension_varray_3d,
        udomain_3d=udomain_3d,
        field_name="magnetic_tension",
        latex_label=r"b^2 \vec{\kappa}",
        sim_time=sim_time,
    )
    grad_p_perp_vfield_3d = field_models.VectorField_3D.from_3d_varray(
        varray_3d=lorentz_force_farrays_3d.grad_p_perp_varray_3d,
        udomain_3d=udomain_3d,
        field_name="magnetic_pressure_gradient",
        latex_label=r"[\partial_i (b_k b_k / 2)]_\perp",
        sim_time=sim_time,
    )
    return LorentzForceFields_3D(
        lorentz_vfield_3d=lorentz_vfield_3d,
        tension_vfield_3d=tension_vfield_3d,
        grad_p_perp_vfield_3d=grad_p_perp_vfield_3d,
    )


## } MODULE
