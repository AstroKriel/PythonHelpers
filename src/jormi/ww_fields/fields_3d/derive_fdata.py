## { MODULE

##
## === DEPENDENCIES
##

from dataclasses import dataclass

from jormi.ww_fields.fields_3d import (
    fdata_types,
    derive_farrays,
)


##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class MagneticCurvatureFData_3D:
    sdata_3d_curvature: fdata_types.ScalarFieldData_3D
    sdata_3d_stretching: fdata_types.ScalarFieldData_3D
    sdata_3d_compression: fdata_types.ScalarFieldData_3D


@dataclass(frozen=True)
class LorentzForceFData_3D:
    vdata_3d_lorentz: fdata_types.VectorFieldData_3D
    vdata_3d_tension: fdata_types.VectorFieldData_3D
    vdata_3d_gradP_perp: fdata_types.VectorFieldData_3D


##
## === MAGNETIC CURVATURE ON FDATA
##


def compute_magnetic_curvature_terms(
    *,
    vdata_3d_u: fdata_types.VectorFieldData_3D,
    uvdata_3d_tangent: fdata_types.VectorFieldData_3D,
    uvdata_3d_normal: fdata_types.VectorFieldData_3D,
    cell_widths: tuple[float, float, float],
    grad_order: int = 2,
) -> MagneticCurvatureFData_3D:
    """Compute curvature, stretching, compression from 3D VectorFieldData_3D."""
    fdata_types.ensure_3d_vdata(
        vdata_3d=vdata_3d_u,
        param_name="<vdata_3d_u>",
    )
    fdata_types.ensure_3d_vdata(
        vdata_3d=uvdata_3d_tangent,
        param_name="<uvdata_3d_tangent>",
    )
    fdata_types.ensure_3d_vdata(
        vdata_3d=uvdata_3d_normal,
        param_name="<uvdata_3d_normal>",
    )
    varray_3d_u = fdata_types.as_3d_varray(
        vdata_3d=vdata_3d_u,
        param_name="<vdata_3d_u>",
    )
    uvarray_3d_tangent = fdata_types.as_3d_varray(
        vdata_3d=uvdata_3d_tangent,
        param_name="<uvdata_3d_tangent>",
    )
    uvarray_3d_normal = fdata_types.as_3d_varray(
        vdata_3d=uvdata_3d_normal,
        param_name="<uvdata_3d_normal>",
    )
    mct_3d_farrays = derive_farrays.compute_magnetic_curvature_terms(
        varray_3d_u=varray_3d_u,
        uvarray_3d_tangent=uvarray_3d_tangent,
        uvarray_3d_normal=uvarray_3d_normal,
        cell_widths=cell_widths,
        r2tarray_3d_gradu=None,
        grad_order=grad_order,
    )
    sdata_3d_curvature = fdata_types.ScalarFieldData_3D(
        farray=mct_3d_farrays.sarray_3d_curvature,
        param_name="<mc.sdata_3d_curvature>",
    )
    sdata_3d_stretching = fdata_types.ScalarFieldData_3D(
        farray=mct_3d_farrays.sarray_3d_stretching,
        param_name="<mc.sdata_3d_stretching>",
    )
    sdata_3d_compression = fdata_types.ScalarFieldData_3D(
        farray=mct_3d_farrays.sarray_3d_compression,
        param_name="<mc.sdata_3d_compression>",
    )
    return MagneticCurvatureFData_3D(
        sdata_3d_curvature=sdata_3d_curvature,
        sdata_3d_stretching=sdata_3d_stretching,
        sdata_3d_compression=sdata_3d_compression,
    )


##
## === LORENTZ FORCE ON FDATA
##


def compute_lorentz_force_terms(
    *,
    vdata_3d_b: fdata_types.VectorFieldData_3D,
    cell_widths: tuple[float, float, float],
    grad_order: int = 2,
) -> LorentzForceFData_3D:
    """Compute Lorentz, tension, and perpendicular pressure from 3D B-field fdata."""
    fdata_types.ensure_3d_vdata(
        vdata_3d=vdata_3d_b,
        param_name="<vdata_3d_b>",
    )
    varray_3d_b = fdata_types.as_3d_varray(
        vdata_3d=vdata_3d_b,
        param_name="<vdata_3d_b>",
    )
    lft_3d_farrays = derive_farrays.compute_lorentz_force_terms(
        varray_3d_b=varray_3d_b,
        cell_widths=cell_widths,
        grad_order=grad_order,
    )
    vdata_3d_lorentz = fdata_types.VectorFieldData_3D(
        farray=lft_3d_farrays.varray_3d_lorentz,
        param_name="<lf.vdata_3d_lorentz>",
    )
    vdata_3d_tension = fdata_types.VectorFieldData_3D(
        farray=lft_3d_farrays.varray_3d_tension,
        param_name="<lf.vdata_3d_tension>",
    )
    vdata_3d_gradP_perp = fdata_types.VectorFieldData_3D(
        farray=lft_3d_farrays.varray_3d_gradP_perp,
        param_name="<lf.vdata_3d_gradP_perp>",
    )
    return LorentzForceFData_3D(
        vdata_3d_lorentz=vdata_3d_lorentz,
        vdata_3d_tension=vdata_3d_tension,
        vdata_3d_gradP_perp=vdata_3d_gradP_perp,
    )


##
## === KINETIC DISSIPATION ON FDATA
##


def compute_kinetic_dissipation(
    *,
    vdata_3d_u: fdata_types.VectorFieldData_3D,
    cell_widths: tuple[float, float, float],
    grad_order: int = 2,
) -> fdata_types.VectorFieldData_3D:
    """Compute d_j S_ji for a 3D VectorFieldData_3D."""
    fdata_types.ensure_3d_vdata(
        vdata_3d=vdata_3d_u,
        param_name="<vdata_3d_u>",
    )
    varray_3d_u = fdata_types.as_3d_varray(
        vdata_3d=vdata_3d_u,
        param_name="<vdata_3d_u>",
    )
    varray_3d_df = derive_farrays.compute_kinetic_dissipation(
        varray_3d_u=varray_3d_u,
        cell_widths=cell_widths,
        grad_order=grad_order,
    )
    return fdata_types.VectorFieldData_3D(
        farray=varray_3d_df,
        param_name="<dissipation.vdata_3d_df>",
    )


## } MODULE
