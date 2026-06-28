## { MODULE

##
## === DEPENDENCIES
##

## local
from jormi.ww_arrays.farrays_3d import (
    decompose_farrays,
    farray_operators,
)
from jormi.ww_fields.fields_3d import (
    field_operators,
    field_models,
)
from jormi.ww_validation import validate_types

##
## === MAGNETIC FIELD ENERGY
##


def compute_magnetic_energy_density_sfield(
    magnetic_vfield_3d: field_models.VectorField_3D,
    *,
    energy_prefactor: float = 0.5,
    field_name: str,
    latex_label: str,
) -> field_models.ScalarField_3D:
    """Compute magnetic energy density from a 3D magnetic field (proportional to b_i b_i)."""
    validate_types.ensure_finite_float(
        param=energy_prefactor,
        param_name="<energy_prefactor>",
        allow_none=False,
    )
    field_models.ensure_3d_vfield(
        vfield_3d=magnetic_vfield_3d,
        param_name="<magnetic_vfield_3d>",
    )
    b_varray_3d = field_models.extract_3d_varray(
        vfield_3d=magnetic_vfield_3d,
        param_name="<magnetic_vfield_3d>",
    )
    udomain_3d = magnetic_vfield_3d.udomain
    sim_time = magnetic_vfield_3d.sim_time
    b2_sarray_3d = farray_operators.compute_sum_of_varray_comps_squared(
        varray_3d=b_varray_3d,
    )
    farray_operators.scale_sarray_inplace(
        sarray_3d=b2_sarray_3d,
        scale=energy_prefactor,
    )
    return field_models.ScalarField_3D.from_3d_sarray(
        sarray_3d=b2_sarray_3d,
        udomain_3d=udomain_3d,
        field_name=field_name,
        latex_label=latex_label,
        sim_time=sim_time,
    )


def compute_total_magnetic_energy_value(
    magnetic_vfield_3d: field_models.VectorField_3D,
    *,
    energy_prefactor: float = 0.5,
) -> float:
    """Compute total magnetic energy as the volume integral of the energy density."""
    e_mag_sfield_3d = compute_magnetic_energy_density_sfield(
        magnetic_vfield_3d=magnetic_vfield_3d,
        energy_prefactor=energy_prefactor,
        field_name="magnetic_energy",
        latex_label=r"E_\mathrm{mag}",
    )
    return field_operators.compute_sfield_volume_integral(
        sfield_3d=e_mag_sfield_3d,
    )


##
## === KINETIC ENERGY DISSIPATION FUNCTION
##


def compute_kinetic_dissipation_vfield(
    velocity_vfield_3d: field_models.VectorField_3D,
    *,
    grad_order: int = 2,
) -> field_models.VectorField_3D:
    """
    Compute d_j S_ji for a 3D velocity field v_i, where

        S_ij = 0.5 * (d_i v_j + d_j v_i) - (1/3) delta_ij (d_k v_k)
    """
    field_models.ensure_3d_vfield(
        vfield_3d=velocity_vfield_3d,
        param_name="<velocity_vfield_3d>",
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
    udomain_3d = velocity_vfield_3d.udomain
    sim_time = velocity_vfield_3d.sim_time
    kinetic_dissipation_varray_3d = farray_operators.compute_varray_kinetic_dissipation(
        v_varray_3d=v_varray_3d,
        cell_widths_3d=udomain_3d.cell_widths,
        grad_order=grad_order,
    )
    return field_models.VectorField_3D.from_3d_varray(
        varray_3d=kinetic_dissipation_varray_3d,
        udomain_3d=udomain_3d,
        field_name="kinetic_dissipation",
        latex_label=r"\partial_j \mathcal{S}_{ji}",
        sim_time=sim_time,
    )


##
## === MAGNETIC FIELD LINE CURVATURE
##


def compute_curvature_sfield(
    vfield_3d: field_models.VectorField_3D,
    *,
    grad_order: int = 2,
) -> field_models.ScalarField_3D:
    """Compute field line curvature magnitude sqrt(kappa_i kappa_i) from a 3D vector field."""
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
    varray_3d = field_models.extract_3d_varray(
        vfield_3d=vfield_3d,
        param_name="<vfield_3d>",
    )
    udomain_3d = vfield_3d.udomain
    sim_time = vfield_3d.sim_time
    kappa_sarray_3d = decompose_farrays.compute_curvature_sarray(
        varray_3d=varray_3d,
        cell_widths_3d=udomain_3d.cell_widths,
        grad_order=grad_order,
    )
    return field_models.ScalarField_3D.from_3d_sarray(
        sarray_3d=kappa_sarray_3d,
        udomain_3d=udomain_3d,
        field_name="curvature_magnitude",
        latex_label=r"|\vec{\kappa}|",
        sim_time=sim_time,
    )


## } MODULE
