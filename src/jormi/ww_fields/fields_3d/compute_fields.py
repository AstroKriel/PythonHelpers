## { MODULE

##
## === DEPENDENCIES
##

import numpy

from jormi.ww_types import type_manager
from jormi.ww_fields.fields_3d import (
    _finite_difference_sarrays,
    _farray_operators,
    _fdata_types,
    field_types,
    field_operators,
)

##
## === MAGNETIC FIELD ENERGY
##


def compute_magnetic_energy_density_sfield(
    vfield_3d_b: field_types.VectorField_3D,
    *,
    energy_prefactor: float = 0.5,
    field_label: str = "E_mag",
) -> field_types.ScalarField_3D:
    """Compute magnetic energy density from a 3D magnetic field (proportional to b_i b_i)."""
    type_manager.ensure_finite_float(
        param=energy_prefactor,
        param_name="<energy_prefactor>",
        allow_none=False,
    )
    varray_3d_b = field_types.extract_3d_varray(vfield_3d_b)
    udomain_3d = vfield_3d_b.udomain
    sim_time = vfield_3d_b.sim_time
    sarray_3d_b2 = _farray_operators.sum_of_varray_comps_squared(varray_3d_b)
    _farray_operators.scale_sarray_inplace(
        sarray_3d=sarray_3d_b2,
        scale=energy_prefactor,
    )
    return field_types.ScalarField_3D.from_3d_sarray(
        sarray_3d=sarray_3d_b2,
        udomain_3d=udomain_3d,
        field_label=field_label,
        sim_time=sim_time,
    )


def compute_total_magnetic_energy(
    vfield_3d_b: field_types.VectorField_3D,
    *,
    energy_prefactor: float = 0.5,
) -> float:
    """Compute total magnetic energy as the volume integral of the energy density."""
    sfield_3d_Emag = compute_magnetic_energy_density_sfield(
        vfield_3d_b=vfield_3d_b,
        energy_prefactor=energy_prefactor,
    )
    return field_operators.compute_sfield_volume_integral(
        sfield_3d=sfield_3d_Emag,
    )


##
## === KINETIC ENERGY DISSIPATION FUNCTION
##


def _compute_kinetic_dissipation_varray(
    *,
    varray_3d_u: numpy.ndarray,
    cell_widths_3d: tuple[float, float, float],
    grad_order: int = 2,
) -> numpy.ndarray:
    """
    Compute d_j S_ji for a 3D velocity varray u_j, where

        S_ij = 0.5 * (d_i u_j + d_j u_i) - (1/3) delta_ij (d_k u_k)
    """
    _fdata_types.ensure_3d_varray(varray_3d_u)
    if varray_3d_u.shape[0] != 3:
        raise ValueError(
            "`<varray_3d_u>` must have shape (3, Nx, Ny, Nz);"
            f" got shape={varray_3d_u.shape}.",
        )
    num_cells_x, num_cells_y, num_cells_z = varray_3d_u.shape[1:]
    dtype = varray_3d_u.dtype
    _farray_operators._validate_3d_cell_widths(cell_widths_3d)
    type_manager.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    ## d_i u_j
    r2tarray_3d_gradu = _farray_operators.compute_varray_grad(
        varray_3d=varray_3d_u,
        cell_widths_3d=cell_widths_3d,
        r2tarray_3d_gradf=None,
        grad_order=grad_order,
    )
    sarray_3d_divu = numpy.trace(
        r2tarray_3d_gradu,
        axis1=0,
        axis2=1,
    )
    ## S_ij = 0.5 * (d_i u_j + d_j u_i) - (1/3) delta_ij (d_k u_k)
    r2tarray_3d_sym = 0.5 * r2tarray_3d_gradu + numpy.transpose(
        r2tarray_3d_gradu,
        axes=(1, 0, 2, 3, 4),
    )
    identity_matrix = numpy.eye(3, dtype=dtype)
    r2tarray_3d_bulk = numpy.einsum(
        "ij,xyz->jixyz",
        identity_matrix,
        sarray_3d_divu,
        optimize=True,
    )
    r2tarray_3d_S = r2tarray_3d_sym - (1.0 / 3.0) * r2tarray_3d_bulk
    ## d_j S_ji
    nabla = _finite_difference_sarrays.get_grad_fn(grad_order)
    cell_width_x, cell_width_y, cell_width_z = cell_widths_3d
    varray_3d_df = _fdata_types.ensure_farray_metadata(
        farray_shape=(3, num_cells_x, num_cells_y, num_cells_z),
        farray=None,
        dtype=dtype,
    )
    for comp_i in range(3):
        varray_3d_df[comp_i, ...] = nabla(
            sarray_3d=r2tarray_3d_S[0, comp_i],
            cell_width=cell_width_x,
            grad_axis=0,
        )
        numpy.add(
            varray_3d_df[comp_i, ...],
            nabla(
                sarray_3d=r2tarray_3d_S[1, comp_i],
                cell_width=cell_width_y,
                grad_axis=1,
            ),
            out=varray_3d_df[comp_i, ...],
        )
        numpy.add(
            varray_3d_df[comp_i, ...],
            nabla(
                sarray_3d=r2tarray_3d_S[2, comp_i],
                cell_width=cell_width_z,
                grad_axis=2,
            ),
            out=varray_3d_df[comp_i, ...],
        )
    return varray_3d_df


def compute_kinetic_dissipation_vfield(
    vfield_3d_u: field_types.VectorField_3D,
    *,
    grad_order: int = 2,
) -> field_types.VectorField_3D:
    """
    Compute d_j S_ji for a 3D velocity field u_i, where

        S_ij = 0.5 * (d_i u_j + d_j u_i) - (1/3) delta_ij (d_k u_k)
    """
    varray_3d_u = field_types.extract_3d_varray(vfield_3d_u)
    udomain_3d = vfield_3d_u.udomain
    sim_time = vfield_3d_u.sim_time
    varray_3d_df = _compute_kinetic_dissipation_varray(
        varray_3d_u=varray_3d_u,
        cell_widths_3d=udomain_3d.cell_widths,
        grad_order=grad_order,
    )
    return field_types.VectorField_3D.from_3d_varray(
        varray_3d=varray_3d_df,
        udomain_3d=udomain_3d,
        field_label=r"$d_j \mathcal{S}_{j i}$",
        sim_time=sim_time,
    )


## } MODULE
