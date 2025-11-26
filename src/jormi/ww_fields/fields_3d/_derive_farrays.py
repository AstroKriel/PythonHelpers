## { MODULE

##
## === DEPENDENCIES
##

import numpy

from dataclasses import dataclass

from jormi.ww_types import array_checks
from jormi.ww_fields.fields_3d import (
    finite_difference,
    farray_operators,
    decompose_farrays,
)


##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class MagneticCurvatureFArrays_3D:
    sarray_3d_curvature: numpy.ndarray
    sarray_3d_stretching: numpy.ndarray
    sarray_3d_compression: numpy.ndarray

    def __post_init__(
        self,
    ) -> None:
        array_checks.ensure_dims(
            array=self.sarray_3d_curvature,
            param_name="<sarray_3d_curvature>",
            num_dims=3,
        )
        array_checks.ensure_dims(
            array=self.sarray_3d_stretching,
            param_name="<sarray_3d_stretching>",
            num_dims=3,
        )
        array_checks.ensure_dims(
            array=self.sarray_3d_compression,
            param_name="<sarray_3d_compression>",
            num_dims=3,
        )
        if any(
            [
                self.sarray_3d_curvature.shape != self.sarray_3d_stretching.shape,
                self.sarray_3d_curvature.shape != self.sarray_3d_compression.shape,
            ],
        ):
            raise ValueError(
                "MagneticCurvatureFArrays_3D components must share the same shape:"
                f" curvature={self.sarray_3d_curvature.shape},"
                f" stretching={self.sarray_3d_stretching.shape},"
                f" compression={self.sarray_3d_compression.shape}.",
            )


@dataclass(frozen=True)
class LorentzForceFArrays_3D:
    varray_3d_lorentz: numpy.ndarray
    varray_3d_tension: numpy.ndarray
    varray_3d_gradP_perp: numpy.ndarray

    def __post_init__(
        self,
    ) -> None:
        array_checks.ensure_dims(
            array=self.varray_3d_lorentz,
            param_name="<varray_3d_lorentz>",
            num_dims=4,
        )
        array_checks.ensure_dims(
            array=self.varray_3d_tension,
            param_name="<varray_3d_tension>",
            num_dims=4,
        )
        array_checks.ensure_dims(
            array=self.varray_3d_gradP_perp,
            param_name="<varray_3d_gradP_perp>",
            num_dims=4,
        )
        if any(
            [
                self.varray_3d_lorentz.shape != self.varray_3d_tension.shape,
                self.varray_3d_lorentz.shape != self.varray_3d_gradP_perp.shape,
            ],
        ):
            raise ValueError(
                "LorentzForceFArrays_3D components must share the same shape:"
                f" lorentz={self.varray_3d_lorentz.shape},"
                f" tension={self.varray_3d_tension.shape},"
                f" gradP_perp={self.varray_3d_gradP_perp.shape}.",
            )
        if self.varray_3d_lorentz.shape[0] != 3:
            raise ValueError(
                "LorentzForceFArrays_3D expects leading axis of length 3; got"
                f" shape={self.varray_3d_lorentz.shape}.",
            )


##
## === MAGNETIC CURVATURE
##


def compute_magnetic_curvature_terms(
    *,
    varray_3d_u: numpy.ndarray,
    uvarray_3d_tangent: numpy.ndarray,
    uvarray_3d_normal: numpy.ndarray,
    cell_widths: tuple[float, float, float],
    r2tarray_3d_gradu: numpy.ndarray | None = None,
    grad_order: int = 2,
) -> MagneticCurvatureFArrays_3D:
    """
    Compute curvature, stretching, and compression terms for a 3D velocity farray.

    Index notation (Einstein summation):
        curvature   = n_i n_j d_i u_j
        stretching  = t_i t_j d_i u_j
        compression = d_i u_i
    """
    array_checks.ensure_dims(
        array=varray_3d_u,
        param_name="<varray_3d_u>",
        num_dims=4,
    )
    array_checks.ensure_dims(
        array=uvarray_3d_tangent,
        param_name="<uvarray_3d_tangent>",
        num_dims=4,
    )
    array_checks.ensure_dims(
        array=uvarray_3d_normal,
        param_name="<uvarray_3d_normal>",
        num_dims=4,
    )
    if any(
        [
            varray_3d_u.shape != uvarray_3d_tangent.shape,
            varray_3d_u.shape != uvarray_3d_normal.shape,
        ],
    ):
        raise ValueError(
            "Velocity/tangent/normal farrays must share the same shape:"
            f" u={varray_3d_u.shape},"
            f" tangent={uvarray_3d_tangent.shape},"
            f" normal={uvarray_3d_normal.shape}.",
        )
    if varray_3d_u.shape[0] != 3:
        raise ValueError(
            "`<varray_3d_u>` must have shape (3, Nx, Ny, Nz);"
            f" got shape={varray_3d_u.shape}.",
        )
    ## d_i u_j: (j, i, x, y, z)
    r2tarray_3d_gradu = farray_operators.compute_varray_grad(
        varray_3d=varray_3d_u,
        cell_widths=cell_widths,
        r2tarray_3d_gradf=r2tarray_3d_gradu,
        grad_order=grad_order,
    )
    ## curvature = n_i n_j d_i u_j
    sarray_3d_curvature = numpy.einsum(
        "ixyz,jxyz,jixyz->xyz",
        uvarray_3d_normal,
        uvarray_3d_normal,
        r2tarray_3d_gradu,
        optimize=True,
    )
    ## stretching = t_i t_j d_i u_j
    sarray_3d_stretching = numpy.einsum(
        "ixyz,jxyz,jixyz->xyz",
        uvarray_3d_tangent,
        uvarray_3d_tangent,
        r2tarray_3d_gradu,
        optimize=True,
    )
    ## compression = d_i u_i = trace over (i,j)
    sarray_3d_compression = numpy.trace(
        r2tarray_3d_gradu,
        axis1=0,  # grad-dir i
        axis2=1,  # comp j
    )
    return MagneticCurvatureFArrays_3D(
        sarray_3d_curvature=sarray_3d_curvature,
        sarray_3d_stretching=sarray_3d_stretching,
        sarray_3d_compression=sarray_3d_compression,
    )


##
## === LORENTZ FORCE
##


def compute_lorentz_force_terms(
    *,
    varray_3d_b: numpy.ndarray,
    cell_widths: tuple[float, float, float],
    grad_order: int = 2,
) -> LorentzForceFArrays_3D:
    """
    Lorentz force decomposition in index notation:

        tension_i    = (b_k b_k) kappa_i
        gradP_perp_i = d_i (b_k b_k / 2) - t_i t_j d_j (b_k b_k / 2)
        lorentz_i    = tension_i - gradP_perp_i
    """
    array_checks.ensure_dims(
        array=varray_3d_b,
        param_name="<varray_3d_b>",
        num_dims=4,
    )
    if varray_3d_b.shape[0] != 3:
        raise ValueError(
            "`<varray_3d_b>` must have shape (3, Nx, Ny, Nz);"
            f" got shape={varray_3d_b.shape}.",
        )
    ## TNB + curvature for B
    tnb_farrays = decompose_farrays.compute_tnb_terms(
        varray_3d=varray_3d_b,
        cell_widths=cell_widths,
        grad_order=grad_order,
    )
    uvarray_3d_tangent = tnb_farrays.uvarray_3d_tangent
    uvarray_3d_normal = tnb_farrays.uvarray_3d_normal
    sarray_3d_curvature = tnb_farrays.sarray_3d_curvature
    ## |b|^2
    sarray_3d_b_magn_sq = farray_operators.sum_of_varray_comps_squared(
        varray_3d=varray_3d_b,
    )
    ## d_i P where P = 0.5 * |b|^2
    nabla = finite_difference.get_grad_fn(grad_order)
    cell_width_x, cell_width_y, cell_width_z = cell_widths
    num_cells_x, num_cells_y, num_cells_z = sarray_3d_b_magn_sq.shape
    varray_3d_gradP = farray_operators.ensure_farray_metadata(
        farray_shape=(3, num_cells_x, num_cells_y, num_cells_z),
        farray=None,
        dtype=sarray_3d_b_magn_sq.dtype,
    )
    varray_3d_gradP[0, ...] = nabla(
        sarray_3d=sarray_3d_b_magn_sq,
        cell_width=cell_width_x,
        grad_axis=0,
    )
    varray_3d_gradP[1, ...] = nabla(
        sarray_3d=sarray_3d_b_magn_sq,
        cell_width=cell_width_y,
        grad_axis=1,
    )
    varray_3d_gradP[2, ...] = nabla(
        sarray_3d=sarray_3d_b_magn_sq,
        cell_width=cell_width_z,
        grad_axis=2,
    )
    varray_3d_gradP *= 0.5
    ## pressure aligned with B: t_i t_j d_j P
    varray_3d_gradP_aligned = numpy.einsum(
        "ixyz,jxyz,jxyz->ixyz",
        uvarray_3d_tangent,
        uvarray_3d_tangent,
        varray_3d_gradP,
        optimize=True,
    )
    ## tension_i = |b|^2 kappa_i
    varray_3d_tension = (
        sarray_3d_b_magn_sq[numpy.newaxis, ...]
        * sarray_3d_curvature[numpy.newaxis, ...]
        * uvarray_3d_normal
    )
    ## gradP_perp_i = d_i P - t_i t_j d_j P
    varray_3d_gradP_perp = varray_3d_gradP - varray_3d_gradP_aligned
    ## lorentz_i = tension_i - gradP_perp_i
    varray_3d_lorentz = varray_3d_tension - varray_3d_gradP_perp
    return LorentzForceFArrays_3D(
        varray_3d_lorentz=varray_3d_lorentz,
        varray_3d_tension=varray_3d_tension,
        varray_3d_gradP_perp=varray_3d_gradP_perp,
    )


##
## === KINETIC DISSIPATION
##


def compute_kinetic_dissipation(
    *,
    varray_3d_u: numpy.ndarray,
    cell_widths: tuple[float, float, float],
    grad_order: int = 2,
) -> numpy.ndarray:
    """
    Compute d_j S_ji for a 3D velocity farray u_j, where

        S_ij = 0.5 * (d_i u_j + d_j u_i) - (1/3) delta_ij (d_k u_k)
    """
    array_checks.ensure_dims(
        array=varray_3d_u,
        param_name="<varray_3d_u>",
        num_dims=4,
    )
    if varray_3d_u.shape[0] != 3:
        raise ValueError(
            "`<varray_3d_u>` must have shape (3, Nx, Ny, Nz);"
            f" got shape={varray_3d_u.shape}.",
        )
    dtype = varray_3d_u.dtype
    num_cells_x, num_cells_y, num_cells_z = varray_3d_u.shape[1:]
    ## d_i u_j
    r2tarray_3d_gradu = farray_operators.compute_varray_grad(
        varray_3d=varray_3d_u,
        cell_widths=cell_widths,
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
    nabla = finite_difference.get_grad_fn(grad_order)
    cell_width_x, cell_width_y, cell_width_z = cell_widths
    varray_3d_df = farray_operators.ensure_farray_metadata(
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


## } MODULE
