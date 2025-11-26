## { MODULE

##
## === DEPENDENCIES
##

import numpy

from dataclasses import dataclass

from jormi.ww_types import type_manager
from jormi.ww_fields.fields_3d import (
    _farray_operators,
    _finite_difference_sarrays,
    _fdata_types,
)

##
## === HELMHOLTZ DECOMPOSITION
##


@dataclass(frozen=True)
class HelmholtzDecomposedFArrays_3D:
    """Helmholtz decomposition of a 3D vector farray into div/sol/bulk components."""

    varray_3d_div: numpy.ndarray
    varray_3d_sol: numpy.ndarray
    varray_3d_bulk: numpy.ndarray

    def __post_init__(
        self,
    ) -> None:
        _fdata_types.ensure_3d_varray(
            varray_3d=self.varray_3d_div,
            param_name="<varray_3d_div>",
        )
        _fdata_types.ensure_3d_varray(
            varray_3d=self.varray_3d_sol,
            param_name="<varray_3d_sol>",
        )
        _fdata_types.ensure_3d_varray(
            varray_3d=self.varray_3d_bulk,
            param_name="<varray_3d_bulk>",
        )
        if any([
                self.varray_3d_div.shape != self.varray_3d_sol.shape,
                self.varray_3d_div.shape != self.varray_3d_bulk.shape,
        ]):
            raise ValueError(
                "HelmholtzDecomposedFArrays_3D components must share the same shape:"
                f" div={self.varray_3d_div.shape},"
                f" sol={self.varray_3d_sol.shape},"
                f" bulk={self.varray_3d_bulk.shape}.",
            )


def compute_helmholtz_decomposed_farrays(
    *,
    varray_3d_q: numpy.ndarray,
    resolution: tuple[int, int, int],
    cell_widths_3d: tuple[float, float, float],
) -> HelmholtzDecomposedFArrays_3D:
    """
    Helmholtz decompose a 3D vector farray into (div, sol, bulk) components.

    Parameters
    ----------
    varray_3d_q : ndarray
        3D vector field with shape (3, Nx, Ny, Nz).
    resolution : (int, int, int)
        Spatial resolution (Nx, Ny, Nz).
    cell_widths_3d : (float, float, float)
        Cell widths (dx, dy, dz).

    Returns
    -------
    HelmholtzDecomposedFArrays_3D
        Decomposed varrays.
    """
    _fdata_types.ensure_3d_varray(
        varray_3d=varray_3d_q,
        param_name="<varray_3d_q>",
    )
    if varray_3d_q.shape[1:] != resolution:
        raise ValueError(
            "`<resolution>` must match the spatial shape of `<varray_3d_q>`:"
            f" resolution={resolution},"
            f" varray_3d_q.shape[1:]={varray_3d_q.shape[1:]}.",
        )
    _farray_operators._validate_3d_cell_widths(cell_widths_3d)
    num_cells_x, num_cells_y, num_cells_z = resolution
    cell_width_x, cell_width_y, cell_width_z = cell_widths_3d
    dtype = varray_3d_q.dtype
    kx_values = 2.0 * numpy.pi * numpy.fft.fftfreq(num_cells_x, d=cell_width_x)
    ky_values = 2.0 * numpy.pi * numpy.fft.fftfreq(num_cells_y, d=cell_width_y)
    kz_values = 2.0 * numpy.pi * numpy.fft.fftfreq(num_cells_z, d=cell_width_z)
    kx_grid, ky_grid, kz_grid = numpy.meshgrid(
        kx_values,
        ky_values,
        kz_values,
        indexing="ij",
    )
    k_magn_grid = kx_grid**2 + ky_grid**2 + kz_grid**2
    k_magn_grid[0, 0, 0] = 1.0  ## avoid division by zero at k=0
    varray_3d_fft_q = numpy.fft.fftn(
        varray_3d_q,
        axes=(1, 2, 3),
        norm="forward",
    )
    varray_3d_fft_bulk = numpy.zeros_like(varray_3d_fft_q)
    varray_3d_fft_bulk[:, 0, 0, 0] = varray_3d_fft_q[:, 0, 0, 0]
    varray_3d_fft_q[:, 0, 0, 0] = 0.0
    sarray_3d_k_dot_fft_q = (
        kx_grid * varray_3d_fft_q[0] + ky_grid * varray_3d_fft_q[1] + kz_grid * varray_3d_fft_q[2]
    )
    with numpy.errstate(divide="ignore", invalid="ignore"):
        varray_3d_fft_div = numpy.stack(
            [
                (kx_grid / k_magn_grid) * sarray_3d_k_dot_fft_q,
                (ky_grid / k_magn_grid) * sarray_3d_k_dot_fft_q,
                (kz_grid / k_magn_grid) * sarray_3d_k_dot_fft_q,
            ],
            axis=0,
        )
    varray_3d_fft_sol = varray_3d_fft_q - varray_3d_fft_div
    varray_3d_div = numpy.fft.ifftn(
        varray_3d_fft_div,
        axes=(1, 2, 3),
        norm="forward",
    ).real.astype(
        dtype,
        copy=False,
    )
    varray_3d_sol = numpy.fft.ifftn(
        varray_3d_fft_sol,
        axes=(1, 2, 3),
        norm="forward",
    ).real.astype(
        dtype,
        copy=False,
    )
    varray_3d_bulk = numpy.fft.ifftn(
        varray_3d_fft_bulk,
        axes=(1, 2, 3),
        norm="forward",
    ).real.astype(
        dtype,
        copy=False,
    )
    del (
        kx_values,
        ky_values,
        kz_values,
        kx_grid,
        ky_grid,
        kz_grid,
        k_magn_grid,
        varray_3d_fft_q,
        sarray_3d_k_dot_fft_q,
        varray_3d_fft_div,
        varray_3d_fft_sol,
        varray_3d_fft_bulk,
    )
    return HelmholtzDecomposedFArrays_3D(
        varray_3d_div=varray_3d_div,
        varray_3d_sol=varray_3d_sol,
        varray_3d_bulk=varray_3d_bulk,
    )


##
## === TNB BASIS
##


@dataclass(frozen=True)
class TNBDecomposedFArrays_3D:
    """TNB decomposition of a 3D vector farray into unit bases and curvature."""

    uvarray_3d_tangent: numpy.ndarray
    uvarray_3d_normal: numpy.ndarray
    uvarray_3d_binormal: numpy.ndarray
    sarray_3d_curvature: numpy.ndarray

    def __post_init__(
        self,
    ) -> None:
        _fdata_types.ensure_3d_varray(
            varray_3d=self.uvarray_3d_tangent,
            param_name="<uvarray_3d_tangent>",
        )
        _fdata_types.ensure_3d_varray(
            varray_3d=self.uvarray_3d_normal,
            param_name="<uvarray_3d_normal>",
        )
        _fdata_types.ensure_3d_varray(
            varray_3d=self.uvarray_3d_binormal,
            param_name="<uvarray_3d_binormal>",
        )
        _fdata_types.ensure_3d_sarray(
            sarray_3d=self.sarray_3d_curvature,
            param_name="<sarray_3d_curvature>",
        )
        if any([
                self.uvarray_3d_tangent.shape != self.uvarray_3d_normal.shape,
                self.uvarray_3d_tangent.shape != self.uvarray_3d_binormal.shape,
        ]):
            raise ValueError(
                "TNBDecomposedFArrays_3D vector components must share the same shape:"
                f" tangent={self.uvarray_3d_tangent.shape},"
                f" normal={self.uvarray_3d_normal.shape},"
                f" binormal={self.uvarray_3d_binormal.shape}.",
            )
        if self.uvarray_3d_tangent.shape[1:] != self.sarray_3d_curvature.shape:
            raise ValueError(
                "TNBDecomposedFArrays_3D curvature shape must match spatial shape of"
                f" vectors: curvature={self.sarray_3d_curvature.shape},"
                f" vectors={self.uvarray_3d_tangent.shape[1:]}.",
            )


def compute_tnb_farrays(
    *,
    varray_3d: numpy.ndarray,
    cell_widths_3d: tuple[float, float, float],
    grad_order: int = 2,
) -> TNBDecomposedFArrays_3D:
    """
    Compute T_i, N_i, B_i and curvature sqrt(kappa_i kappa_i) from a 3D vector farray.

    Returns:
      - uvarray_3d_tangent   (3, Nx, Ny, Nz)
      - uvarray_3d_normal    (3, Nx, Ny, Nz)
      - uvarray_3d_binormal  (3, Nx, Ny, Nz)
      - sarray_3d_curvature  (Nx, Ny, Nz)
    """
    _fdata_types.ensure_3d_varray(
        varray_3d=varray_3d,
        param_name="<varray_3d>",
    )
    _farray_operators._validate_3d_cell_widths(cell_widths_3d)
    type_manager.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    ## |f|^2 = f_i f_i
    sarray_3d_f_magn_sq = _farray_operators.sum_of_varray_comps_squared(
        varray_3d=varray_3d,
    )
    ## |f| = sqrt(f_i f_i)
    sarray_3d_f_magn = numpy.sqrt(
        sarray_3d_f_magn_sq,
        dtype=sarray_3d_f_magn_sq.dtype,
    )
    ## T_i = f_i / |f|
    uvarray_3d_tangent = numpy.zeros_like(varray_3d)
    numpy.divide(
        varray_3d,
        sarray_3d_f_magn,
        out=uvarray_3d_tangent,
        where=(sarray_3d_f_magn > 0),
    )
    ## grad f: d_i f_j, layout (j, i, x, y, z)
    r2tarray_3d_gradf = _farray_operators.compute_varray_grad(
        varray_3d=varray_3d,
        cell_widths_3d=cell_widths_3d,
        r2tarray_3d_gradf=None,
        grad_order=grad_order,
    )
    ## term1_j = f_i * (d_i f_j)
    varray_3d_normal_term1 = numpy.einsum(
        "ixyz,jixyz->jxyz",
        varray_3d,
        r2tarray_3d_gradf,
        optimize=True,
    )
    ## term2_j = f_i f_j f_m (d_i f_m)
    varray_3d_normal_term2 = numpy.einsum(
        "ixyz,jxyz,mxyz,mixyz->jxyz",
        varray_3d,
        varray_3d,
        varray_3d,
        r2tarray_3d_gradf,
        optimize=True,
    )
    ## curvature vector kappa_j and magnitude |kappa|
    sarray_3d_inv_magn_sq = numpy.zeros_like(sarray_3d_f_magn_sq)
    numpy.divide(
        1.0,
        sarray_3d_f_magn_sq,
        out=sarray_3d_inv_magn_sq,
        where=(sarray_3d_f_magn_sq > 0),
    )
    sarray_3d_inv_magn4 = sarray_3d_inv_magn_sq**2
    varray_3d_kappa = (
        varray_3d_normal_term1 * sarray_3d_inv_magn_sq - varray_3d_normal_term2 * sarray_3d_inv_magn4
    )
    sarray_3d_curvature = _farray_operators.sum_of_varray_comps_squared(
        varray_3d=varray_3d_kappa,
    )
    numpy.sqrt(sarray_3d_curvature, out=sarray_3d_curvature)
    ## N_i = kappa_i / |kappa|
    uvarray_3d_normal = numpy.zeros_like(varray_3d_kappa)
    numpy.divide(
        varray_3d_kappa,
        sarray_3d_curvature,
        out=uvarray_3d_normal,
        where=(sarray_3d_curvature > 0.0),
    )
    ## B_i = (T x N)_i
    uvarray_3d_binormal = _farray_operators.compute_varray_cross_product(
        varray_3d_a=uvarray_3d_tangent,
        varray_3d_b=uvarray_3d_normal,
    )
    del (
        varray_3d_normal_term1,
        varray_3d_normal_term2,
        sarray_3d_inv_magn_sq,
        sarray_3d_inv_magn4,
    )
    return TNBDecomposedFArrays_3D(
        uvarray_3d_tangent=uvarray_3d_tangent,
        uvarray_3d_normal=uvarray_3d_normal,
        uvarray_3d_binormal=uvarray_3d_binormal,
        sarray_3d_curvature=sarray_3d_curvature,
    )


##
## === MAGNETIC CURVATURE
##


@dataclass(frozen=True)
class MagneticCurvatureFArrays_3D:
    sarray_3d_curvature: numpy.ndarray
    sarray_3d_stretching: numpy.ndarray
    sarray_3d_compression: numpy.ndarray

    def __post_init__(
        self,
    ) -> None:
        _fdata_types.ensure_3d_sarray(
            sarray_3d=self.sarray_3d_curvature,
            param_name="<sarray_3d_curvature>",
        )
        _fdata_types.ensure_3d_sarray(
            sarray_3d=self.sarray_3d_stretching,
            param_name="<sarray_3d_stretching>",
        )
        _fdata_types.ensure_3d_sarray(
            sarray_3d=self.sarray_3d_compression,
            param_name="<sarray_3d_compression>",
        )
        if any([
                self.sarray_3d_curvature.shape != self.sarray_3d_stretching.shape,
                self.sarray_3d_curvature.shape != self.sarray_3d_compression.shape,
        ]):
            raise ValueError(
                "MagneticCurvatureFArrays_3D components must share the same shape:"
                f" curvature={self.sarray_3d_curvature.shape},"
                f" stretching={self.sarray_3d_stretching.shape},"
                f" compression={self.sarray_3d_compression.shape}.",
            )


def compute_magnetic_curvature_farrays(
    *,
    varray_3d_u: numpy.ndarray,
    uvarray_3d_tangent: numpy.ndarray,
    uvarray_3d_normal: numpy.ndarray,
    cell_widths_3d: tuple[float, float, float],
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
    _fdata_types.ensure_3d_varray(
        varray_3d=varray_3d_u,
        param_name="<varray_3d_u>",
    )
    _fdata_types.ensure_3d_varray(
        varray_3d=uvarray_3d_tangent,
        param_name="<uvarray_3d_tangent>",
    )
    _fdata_types.ensure_3d_varray(
        varray_3d=uvarray_3d_normal,
        param_name="<uvarray_3d_normal>",
    )
    if any([
            varray_3d_u.shape != uvarray_3d_tangent.shape,
            varray_3d_u.shape != uvarray_3d_normal.shape,
    ]):
        raise ValueError(
            "Velocity/tangent/normal farrays must share the same shape:"
            f" u={varray_3d_u.shape},"
            f" tangent={uvarray_3d_tangent.shape},"
            f" normal={uvarray_3d_normal.shape}.",
        )
    _farray_operators._validate_3d_cell_widths(cell_widths_3d)
    type_manager.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    ## d_i u_j: (j, i, x, y, z)
    r2tarray_3d_gradu = _farray_operators.compute_varray_grad(
        varray_3d=varray_3d_u,
        cell_widths_3d=cell_widths_3d,
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


@dataclass(frozen=True)
class LorentzForceFArrays_3D:
    varray_3d_lorentz: numpy.ndarray
    varray_3d_tension: numpy.ndarray
    varray_3d_gradP_perp: numpy.ndarray

    def __post_init__(
        self,
    ) -> None:
        _fdata_types.ensure_3d_varray(
            varray_3d=self.varray_3d_lorentz,
            param_name="<varray_3d_lorentz>",
        )
        _fdata_types.ensure_3d_varray(
            varray_3d=self.varray_3d_tension,
            param_name="<varray_3d_tension>",
        )
        _fdata_types.ensure_3d_varray(
            varray_3d=self.varray_3d_gradP_perp,
            param_name="<varray_3d_gradP_perp>",
        )
        if any([
                self.varray_3d_lorentz.shape != self.varray_3d_tension.shape,
                self.varray_3d_lorentz.shape != self.varray_3d_gradP_perp.shape,
        ]):
            raise ValueError(
                "LorentzForceFArrays_3D components must share the same shape:"
                f" lorentz={self.varray_3d_lorentz.shape},"
                f" tension={self.varray_3d_tension.shape},"
                f" gradP_perp={self.varray_3d_gradP_perp.shape}.",
            )


def compute_lorentz_force_farrays(
    *,
    varray_3d_b: numpy.ndarray,
    cell_widths_3d: tuple[float, float, float],
    grad_order: int = 2,
) -> LorentzForceFArrays_3D:
    """
    Lorentz force decomposition in index notation:

        tension_i    = (b_k b_k) kappa_i
        gradP_perp_i = d_i (b_k b_k / 2) - t_i t_j d_j (b_k b_k / 2)
        lorentz_i    = tension_i - gradP_perp_i
    """
    _fdata_types.ensure_3d_varray(
        varray_3d=varray_3d_b,
        param_name="<varray_3d_b>",
    )
    _farray_operators._validate_3d_cell_widths(cell_widths_3d)
    type_manager.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    ## TNB + curvature for B
    tnb_farrays = compute_tnb_farrays(
        varray_3d=varray_3d_b,
        cell_widths_3d=cell_widths_3d,
        grad_order=grad_order,
    )
    uvarray_3d_tangent = tnb_farrays.uvarray_3d_tangent
    uvarray_3d_normal = tnb_farrays.uvarray_3d_normal
    sarray_3d_curvature = tnb_farrays.sarray_3d_curvature
    ## |b|^2
    sarray_3d_b_magn_sq = _farray_operators.sum_of_varray_comps_squared(
        varray_3d=varray_3d_b,
    )
    ## d_i P where P = 0.5 * |b|^2
    nabla = _finite_difference_sarrays.get_grad_fn(grad_order)
    cell_width_x, cell_width_y, cell_width_z = cell_widths_3d
    num_cells_x, num_cells_y, num_cells_z = sarray_3d_b_magn_sq.shape
    varray_3d_gradP = _fdata_types.ensure_farray_metadata(
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
        sarray_3d_b_magn_sq[numpy.newaxis, ...] * sarray_3d_curvature[numpy.newaxis, ...] * uvarray_3d_normal
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


## } MODULE
