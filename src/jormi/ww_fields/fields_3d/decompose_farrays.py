## { MODULE
##
## === DEPENDENCIES
##

import numpy

from dataclasses import dataclass

from jormi.ww_types import array_checks
from jormi.ww_fields.fields_3d import farray_operators


##
## === DATA STRUCTURES
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
        array_checks.ensure_dims(
            array=self.varray_3d_div,
            param_name="<varray_3d_div>",
            num_dims=4,
        )
        array_checks.ensure_dims(
            array=self.varray_3d_sol,
            param_name="<varray_3d_sol>",
            num_dims=4,
        )
        array_checks.ensure_dims(
            array=self.varray_3d_bulk,
            param_name="<varray_3d_bulk>",
            num_dims=4,
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
        if self.varray_3d_div.shape[0] != 3:
            raise ValueError(
                "HelmholtzDecomposedFArrays_3D expects leading axis of length 3; got"
                f" shape={self.varray_3d_div.shape}.",
            )


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
        array_checks.ensure_dims(
            array=self.uvarray_3d_tangent,
            param_name="<uvarray_3d_tangent>",
            num_dims=4,
        )
        array_checks.ensure_dims(
            array=self.uvarray_3d_normal,
            param_name="<uvarray_3d_normal>",
            num_dims=4,
        )
        array_checks.ensure_dims(
            array=self.uvarray_3d_binormal,
            param_name="<uvarray_3d_binormal>",
            num_dims=4,
        )
        array_checks.ensure_dims(
            array=self.sarray_3d_curvature,
            param_name="<sarray_3d_curvature>",
            num_dims=3,
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
        if self.uvarray_3d_tangent.shape[0] != 3:
            raise ValueError(
                "TNBDecomposedFArrays_3D expects vector arrays with leading axis=3; got"
                f" shape={self.uvarray_3d_tangent.shape}.",
            )
        if self.uvarray_3d_tangent.shape[1:] != self.sarray_3d_curvature.shape:
            raise ValueError(
                "TNBDecomposedFArrays_3D curvature shape must match spatial shape of"
                f" vectors: curvature={self.sarray_3d_curvature.shape},"
                f" vectors={self.uvarray_3d_tangent.shape[1:]}.",
            )


##
## === HELMHOLTZ ON FARRAYS
##


def compute_helmholtz_decomposition(
    *,
    varray_3d_q: numpy.ndarray,
    resolution: tuple[int, int, int],
    cell_widths: tuple[float, float, float],
) -> HelmholtzDecomposedFArrays_3D:
    """
    Helmholtz decompose a 3D vector farray into (div, sol, bulk) components.

    Parameters
    ----------
    varray_3d_q : ndarray
        3D vector field with shape (3, Nx, Ny, Nz).
    resolution : (int, int, int)
        Spatial resolution (Nx, Ny, Nz).
    cell_widths : (float, float, float)
        Cell widths (dx, dy, dz).

    Returns
    -------
    HelmholtzDecomposedFArrays_3D
        Decomposed varrays.
    """
    array_checks.ensure_dims(
        array=varray_3d_q,
        param_name="<varray_3d_q>",
        num_dims=4,
    )
    if varray_3d_q.shape[0] != 3:
        raise ValueError(
            "`<varray_3d_q>` must have shape (3, Nx, Ny, Nz);"
            f" got shape={varray_3d_q.shape}.",
        )
    num_cells_x, num_cells_y, num_cells_z = resolution
    cell_width_x, cell_width_y, cell_width_z = cell_widths
    dtype = varray_3d_q.dtype
    kx_values = 2.0 * numpy.pi * numpy.fft.fftfreq(num_cells_x, d=cell_width_x)
    ky_values = 2.0 * numpy.pi * numpy.fft.fftfreq(num_cells_y, d=cell_width_y)
    kz_values = 2.0 * numpy.pi * numpy.fft.fftfreq(num_cells_z, d=cell_width_z)
    kx_grid, ky_grid, kz_grid = numpy.meshgrid(kx_values, ky_values, kz_values, indexing="ij")
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
        kx_grid * varray_3d_fft_q[0]
        + ky_grid * varray_3d_fft_q[1]
        + kz_grid * varray_3d_fft_q[2]
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
## === TNB ON FARRAYS
##


def compute_tnb_terms(
    *,
    varray_3d: numpy.ndarray,
    cell_widths: tuple[float, float, float],
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
    array_checks.ensure_dims(
        array=varray_3d,
        param_name="<varray_3d>",
        num_dims=4,
    )
    if varray_3d.shape[0] != 3:
        raise ValueError(
            "`<varray_3d>` must have shape (3, Nx, Ny, Nz);"
            f" got shape={varray_3d.shape}.",
        )
    ## |f|^2 = f_i f_i
    sarray_3d_f_magn_sq = farray_operators.sum_of_varray_comps_squared(
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
    r2tarray_3d_gradf = farray_operators.compute_varray_grad(
        varray_3d=varray_3d,
        cell_widths=cell_widths,
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
        varray_3d_normal_term1 * sarray_3d_inv_magn_sq
        - varray_3d_normal_term2 * sarray_3d_inv_magn4
    )
    sarray_3d_curvature = farray_operators.sum_of_varray_comps_squared(
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
    uvarray_3d_binormal = farray_operators.compute_varray_cross_product(
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


## } MODULE
