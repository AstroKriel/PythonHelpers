## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from typing import Any

## third-party
import numpy
from numpy.typing import NDArray

## local
from jormi.ww_arrays.farrays_3d import (
    difference_sarrays,
    farray_operators,
    farray_types,
)
from jormi.ww_validation import validate_types

##
## === HELMHOLTZ DECOMPOSITION
##


@dataclass(frozen=True)
class HelmholtzDecomposedFArrays_3D:
    """Helmholtz decomposition of a 3D varray into div/sol/bulk components."""

    varray_3d_div: NDArray[Any]
    varray_3d_sol: NDArray[Any]
    varray_3d_bulk: NDArray[Any]

    def __post_init__(
        self,
    ) -> None:
        ## validate each decomposed vector field
        farray_types.ensure_3d_varray(
            varray_3d=self.varray_3d_div,
            param_name="<varray_3d_div>",
        )
        farray_types.ensure_3d_varray(
            varray_3d=self.varray_3d_sol,
            param_name="<varray_3d_sol>",
        )
        farray_types.ensure_3d_varray(
            varray_3d=self.varray_3d_bulk,
            param_name="<varray_3d_bulk>",
        )
        ## validate shared decomposition geometry
        if any([
                self.varray_3d_div.shape != self.varray_3d_sol.shape,
                self.varray_3d_div.shape != self.varray_3d_bulk.shape,
        ], ):
            raise ValueError(
                "HelmholtzDecomposedFArrays_3D components must share the same shape:"
                f" div={self.varray_3d_div.shape},"
                f" sol={self.varray_3d_sol.shape},"
                f" bulk={self.varray_3d_bulk.shape}.",
            )


def compute_helmholtz_decomposed_farrays(
    *,
    varray_3d: NDArray[Any],
    resolution: tuple[int, int, int],
    cell_widths_3d: tuple[float, float, float],
) -> HelmholtzDecomposedFArrays_3D:
    """
    Helmholtz decompose a 3D varray into (div, sol, bulk) components.

    Parameters
    ----------
    varray_3d : ndarray
        3D vector field with shape (3, num_x0_cells, num_x1_cells, num_x2_cells).
    resolution : (int, int, int)
        Spatial resolution (num_x0_cells, num_x1_cells, num_x2_cells).
    cell_widths_3d : (float, float, float)
        Cell widths (delta_x0, delta_x1, delta_x2).

    Returns
    -------
    HelmholtzDecomposedFArrays_3D
        Decomposed varrays.
    """
    farray_types.ensure_3d_varray(
        varray_3d=varray_3d,
        param_name="<varray_3d>",
    )
    if varray_3d.shape[1:] != resolution:
        raise ValueError(
            "`<resolution>` must match the spatial shape of `<varray_3d>`:"
            f" resolution={resolution},"
            f" varray_3d.shape[1:]={varray_3d.shape[1:]}.",
        )
    farray_types.ensure_3d_cell_widths(cell_widths_3d)
    num_cells_x, num_cells_y, num_cells_z = resolution
    cell_width_x, cell_width_y, cell_width_z = cell_widths_3d
    dtype = varray_3d.dtype
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
    varray_3d_fft = numpy.fft.fftn(
        varray_3d,
        axes=(1, 2, 3),
        norm="forward",
    )
    varray_3d_fft_bulk = numpy.zeros_like(varray_3d_fft)
    varray_3d_fft_bulk[:, 0, 0, 0] = varray_3d_fft[:, 0, 0, 0]
    varray_3d_fft[:, 0, 0, 0] = 0.0
    sarray_3d_k_dot_fft = (
        kx_grid * varray_3d_fft[0] + ky_grid * varray_3d_fft[1] + kz_grid * varray_3d_fft[2]
    )
    with numpy.errstate(divide="ignore", invalid="ignore"):
        varray_3d_fft_div = numpy.stack(
            [
                (kx_grid / k_magn_grid) * sarray_3d_k_dot_fft,
                (ky_grid / k_magn_grid) * sarray_3d_k_dot_fft,
                (kz_grid / k_magn_grid) * sarray_3d_k_dot_fft,
            ],
            axis=0,
        )
    del kx_grid, ky_grid, kz_grid, k_magn_grid, sarray_3d_k_dot_fft
    varray_3d_fft_sol = varray_3d_fft - varray_3d_fft_div
    del varray_3d_fft
    varray_3d_div = numpy.fft.ifftn(
        varray_3d_fft_div,
        axes=(1, 2, 3),
        norm="forward",
    ).real.astype(
        dtype,
        copy=False,
    )
    del varray_3d_fft_div
    varray_3d_sol = numpy.fft.ifftn(
        varray_3d_fft_sol,
        axes=(1, 2, 3),
        norm="forward",
    ).real.astype(
        dtype,
        copy=False,
    )
    del varray_3d_fft_sol
    varray_3d_bulk = numpy.fft.ifftn(
        varray_3d_fft_bulk,
        axes=(1, 2, 3),
        norm="forward",
    ).real.astype(
        dtype,
        copy=False,
    )
    del varray_3d_fft_bulk
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
    """TNB decomposition of a 3D varray into unit bases and curvature."""

    uvarray_3d_tangent: NDArray[Any]
    uvarray_3d_normal: NDArray[Any]
    uvarray_3d_binormal: NDArray[Any]
    sarray_3d_curvature: NDArray[Any]

    def __post_init__(
        self,
    ) -> None:
        ## validate the TNB basis arrays individually
        farray_types.ensure_3d_varray(
            varray_3d=self.uvarray_3d_tangent,
            param_name="<uvarray_3d_tangent>",
        )
        farray_types.ensure_3d_varray(
            varray_3d=self.uvarray_3d_normal,
            param_name="<uvarray_3d_normal>",
        )
        farray_types.ensure_3d_varray(
            varray_3d=self.uvarray_3d_binormal,
            param_name="<uvarray_3d_binormal>",
        )
        farray_types.ensure_3d_sarray(
            sarray_3d=self.sarray_3d_curvature,
            param_name="<sarray_3d_curvature>",
        )
        ## validate that the vector basis shares one grid
        if any([
                self.uvarray_3d_tangent.shape != self.uvarray_3d_normal.shape,
                self.uvarray_3d_tangent.shape != self.uvarray_3d_binormal.shape,
        ], ):
            raise ValueError(
                "TNBDecomposedFArrays_3D vector components must share the same shape:"
                f" tangent={self.uvarray_3d_tangent.shape},"
                f" normal={self.uvarray_3d_normal.shape},"
                f" binormal={self.uvarray_3d_binormal.shape}.",
            )
        ## validate curvature against the vector spatial grid
        if self.uvarray_3d_tangent.shape[1:] != self.sarray_3d_curvature.shape:
            raise ValueError(
                "TNBDecomposedFArrays_3D curvature shape must match spatial shape of"
                f" vectors: curvature={self.sarray_3d_curvature.shape},"
                f" vectors={self.uvarray_3d_tangent.shape[1:]}.",
            )


def compute_tnb_farrays(
    *,
    varray_3d: NDArray[Any],
    cell_widths_3d: tuple[float, float, float],
    grad_order: int = 2,
) -> TNBDecomposedFArrays_3D:
    """
    Compute t_i, n_i, b_i and curvature sqrt(kappa_i kappa_i) from a 3D varray.

    Returns:
      - uvarray_3d_tangent   (3, num_x0_cells, num_x1_cells, num_x2_cells)
      - uvarray_3d_normal    (3, num_x0_cells, num_x1_cells, num_x2_cells)
      - uvarray_3d_binormal  (3, num_x0_cells, num_x1_cells, num_x2_cells)
      - sarray_3d_curvature  (num_x0_cells, num_x1_cells, num_x2_cells)
    """
    farray_types.ensure_3d_varray(
        varray_3d=varray_3d,
        param_name="<varray_3d>",
    )
    farray_types.ensure_3d_cell_widths(cell_widths_3d)
    validate_types.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
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
    ## t_i = f_i / |f|
    uvarray_3d_tangent = numpy.zeros_like(varray_3d)
    numpy.divide(
        varray_3d,
        sarray_3d_f_magn,
        out=uvarray_3d_tangent,
        where=(sarray_3d_f_magn > 0),
    )
    del sarray_3d_f_magn
    ## grad f: d_i f_j, layout (j, i, x0, x1, x2)
    r2tarray_3d_grad_f = farray_operators.compute_varray_grad(
        varray_3d=varray_3d,
        cell_widths_3d=cell_widths_3d,
        r2tarray_3d_grad_f=None,
        grad_order=grad_order,
    )
    ## term1_j = f_i * (d_i f_j)
    varray_3d_normal_term1 = numpy.einsum(
        "ixyz,jixyz->jxyz",
        varray_3d,
        r2tarray_3d_grad_f,
        optimize=True,
    )
    ## term2_j = f_i f_j f_m (d_i f_m)
    varray_3d_normal_term2 = numpy.einsum(
        "ixyz,jxyz,mxyz,mixyz->jxyz",
        varray_3d,
        varray_3d,
        varray_3d,
        r2tarray_3d_grad_f,
        optimize=True,
    )
    del r2tarray_3d_grad_f
    ## curvature vector kappa_j and magnitude |kappa|
    sarray_3d_inv_magn_sq = numpy.zeros_like(sarray_3d_f_magn_sq)
    numpy.divide(
        1.0,
        sarray_3d_f_magn_sq,
        out=sarray_3d_inv_magn_sq,
        where=(sarray_3d_f_magn_sq > 0),
    )
    del sarray_3d_f_magn_sq
    sarray_3d_inv_magn4 = sarray_3d_inv_magn_sq**2
    varray_3d_kappa = (
        varray_3d_normal_term1 * sarray_3d_inv_magn_sq - varray_3d_normal_term2 * sarray_3d_inv_magn4
    )
    del varray_3d_normal_term1, varray_3d_normal_term2, sarray_3d_inv_magn_sq, sarray_3d_inv_magn4
    sarray_3d_curvature = farray_operators.sum_of_varray_comps_squared(
        varray_3d=varray_3d_kappa,
    )
    numpy.sqrt(sarray_3d_curvature, out=sarray_3d_curvature)
    ## n_i = kappa_i / |kappa|
    uvarray_3d_normal = numpy.zeros_like(varray_3d_kappa)
    numpy.divide(
        varray_3d_kappa,
        sarray_3d_curvature,
        out=uvarray_3d_normal,
        where=(sarray_3d_curvature > 0.0),
    )
    del varray_3d_kappa
    ## b_i = (t x n)_i
    uvarray_3d_binormal = farray_operators.compute_varray_cross_product(
        varray_3d_a=uvarray_3d_tangent,
        varray_3d_b=uvarray_3d_normal,
    )
    return TNBDecomposedFArrays_3D(
        uvarray_3d_tangent=uvarray_3d_tangent,
        uvarray_3d_normal=uvarray_3d_normal,
        uvarray_3d_binormal=uvarray_3d_binormal,
        sarray_3d_curvature=sarray_3d_curvature,
    )


def compute_curvature_sarray(
    *,
    varray_3d: NDArray[Any],
    cell_widths_3d: tuple[float, float, float],
    grad_order: int = 2,
) -> NDArray[Any]:
    """
    Compute curvature magnitude sqrt(kappa_i kappa_i) from a 3D varray.

    This implementation is tailored to the scalar curvature magnitude and
    avoids building the full curvature vector first.
    """
    farray_types.ensure_3d_varray(
        varray_3d=varray_3d,
        param_name="<varray_3d>",
    )
    farray_types.ensure_3d_cell_widths(cell_widths_3d)
    validate_types.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    ## promote low-precision inputs so gradient and curvature algebra run in float64
    varray_3d = farray_operators._as_float_view(varray_3d)
    ## |f|^2 = f_i f_i
    sarray_3d_f_magn_sq = farray_operators.sum_of_varray_comps_squared(
        varray_3d=varray_3d,
    )
    ## term1_j = f_i * (d_i f_j) = ((f . grad) f)_j
    varray_3d_normal_term1 = farray_operators.compute_varray_directional_derivative(
        varray_3d_target=varray_3d,
        varray_3d_along=varray_3d,
        cell_widths_3d=cell_widths_3d,
        grad_order=grad_order,
    )
    ## term2_j = f_j * [f_m * term1_m]
    sarray_3d_normal_prefactor = farray_operators.dot_over_varray_comps(
        varray_3d_a=varray_3d,
        varray_3d_b=varray_3d_normal_term1,
    )
    sarray_3d_inv_magn_sq = numpy.zeros_like(sarray_3d_f_magn_sq)
    numpy.divide(
        1.0,
        sarray_3d_f_magn_sq,
        out=sarray_3d_inv_magn_sq,
        where=(sarray_3d_f_magn_sq > 0),
    )
    del sarray_3d_f_magn_sq
    sarray_3d_inv_magn4 = sarray_3d_inv_magn_sq**2
    varray_3d_kappa = (
        varray_3d_normal_term1 * sarray_3d_inv_magn_sq -
        varray_3d * sarray_3d_normal_prefactor * sarray_3d_inv_magn4
    )
    del (
        varray_3d_normal_term1,
        sarray_3d_normal_prefactor,
        sarray_3d_inv_magn_sq,
        sarray_3d_inv_magn4,
    )
    ## |kappa| = sqrt(kappa_i kappa_i)
    sarray_3d_curvature = farray_operators.sum_of_varray_comps_squared(
        varray_3d=varray_3d_kappa,
    )
    del varray_3d_kappa
    numpy.sqrt(sarray_3d_curvature, out=sarray_3d_curvature)
    return sarray_3d_curvature


##
## === MAGNETIC CURVATURE
##


@dataclass(frozen=True)
class MagneticCurvatureFArrays_3D:
    """Curvature, stretching, and compression sarrays derived from a 3D varray."""

    sarray_3d_curvature: NDArray[Any]
    sarray_3d_stretching: NDArray[Any]
    sarray_3d_compression: NDArray[Any]

    def __post_init__(
        self,
    ) -> None:
        ## validate each scalar decomposition output
        farray_types.ensure_3d_sarray(
            sarray_3d=self.sarray_3d_curvature,
            param_name="<sarray_3d_curvature>",
        )
        farray_types.ensure_3d_sarray(
            sarray_3d=self.sarray_3d_stretching,
            param_name="<sarray_3d_stretching>",
        )
        farray_types.ensure_3d_sarray(
            sarray_3d=self.sarray_3d_compression,
            param_name="<sarray_3d_compression>",
        )
        ## validate shared decomposition geometry
        if any([
                self.sarray_3d_curvature.shape != self.sarray_3d_stretching.shape,
                self.sarray_3d_curvature.shape != self.sarray_3d_compression.shape,
        ], ):
            raise ValueError(
                "MagneticCurvatureFArrays_3D components must share the same shape:"
                f" curvature={self.sarray_3d_curvature.shape},"
                f" stretching={self.sarray_3d_stretching.shape},"
                f" compression={self.sarray_3d_compression.shape}.",
            )


def compute_magnetic_curvature_farrays(
    *,
    varray_3d_u: NDArray[Any],
    uvarray_3d_tangent: NDArray[Any],
    uvarray_3d_normal: NDArray[Any],
    cell_widths_3d: tuple[float, float, float],
    r2tarray_3d_grad_u: NDArray[Any] | None = None,
    grad_order: int = 2,
) -> MagneticCurvatureFArrays_3D:
    """
    Compute curvature, stretching, and compression terms for a 3D velocity farray.

    Index notation (Einstein summation):
        curvature   = n_i n_j d_i u_j
        stretching  = t_i t_j d_i u_j
        compression = d_i u_i
    """
    farray_types.ensure_3d_varray(
        varray_3d=varray_3d_u,
        param_name="<varray_3d_u>",
    )
    farray_types.ensure_3d_varray(
        varray_3d=uvarray_3d_tangent,
        param_name="<uvarray_3d_tangent>",
    )
    farray_types.ensure_3d_varray(
        varray_3d=uvarray_3d_normal,
        param_name="<uvarray_3d_normal>",
    )
    if any([
            varray_3d_u.shape != uvarray_3d_tangent.shape,
            varray_3d_u.shape != uvarray_3d_normal.shape,
    ], ):
        raise ValueError(
            "Velocity/tangent/normal farrays must share the same shape:"
            f" u={varray_3d_u.shape},"
            f" tangent={uvarray_3d_tangent.shape},"
            f" normal={uvarray_3d_normal.shape}.",
        )
    farray_types.ensure_3d_cell_widths(cell_widths_3d)
    validate_types.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    ## d_i u_j: (j, i, x0, x1, x2)
    r2tarray_3d_grad_u = farray_operators.compute_varray_grad(
        varray_3d=varray_3d_u,
        cell_widths_3d=cell_widths_3d,
        r2tarray_3d_grad_f=r2tarray_3d_grad_u,
        grad_order=grad_order,
    )
    ## curvature = n_i n_j d_i u_j
    sarray_3d_curvature = numpy.einsum(
        "ixyz,jxyz,jixyz->xyz",
        uvarray_3d_normal,
        uvarray_3d_normal,
        r2tarray_3d_grad_u,
        optimize=True,
    )
    ## stretching = t_i t_j d_i u_j
    sarray_3d_stretching = numpy.einsum(
        "ixyz,jxyz,jixyz->xyz",
        uvarray_3d_tangent,
        uvarray_3d_tangent,
        r2tarray_3d_grad_u,
        optimize=True,
    )
    ## compression = d_i u_i = trace over (i,j)
    sarray_3d_compression = numpy.trace(
        r2tarray_3d_grad_u,
        axis1=0,  # grad-dir i
        axis2=1,  # comp j
    )
    del r2tarray_3d_grad_u
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
    """Lorentz force decomposition: total force, magnetic tension, and perpendicular pressure gradient farrays."""

    varray_3d_lorentz: NDArray[Any]
    varray_3d_tension: NDArray[Any]
    varray_3d_grad_p_perp: NDArray[Any]

    def __post_init__(
        self,
    ) -> None:
        ## validate each force component individually
        farray_types.ensure_3d_varray(
            varray_3d=self.varray_3d_lorentz,
            param_name="<varray_3d_lorentz>",
        )
        farray_types.ensure_3d_varray(
            varray_3d=self.varray_3d_tension,
            param_name="<varray_3d_tension>",
        )
        farray_types.ensure_3d_varray(
            varray_3d=self.varray_3d_grad_p_perp,
            param_name="<varray_3d_grad_p_perp>",
        )
        ## validate shared decomposition geometry
        if any([
                self.varray_3d_lorentz.shape != self.varray_3d_tension.shape,
                self.varray_3d_lorentz.shape != self.varray_3d_grad_p_perp.shape,
        ], ):
            raise ValueError(
                "LorentzForceFArrays_3D components must share the same shape:"
                f" lorentz={self.varray_3d_lorentz.shape},"
                f" tension={self.varray_3d_tension.shape},"
                f" grad_p_perp={self.varray_3d_grad_p_perp.shape}.",
            )


def compute_lorentz_force_farrays(
    *,
    varray_3d_b: NDArray[Any],
    cell_widths_3d: tuple[float, float, float],
    grad_order: int = 2,
) -> LorentzForceFArrays_3D:
    """
    Lorentz force decomposition in index notation:

        tension_i     = (b_k b_k) kappa_i
        grad_p_perp_i = d_i (b_k b_k / 2) - t_i t_j d_j (b_k b_k / 2)
        lorentz_i     = tension_i - grad_p_perp_i
    """
    farray_types.ensure_3d_varray(
        varray_3d=varray_3d_b,
        param_name="<varray_3d_b>",
    )
    farray_types.ensure_3d_cell_widths(cell_widths_3d)
    validate_types.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    ## TNB + curvature for b
    tnb_farrays = compute_tnb_farrays(
        varray_3d=varray_3d_b,
        cell_widths_3d=cell_widths_3d,
        grad_order=grad_order,
    )
    uvarray_3d_tangent = tnb_farrays.uvarray_3d_tangent
    uvarray_3d_normal = tnb_farrays.uvarray_3d_normal
    sarray_3d_curvature = tnb_farrays.sarray_3d_curvature
    ## |b|^2
    sarray_3d_b_magn_sq = farray_operators.sum_of_varray_comps_squared(
        varray_3d=varray_3d_b,
    )
    ## d_i p where p = 0.5 * |b|^2
    nabla = difference_sarrays.get_grad_fn(grad_order)
    cell_width_x, cell_width_y, cell_width_z = cell_widths_3d
    num_cells_x, num_cells_y, num_cells_z = sarray_3d_b_magn_sq.shape
    varray_3d_grad_p = farray_types.ensure_farray_metadata(
        farray_shape=(3, num_cells_x, num_cells_y, num_cells_z),
        farray=None,
        dtype=sarray_3d_b_magn_sq.dtype,
    )
    varray_3d_grad_p[0, ...] = nabla(
        sarray_3d=sarray_3d_b_magn_sq,
        cell_width=cell_width_x,
        grad_axis=0,
    )
    varray_3d_grad_p[1, ...] = nabla(
        sarray_3d=sarray_3d_b_magn_sq,
        cell_width=cell_width_y,
        grad_axis=1,
    )
    varray_3d_grad_p[2, ...] = nabla(
        sarray_3d=sarray_3d_b_magn_sq,
        cell_width=cell_width_z,
        grad_axis=2,
    )
    varray_3d_grad_p *= 0.5
    ## pressure aligned with b: t_i t_j d_j p
    varray_3d_grad_p_aligned = numpy.einsum(
        "ixyz,jxyz,jxyz->ixyz",
        uvarray_3d_tangent,
        uvarray_3d_tangent,
        varray_3d_grad_p,
        optimize=True,
    )
    del uvarray_3d_tangent
    ## tension_i = |b|^2 kappa_i
    sarray_3d_tension_scalar = sarray_3d_b_magn_sq * sarray_3d_curvature
    del sarray_3d_b_magn_sq, sarray_3d_curvature
    varray_3d_tension = sarray_3d_tension_scalar[numpy.newaxis, ...] * uvarray_3d_normal
    del sarray_3d_tension_scalar, uvarray_3d_normal
    ## grad_p_perp_i = d_i p - t_i t_j d_j p
    varray_3d_grad_p_perp = varray_3d_grad_p - varray_3d_grad_p_aligned
    del varray_3d_grad_p, varray_3d_grad_p_aligned
    ## lorentz_i = tension_i - grad_p_perp_i
    varray_3d_lorentz = varray_3d_tension - varray_3d_grad_p_perp
    return LorentzForceFArrays_3D(
        varray_3d_lorentz=varray_3d_lorentz,
        varray_3d_tension=varray_3d_tension,
        varray_3d_grad_p_perp=varray_3d_grad_p_perp,
    )


## } MODULE
