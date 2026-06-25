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

    div_varray_3d: NDArray[Any]
    sol_varray_3d: NDArray[Any]
    bulk_varray_3d: NDArray[Any]

    def __post_init__(
        self,
    ) -> None:
        ## validate each decomposed vector field
        farray_types.ensure_3d_varray(
            varray_3d=self.div_varray_3d,
            param_name="<div_varray_3d>",
        )
        farray_types.ensure_3d_varray(
            varray_3d=self.sol_varray_3d,
            param_name="<sol_varray_3d>",
        )
        farray_types.ensure_3d_varray(
            varray_3d=self.bulk_varray_3d,
            param_name="<bulk_varray_3d>",
        )
        ## validate shared decomposition geometry
        if any([
                self.div_varray_3d.shape != self.sol_varray_3d.shape,
                self.div_varray_3d.shape != self.bulk_varray_3d.shape,
        ]):
            raise ValueError(
                "HelmholtzDecomposedFArrays_3D components must share the same shape:"
                f" div={self.div_varray_3d.shape},"
                f" sol={self.sol_varray_3d.shape},"
                f" bulk={self.bulk_varray_3d.shape}.",
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
    fft_varray_3d = numpy.fft.fftn(
        varray_3d,
        axes=(1, 2, 3),
        norm="forward",
    )
    fft_bulk_varray_3d = numpy.zeros_like(fft_varray_3d)
    fft_bulk_varray_3d[:, 0, 0, 0] = fft_varray_3d[:, 0, 0, 0]
    fft_varray_3d[:, 0, 0, 0] = 0.0
    k_dot_fft_sarray_3d = (
        kx_grid * fft_varray_3d[0] + ky_grid * fft_varray_3d[1] + kz_grid * fft_varray_3d[2]
    )
    with numpy.errstate(
            divide="ignore",
            invalid="ignore",
    ):
        fft_div_varray_3d = numpy.stack(
            [
                (kx_grid / k_magn_grid) * k_dot_fft_sarray_3d,
                (ky_grid / k_magn_grid) * k_dot_fft_sarray_3d,
                (kz_grid / k_magn_grid) * k_dot_fft_sarray_3d,
            ],
            axis=0,
        )
    del kx_grid, ky_grid, kz_grid, k_magn_grid, k_dot_fft_sarray_3d
    fft_sol_varray_3d = fft_varray_3d - fft_div_varray_3d
    del fft_varray_3d
    div_varray_3d = numpy.fft.ifftn(
        fft_div_varray_3d,
        axes=(1, 2, 3),
        norm="forward",
    ).real.astype(
        dtype,
        copy=False,
    )
    del fft_div_varray_3d
    sol_varray_3d = numpy.fft.ifftn(
        fft_sol_varray_3d,
        axes=(1, 2, 3),
        norm="forward",
    ).real.astype(
        dtype,
        copy=False,
    )
    del fft_sol_varray_3d
    bulk_varray_3d = numpy.fft.ifftn(
        fft_bulk_varray_3d,
        axes=(1, 2, 3),
        norm="forward",
    ).real.astype(
        dtype,
        copy=False,
    )
    del fft_bulk_varray_3d
    return HelmholtzDecomposedFArrays_3D(
        div_varray_3d=div_varray_3d,
        sol_varray_3d=sol_varray_3d,
        bulk_varray_3d=bulk_varray_3d,
    )


##
## === TNB BASIS
##


@dataclass(frozen=True)
class TNBDecomposedFArrays_3D:
    """TNB decomposition of a 3D varray into unit bases and curvature."""

    tangent_uvarray_3d: NDArray[Any]
    normal_uvarray_3d: NDArray[Any]
    binormal_uvarray_3d: NDArray[Any]
    curvature_sarray_3d: NDArray[Any]

    def __post_init__(
        self,
    ) -> None:
        ## validate the TNB basis arrays individually
        farray_types.ensure_3d_varray(
            varray_3d=self.tangent_uvarray_3d,
            param_name="<tangent_uvarray_3d>",
        )
        farray_types.ensure_3d_varray(
            varray_3d=self.normal_uvarray_3d,
            param_name="<normal_uvarray_3d>",
        )
        farray_types.ensure_3d_varray(
            varray_3d=self.binormal_uvarray_3d,
            param_name="<binormal_uvarray_3d>",
        )
        farray_types.ensure_3d_sarray(
            sarray_3d=self.curvature_sarray_3d,
            param_name="<curvature_sarray_3d>",
        )
        ## validate that the vector basis shares one grid
        if any([
                self.tangent_uvarray_3d.shape != self.normal_uvarray_3d.shape,
                self.tangent_uvarray_3d.shape != self.binormal_uvarray_3d.shape,
        ]):
            raise ValueError(
                "TNBDecomposedFArrays_3D vector components must share the same shape:"
                f" tangent={self.tangent_uvarray_3d.shape},"
                f" normal={self.normal_uvarray_3d.shape},"
                f" binormal={self.binormal_uvarray_3d.shape}.",
            )
        ## validate curvature against the vector spatial grid
        if self.tangent_uvarray_3d.shape[1:] != self.curvature_sarray_3d.shape:
            raise ValueError(
                "TNBDecomposedFArrays_3D curvature shape must match spatial shape of"
                f" vectors: curvature={self.curvature_sarray_3d.shape},"
                f" vectors={self.tangent_uvarray_3d.shape[1:]}.",
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
      - tangent_uvarray_3d   (3, num_x0_cells, num_x1_cells, num_x2_cells)
      - normal_uvarray_3d    (3, num_x0_cells, num_x1_cells, num_x2_cells)
      - binormal_uvarray_3d  (3, num_x0_cells, num_x1_cells, num_x2_cells)
      - curvature_sarray_3d  (num_x0_cells, num_x1_cells, num_x2_cells)
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
    f_magn_sq_sarray_3d = farray_operators.compute_sum_of_varray_comps_squared(
        varray_3d=varray_3d,
    )
    ## |f| = sqrt(f_i f_i)
    f_magn_sarray_3d = numpy.sqrt(
        f_magn_sq_sarray_3d,
        dtype=f_magn_sq_sarray_3d.dtype,
    )
    ## t_i = f_i / |f|
    tangent_uvarray_3d = numpy.zeros_like(varray_3d)
    numpy.divide(
        varray_3d,
        f_magn_sarray_3d,
        out=tangent_uvarray_3d,
        where=(f_magn_sarray_3d > 0),
    )
    del f_magn_sarray_3d
    ## grad f: d_i f_j, layout (j, i, x0, x1, x2)
    grad_f_r2tarray_3d = farray_operators.compute_varray_grad(
        varray_3d=varray_3d,
        cell_widths_3d=cell_widths_3d,
        grad_f_r2tarray_3d=None,
        grad_order=grad_order,
    )
    ## term1_j = f_i * (d_i f_j)
    normal_term1_varray_3d = numpy.einsum(
        "ixyz,jixyz->jxyz",
        varray_3d,
        grad_f_r2tarray_3d,
        optimize=True,
    )
    ## term2_j = f_i f_j f_m (d_i f_m)
    normal_term2_varray_3d = numpy.einsum(
        "ixyz,jxyz,mxyz,mixyz->jxyz",
        varray_3d,
        varray_3d,
        varray_3d,
        grad_f_r2tarray_3d,
        optimize=True,
    )
    del grad_f_r2tarray_3d
    ## curvature vector kappa_j and magnitude |kappa|
    inv_magn_sq_sarray_3d = numpy.zeros_like(f_magn_sq_sarray_3d)
    numpy.divide(
        1.0,
        f_magn_sq_sarray_3d,
        out=inv_magn_sq_sarray_3d,
        where=(f_magn_sq_sarray_3d > 0),
    )
    del f_magn_sq_sarray_3d
    inv_magn4_sarray_3d = inv_magn_sq_sarray_3d**2
    kappa_varray_3d = (
        normal_term1_varray_3d * inv_magn_sq_sarray_3d - normal_term2_varray_3d * inv_magn4_sarray_3d
    )
    del (
        normal_term1_varray_3d,
        normal_term2_varray_3d,
        inv_magn_sq_sarray_3d,
        inv_magn4_sarray_3d,
    )
    curvature_sarray_3d = farray_operators.compute_sum_of_varray_comps_squared(
        varray_3d=kappa_varray_3d,
    )
    numpy.sqrt(curvature_sarray_3d, out=curvature_sarray_3d)
    ## n_i = kappa_i / |kappa|
    normal_uvarray_3d = numpy.zeros_like(kappa_varray_3d)
    numpy.divide(
        kappa_varray_3d,
        curvature_sarray_3d,
        out=normal_uvarray_3d,
        where=(curvature_sarray_3d > 0.0),
    )
    del kappa_varray_3d
    ## b_i = (t x n)_i
    binormal_uvarray_3d = farray_operators.compute_varray_cross_product(
        a_varray_3d=tangent_uvarray_3d,
        b_varray_3d=normal_uvarray_3d,
    )
    return TNBDecomposedFArrays_3D(
        tangent_uvarray_3d=tangent_uvarray_3d,
        normal_uvarray_3d=normal_uvarray_3d,
        binormal_uvarray_3d=binormal_uvarray_3d,
        curvature_sarray_3d=curvature_sarray_3d,
    )


def compute_curvature_sarray(
    *,
    varray_3d: NDArray[Any],
    cell_widths_3d: tuple[float, float, float],
    grad_order: int = 2,
) -> NDArray[Any]:
    """
    Compute curvature magnitude sqrt(kappa_i kappa_i) from a 3D varray.

    This implementation is optimised to compute the scalar curvature magnitude
    and avoids building the full curvature vector first.
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
    f_magn_sq_sarray_3d = farray_operators.compute_sum_of_varray_comps_squared(
        varray_3d=varray_3d,
    )
    ## term1_j = f_i * (d_i f_j) = ((f . grad) f)_j
    normal_term1_varray_3d = farray_operators.compute_varray_directional_derivative(
        target_varray_3d=varray_3d,
        along_varray_3d=varray_3d,
        cell_widths_3d=cell_widths_3d,
        grad_order=grad_order,
    )
    ## term2_j = f_j * [f_m * term1_m]
    normal_prefactor_sarray_3d = farray_operators.compute_dot_over_varray_comps(
        a_varray_3d=varray_3d,
        b_varray_3d=normal_term1_varray_3d,
    )
    inv_magn_sq_sarray_3d = numpy.zeros_like(f_magn_sq_sarray_3d)
    numpy.divide(
        1.0,
        f_magn_sq_sarray_3d,
        out=inv_magn_sq_sarray_3d,
        where=(f_magn_sq_sarray_3d > 0),
    )
    del f_magn_sq_sarray_3d
    inv_magn4_sarray_3d = inv_magn_sq_sarray_3d**2
    kappa_varray_3d = (
        normal_term1_varray_3d * inv_magn_sq_sarray_3d -
        varray_3d * normal_prefactor_sarray_3d * inv_magn4_sarray_3d
    )
    del (
        normal_term1_varray_3d,
        normal_prefactor_sarray_3d,
        inv_magn_sq_sarray_3d,
        inv_magn4_sarray_3d,
    )
    ## |kappa| = sqrt(kappa_i kappa_i)
    curvature_sarray_3d = farray_operators.compute_sum_of_varray_comps_squared(
        varray_3d=kappa_varray_3d,
    )
    del kappa_varray_3d
    numpy.sqrt(curvature_sarray_3d, out=curvature_sarray_3d)
    return curvature_sarray_3d


##
## === MAGNETIC CURVATURE
##


@dataclass(frozen=True)
class MagneticCurvatureFArrays_3D:
    """Curvature, stretching, and compression sarrays derived from a 3D varray."""

    curvature_sarray_3d: NDArray[Any]
    stretching_sarray_3d: NDArray[Any]
    compression_sarray_3d: NDArray[Any]

    def __post_init__(
        self,
    ) -> None:
        ## validate each scalar decomposition output
        farray_types.ensure_3d_sarray(
            sarray_3d=self.curvature_sarray_3d,
            param_name="<curvature_sarray_3d>",
        )
        farray_types.ensure_3d_sarray(
            sarray_3d=self.stretching_sarray_3d,
            param_name="<stretching_sarray_3d>",
        )
        farray_types.ensure_3d_sarray(
            sarray_3d=self.compression_sarray_3d,
            param_name="<compression_sarray_3d>",
        )
        ## validate shared decomposition geometry
        if any([
                self.curvature_sarray_3d.shape != self.stretching_sarray_3d.shape,
                self.curvature_sarray_3d.shape != self.compression_sarray_3d.shape,
        ]):
            raise ValueError(
                "MagneticCurvatureFArrays_3D components must share the same shape:"
                f" curvature={self.curvature_sarray_3d.shape},"
                f" stretching={self.stretching_sarray_3d.shape},"
                f" compression={self.compression_sarray_3d.shape}.",
            )


def compute_magnetic_curvature_farrays(
    *,
    v_varray_3d: NDArray[Any],
    tangent_uvarray_3d: NDArray[Any],
    normal_uvarray_3d: NDArray[Any],
    cell_widths_3d: tuple[float, float, float],
    grad_v_r2tarray_3d: NDArray[Any] | None = None,
    grad_order: int = 2,
) -> MagneticCurvatureFArrays_3D:
    """
    Compute the curvature, stretching, and compression terms contributing to the magnetic amplitude and curvature coupling.

    Index notation (Einstein summation):
        curvature   = n_i n_j d_i u_j
        stretching  = t_i t_j d_i u_j
        compression = d_i u_i
    """
    farray_types.ensure_3d_varray(
        varray_3d=v_varray_3d,
        param_name="<v_varray_3d>",
    )
    farray_types.ensure_3d_varray(
        varray_3d=tangent_uvarray_3d,
        param_name="<tangent_uvarray_3d>",
    )
    farray_types.ensure_3d_varray(
        varray_3d=normal_uvarray_3d,
        param_name="<normal_uvarray_3d>",
    )
    if any([
            v_varray_3d.shape != tangent_uvarray_3d.shape,
            v_varray_3d.shape != normal_uvarray_3d.shape,
    ]):
        raise ValueError(
            "the input velocity field and tangent/normal magnetic field farrays must share the same shape:"
            f" u={v_varray_3d.shape},"
            f" tangent={tangent_uvarray_3d.shape},"
            f" normal={normal_uvarray_3d.shape}.",
        )
    farray_types.ensure_3d_cell_widths(cell_widths_3d)
    validate_types.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    ## d_i u_j: (j, i, x0, x1, x2)
    grad_v_r2tarray_3d = farray_operators.compute_varray_grad(
        varray_3d=v_varray_3d,
        cell_widths_3d=cell_widths_3d,
        grad_f_r2tarray_3d=grad_v_r2tarray_3d,
        grad_order=grad_order,
    )
    ## curvature = n_i n_j d_i u_j
    curvature_sarray_3d = numpy.einsum(
        "ixyz,jxyz,jixyz->xyz",
        normal_uvarray_3d,
        normal_uvarray_3d,
        grad_v_r2tarray_3d,
        optimize=True,
    )
    ## stretching = t_i t_j d_i u_j
    stretching_sarray_3d = numpy.einsum(
        "ixyz,jxyz,jixyz->xyz",
        tangent_uvarray_3d,
        tangent_uvarray_3d,
        grad_v_r2tarray_3d,
        optimize=True,
    )
    ## compression = d_i u_i = trace over (i,j)
    compression_sarray_3d = numpy.trace(
        grad_v_r2tarray_3d,
        axis1=0,  # grad-dir i
        axis2=1,  # comp j
    )
    del grad_v_r2tarray_3d
    return MagneticCurvatureFArrays_3D(
        curvature_sarray_3d=curvature_sarray_3d,
        stretching_sarray_3d=stretching_sarray_3d,
        compression_sarray_3d=compression_sarray_3d,
    )


def compute_stretching_farray(
    *,
    v_varray_3d: NDArray[Any],
    b_varray_3d: NDArray[Any],
    cell_widths_3d: tuple[float, float, float],
    grad_order: int = 2,
) -> NDArray[Any]:
    """Compute the dynamo stretching term t_i t_j d_i u_j.

    Index notation (Einstein summation):
        stretching = t_i t_j d_i u_j,  t_i = b_i / |b|

    Leaner than compute_magnetic_curvature_farrays: skips the curvature and
    compression terms and avoids computing the B-field gradient (no normal vector
    needed), so peak memory is lower.
    """
    farray_types.ensure_3d_varray(varray_3d=v_varray_3d, param_name="<v_varray_3d>")
    farray_types.ensure_3d_varray(varray_3d=b_varray_3d, param_name="<b_varray_3d>")
    if v_varray_3d.shape != b_varray_3d.shape:
        raise ValueError(
            "velocity and magnetic field farrays must share the same shape:"
            f" u={v_varray_3d.shape}, b={b_varray_3d.shape}.",
        )
    farray_types.ensure_3d_cell_widths(cell_widths_3d)
    ## t_i = b_i / |b|
    b_magn_sq = farray_operators.compute_sum_of_varray_comps_squared(varray_3d=b_varray_3d)
    b_magn    = numpy.sqrt(b_magn_sq, dtype=b_magn_sq.dtype)
    del b_magn_sq
    tangent_uvarray_3d = numpy.zeros_like(b_varray_3d)
    numpy.divide(b_varray_3d, b_magn, out=tangent_uvarray_3d, where=(b_magn > 0))
    del b_magn
    ## d_i u_j: (j, i, x0, x1, x2)
    grad_v_r2tarray_3d = farray_operators.compute_varray_grad(
        varray_3d=v_varray_3d,
        cell_widths_3d=cell_widths_3d,
        grad_order=grad_order,
    )
    ## t_i t_j d_i u_j
    stretching_sarray_3d = numpy.einsum(
        "ixyz,jxyz,jixyz->xyz",
        tangent_uvarray_3d,
        tangent_uvarray_3d,
        grad_v_r2tarray_3d,
        optimize=True,
    )
    del grad_v_r2tarray_3d, tangent_uvarray_3d
    return stretching_sarray_3d


##
## === LORENTZ FORCE
##


@dataclass(frozen=True)
class LorentzForceFArrays_3D:
    """Lorentz force decomposition: total force, magnetic tension, and perpendicular pressure gradient farrays."""

    lorentz_varray_3d: NDArray[Any]
    tension_varray_3d: NDArray[Any]
    grad_p_perp_varray_3d: NDArray[Any]

    def __post_init__(
        self,
    ) -> None:
        ## validate each force component individually
        farray_types.ensure_3d_varray(
            varray_3d=self.lorentz_varray_3d,
            param_name="<lorentz_varray_3d>",
        )
        farray_types.ensure_3d_varray(
            varray_3d=self.tension_varray_3d,
            param_name="<tension_varray_3d>",
        )
        farray_types.ensure_3d_varray(
            varray_3d=self.grad_p_perp_varray_3d,
            param_name="<grad_p_perp_varray_3d>",
        )
        ## validate shared decomposition geometry
        if any([
                self.lorentz_varray_3d.shape != self.tension_varray_3d.shape,
                self.lorentz_varray_3d.shape != self.grad_p_perp_varray_3d.shape,
        ]):
            raise ValueError(
                "LorentzForceFArrays_3D components must share the same shape:"
                f" lorentz={self.lorentz_varray_3d.shape},"
                f" tension={self.tension_varray_3d.shape},"
                f" grad_p_perp={self.grad_p_perp_varray_3d.shape}.",
            )


def compute_lorentz_force_farrays(
    *,
    b_varray_3d: NDArray[Any],
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
        varray_3d=b_varray_3d,
        param_name="<b_varray_3d>",
    )
    farray_types.ensure_3d_cell_widths(cell_widths_3d)
    validate_types.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    ## TNB + curvature for b
    tnb_farrays_3d = compute_tnb_farrays(
        varray_3d=b_varray_3d,
        cell_widths_3d=cell_widths_3d,
        grad_order=grad_order,
    )
    tangent_uvarray_3d = tnb_farrays_3d.tangent_uvarray_3d
    normal_uvarray_3d = tnb_farrays_3d.normal_uvarray_3d
    curvature_sarray_3d = tnb_farrays_3d.curvature_sarray_3d
    ## |b|^2
    b_magn_sq_sarray_3d = farray_operators.compute_sum_of_varray_comps_squared(
        varray_3d=b_varray_3d,
    )
    ## d_i p where p = 0.5 * |b|^2
    nabla = difference_sarrays.get_grad_fn(grad_order)
    cell_width_x, cell_width_y, cell_width_z = cell_widths_3d
    num_cells_x, num_cells_y, num_cells_z = b_magn_sq_sarray_3d.shape
    grad_p_varray_3d = farray_types.ensure_farray_metadata(
        farray_shape=(3, num_cells_x, num_cells_y, num_cells_z),
        farray=None,
        dtype=b_magn_sq_sarray_3d.dtype,
    )
    grad_p_varray_3d[0, ...] = nabla(
        sarray_3d=b_magn_sq_sarray_3d,
        cell_width=cell_width_x,
        grad_axis=0,
    )
    grad_p_varray_3d[1, ...] = nabla(
        sarray_3d=b_magn_sq_sarray_3d,
        cell_width=cell_width_y,
        grad_axis=1,
    )
    grad_p_varray_3d[2, ...] = nabla(
        sarray_3d=b_magn_sq_sarray_3d,
        cell_width=cell_width_z,
        grad_axis=2,
    )
    grad_p_varray_3d *= 0.5
    ## pressure aligned with b: t_i t_j d_j p
    grad_p_aligned_varray_3d = numpy.einsum(
        "ixyz,jxyz,jxyz->ixyz",
        tangent_uvarray_3d,
        tangent_uvarray_3d,
        grad_p_varray_3d,
        optimize=True,
    )
    del tangent_uvarray_3d
    ## tension_i = |b|^2 kappa_i
    tension_scalar_sarray_3d = b_magn_sq_sarray_3d * curvature_sarray_3d
    del b_magn_sq_sarray_3d, curvature_sarray_3d
    tension_varray_3d = tension_scalar_sarray_3d[numpy.newaxis, ...] * normal_uvarray_3d
    del tension_scalar_sarray_3d, normal_uvarray_3d
    ## grad_p_perp_i = d_i p - t_i t_j d_j p
    grad_p_perp_varray_3d = grad_p_varray_3d - grad_p_aligned_varray_3d
    del grad_p_varray_3d, grad_p_aligned_varray_3d
    ## lorentz_i = tension_i - grad_p_perp_i
    lorentz_varray_3d = tension_varray_3d - grad_p_perp_varray_3d
    return LorentzForceFArrays_3D(
        lorentz_varray_3d=lorentz_varray_3d,
        tension_varray_3d=tension_varray_3d,
        grad_p_perp_varray_3d=grad_p_perp_varray_3d,
    )


## } MODULE
