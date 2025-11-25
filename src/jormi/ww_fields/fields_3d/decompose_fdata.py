## { MODULE
##
## === DEPENDENCIES
##

from dataclasses import dataclass

from jormi.ww_fields.fields_3d import (
    fdata_types,
    decompose_farrays,
)


##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class HelmholtzDecomposedFData_3D:
    """Helmholtz decomposition of VectorFieldData_3D into div/sol/bulk parts."""

    vdata_3d_div: fdata_types.VectorFieldData_3D
    vdata_3d_sol: fdata_types.VectorFieldData_3D
    vdata_3d_bulk: fdata_types.VectorFieldData_3D

    def __post_init__(
        self,
    ) -> None:
        fdata_types.ensure_3d_vdata(
            vdata_3d=self.vdata_3d_div,
            param_name="<vdata_3d_div>",
        )
        fdata_types.ensure_3d_vdata(
            vdata_3d=self.vdata_3d_sol,
            param_name="<vdata_3d_sol>",
        )
        fdata_types.ensure_3d_vdata(
            vdata_3d=self.vdata_3d_bulk,
            param_name="<vdata_3d_bulk>",
        )
        if any([
                self.vdata_3d_div.farray.shape != self.vdata_3d_sol.farray.shape,
                self.vdata_3d_div.farray.shape != self.vdata_3d_bulk.farray.shape,
        ]):
            raise ValueError(
                "HelmholtzDecomposedFData_3D components must share the same shape:"
                f" div={self.vdata_3d_div.farray.shape},"
                f" sol={self.vdata_3d_sol.farray.shape},"
                f" bulk={self.vdata_3d_bulk.farray.shape}.",
            )


@dataclass(frozen=True)
class TNBDecomposedFData_3D:
    """TNB decomposition of VectorFieldData_3D into unit bases and curvature."""

    vdata_3d_tangent: fdata_types.VectorFieldData_3D
    vdata_3d_normal: fdata_types.VectorFieldData_3D
    vdata_3d_binormal: fdata_types.VectorFieldData_3D
    sdata_3d_curvature: fdata_types.ScalarFieldData_3D

    def __post_init__(
        self,
    ) -> None:
        fdata_types.ensure_3d_vdata(
            vdata_3d=self.vdata_3d_tangent,
            param_name="<vdata_3d_tangent>",
        )
        fdata_types.ensure_3d_vdata(
            vdata_3d=self.vdata_3d_normal,
            param_name="<vdata_3d_normal>",
        )
        fdata_types.ensure_3d_vdata(
            vdata_3d=self.vdata_3d_binormal,
            param_name="<vdata_3d_binormal>",
        )
        fdata_types.ensure_3d_sdata(
            sdata_3d=self.sdata_3d_curvature,
            param_name="<sdata_3d_curvature>",
        )
        if any([
                self.vdata_3d_tangent.farray.shape != self.vdata_3d_normal.farray.shape,
                self.vdata_3d_tangent.farray.shape != self.vdata_3d_binormal.farray.shape,
        ]):
            raise ValueError(
                "TNBDecomposedFData_3D vector components must share the same shape:"
                f" tangent={self.vdata_3d_tangent.farray.shape},"
                f" normal={self.vdata_3d_normal.farray.shape},"
                f" binormal={self.vdata_3d_binormal.farray.shape}.",
            )
        if self.vdata_3d_tangent.farray.shape[1:] != self.sdata_3d_curvature.farray.shape:
            raise ValueError(
                "TNBDecomposedFData_3D curvature shape must match spatial shape of"
                f" vectors: curvature={self.sdata_3d_curvature.farray.shape},"
                f" vectors={self.vdata_3d_tangent.farray.shape[1:]}.",
            )


##
## === FUNCTIONS
##


def compute_helmholtz_decomposition(
    *,
    vdata_3d_q: fdata_types.VectorFieldData_3D,
    resolution: tuple[int, int, int],
    cell_widths: tuple[float, float, float],
) -> HelmholtzDecomposedFData_3D:
    """Helmholtz decompose a VectorFieldData_3D into (div, sol, bulk)."""
    fdata_types.ensure_3d_vdata(
        vdata_3d=vdata_3d_q,
        param_name="<vdata_3d_q>",
    )
    varray_3d_q = fdata_types.as_3d_varray(
        vdata_3d=vdata_3d_q,
        param_name="<vdata_3d_q>",
    )
    helmholtz_3d_farrays = decompose_farrays.compute_helmholtz_decomposition(
        varray_3d_q=varray_3d_q,
        resolution=resolution,
        cell_widths=cell_widths,
    )
    vdata_3d_div = fdata_types.VectorFieldData_3D(
        farray=helmholtz_3d_farrays.varray_3d_div,
        param_name="<helmholtz.vdata_3d_div>",
    )
    vdata_3d_sol = fdata_types.VectorFieldData_3D(
        farray=helmholtz_3d_farrays.varray_3d_sol,
        param_name="<helmholtz.vdata_3d_sol>",
    )
    vdata_3d_bulk = fdata_types.VectorFieldData_3D(
        farray=helmholtz_3d_farrays.varray_3d_bulk,
        param_name="<helmholtz.vdata_3d_bulk>",
    )
    return HelmholtzDecomposedFData_3D(
        vdata_3d_div=vdata_3d_div,
        vdata_3d_sol=vdata_3d_sol,
        vdata_3d_bulk=vdata_3d_bulk,
    )


def compute_tnb_terms(
    *,
    vdata_3d: fdata_types.VectorFieldData_3D,
    cell_widths: tuple[float, float, float],
    grad_order: int = 2,
) -> TNBDecomposedFData_3D:
    """Compute T, N, B and curvature from a VectorFieldData_3D."""
    fdata_types.ensure_3d_vdata(
        vdata_3d=vdata_3d,
        param_name="<vdata_3d>",
    )
    varray_3d = fdata_types.as_3d_varray(
        vdata_3d=vdata_3d,
        param_name="<vdata_3d>",
    )
    tnb_3d_farrays = decompose_farrays.compute_tnb_terms(
        varray_3d=varray_3d,
        cell_widths=cell_widths,
        grad_order=grad_order,
    )
    vdata_3d_tangent = fdata_types.VectorFieldData_3D(
        farray=tnb_3d_farrays.uvarray_3d_tangent,
        param_name="<tnb.vdata_3d_tangent>",
    )
    vdata_3d_normal = fdata_types.VectorFieldData_3D(
        farray=tnb_3d_farrays.uvarray_3d_normal,
        param_name="<tnb.vdata_3d_normal>",
    )
    vdata_3d_binormal = fdata_types.VectorFieldData_3D(
        farray=tnb_3d_farrays.uvarray_3d_binormal,
        param_name="<tnb.vdata_3d_binormal>",
    )
    sdata_3d_curvature = fdata_types.ScalarFieldData_3D(
        farray=tnb_3d_farrays.sarray_3d_curvature,
        param_name="<tnb.sdata_3d_curvature>",
    )
    return TNBDecomposedFData_3D(
        vdata_3d_tangent=vdata_3d_tangent,
        vdata_3d_normal=vdata_3d_normal,
        vdata_3d_binormal=vdata_3d_binormal,
        sdata_3d_curvature=sdata_3d_curvature,
    )


## } MODULE
