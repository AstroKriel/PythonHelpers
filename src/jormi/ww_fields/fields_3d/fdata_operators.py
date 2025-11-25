## { MODULE

##
## === DEPENDENCIES
##

import numpy

from jormi.ww_types import array_checks
from jormi.ww_fields.fields_3d import (
    fdata_types,
    farray_operators,
)


##
## === SCALAR FDATA OPERATORS
##


def compute_sdata_rms(
    sdata_3d: fdata_types.ScalarFieldData_3D,
) -> float:
    """Compute the RMS of a ScalarFieldData_3D object."""
    sarray_3d = fdata_types.as_3d_sarray(
        sdata_3d=sdata_3d,
        param_name="<sdata_3d>",
    )
    return farray_operators.compute_sarray_rms(
        sarray_3d=sarray_3d,
    )


def compute_sdata_volume_integral(
    sdata_3d: fdata_types.ScalarFieldData_3D,
    *,
    cell_volume: float,
) -> float:
    """Compute the volume integral of a ScalarFieldData_3D object."""
    sarray_3d = fdata_types.as_3d_sarray(
        sdata_3d=sdata_3d,
        param_name="<sdata_3d>",
    )
    return farray_operators.compute_sarray_volume_integral(
        sarray_3d=sarray_3d,
        cell_volume=cell_volume,
    )


def compute_sdata_grad(
    *,
    sdata_3d: fdata_types.ScalarFieldData_3D,
    cell_widths: tuple[float, float, float] | list[float],
    varray_3d_out: numpy.ndarray | None = None,
    grad_order: int = 2,
) -> fdata_types.VectorFieldData_3D:
    """Compute the gradient of a 3D scalar field, returning VectorFieldData_3D."""
    sarray_3d = fdata_types.as_3d_sarray(
        sdata_3d=sdata_3d,
        param_name="<sdata_3d>",
    )
    varray_3d_gradf = farray_operators.compute_sarray_grad(
        sarray_3d=sarray_3d,
        cell_widths=cell_widths,
        varray_3d_out=varray_3d_out,
        grad_order=grad_order,
    )
    fdata_types.ensure_3d_varray(
        varray_3d=varray_3d_gradf,
        param_name="<varray_3d_gradf>",
    )
    return fdata_types.VectorFieldData_3D(
        farray=varray_3d_gradf,
        param_name="<sdata_3d_grad>",
    )


##
## === VECTOR FDATA OPERATORS
##


def sum_of_vdata_comps_squared(
    *,
    vdata_3d: fdata_types.VectorFieldData_3D,
    sarray_3d_out: numpy.ndarray | None = None,
    sarray_3d_tmp: numpy.ndarray | None = None,
) -> fdata_types.ScalarFieldData_3D:
    """Compute sum_i (v_i v_i) per cell for a 3D VectorFieldData_3D."""
    varray_3d = fdata_types.as_3d_varray(
        vdata_3d=vdata_3d,
        param_name="<vdata_3d>",
    )
    sarray_3d = farray_operators.sum_of_varray_comps_squared(
        varray_3d=varray_3d,
        sarray_3d_out=sarray_3d_out,
        sarray_3d_tmp=sarray_3d_tmp,
    )
    fdata_types.ensure_3d_sarray(
        sarray_3d=sarray_3d,
        param_name="<sdata_3d_from_vdata_sq>",
    )
    return fdata_types.ScalarFieldData_3D(
        farray=sarray_3d,
        param_name="<sdata_3d_from_vdata_sq>",
    )


def dot_over_vdata_comps(
    *,
    vdata_3d_a: fdata_types.VectorFieldData_3D,
    vdata_3d_b: fdata_types.VectorFieldData_3D,
    sarray_3d_out: numpy.ndarray | None = None,
    sarray_3d_tmp: numpy.ndarray | None = None,
) -> fdata_types.ScalarFieldData_3D:
    """Compute vec(a) dot vec(b) per cell for 3D VectorFieldData_3D objects."""
    varray_3d_a = fdata_types.as_3d_varray(
        vdata_3d=vdata_3d_a,
        param_name="<vdata_3d_a>",
    )
    varray_3d_b = fdata_types.as_3d_varray(
        vdata_3d=vdata_3d_b,
        param_name="<vdata_3d_b>",
    )
    array_checks.ensure_same_shape(
        array_a=varray_3d_a,
        array_b=varray_3d_b,
        param_name_a="<vdata_3d_a.farray>",
        param_name_b="<vdata_3d_b.farray>",
    )
    sarray_3d = farray_operators.dot_over_varray_comps(
        varray_3d_a=varray_3d_a,
        varray_3d_b=varray_3d_b,
        sarray_3d_out=sarray_3d_out,
        sarray_3d_tmp=sarray_3d_tmp,
    )
    fdata_types.ensure_3d_sarray(
        sarray_3d=sarray_3d,
        param_name="<sdata_3d_dot>",
    )
    return fdata_types.ScalarFieldData_3D(
        farray=sarray_3d,
        param_name="<sdata_3d_dot>",
    )


def compute_vdata_grad(
    *,
    vdata_3d: fdata_types.VectorFieldData_3D,
    cell_widths: tuple[float, float, float] | list[float],
    r2tarray_3d_gradf: numpy.ndarray | None = None,
    grad_order: int = 2,
) -> fdata_types.Rank2TensorData_3D:
    """Compute gradient of a 3D vector field, returning Rank2TensorData_3D."""
    varray_3d = fdata_types.as_3d_varray(
        vdata_3d=vdata_3d,
        param_name="<vdata_3d>",
    )
    r2tarray_3d = farray_operators.compute_varray_grad(
        varray_3d=varray_3d,
        cell_widths=cell_widths,
        r2tarray_3d_gradf=r2tarray_3d_gradf,
        grad_order=grad_order,
    )
    fdata_types.ensure_3d_r2tarray(
        r2tarray_3d=r2tarray_3d,
        param_name="<r2tarray_3d_gradf>",
    )
    return fdata_types.Rank2TensorData_3D(
        farray=r2tarray_3d,
        param_name="<vdata_3d_grad>",
    )


def compute_vdata_cross_product(
    *,
    vdata_3d_a: fdata_types.VectorFieldData_3D,
    vdata_3d_b: fdata_types.VectorFieldData_3D,
    varray_3d_out: numpy.ndarray | None = None,
    sarray_3d_tmp: numpy.ndarray | None = None,
) -> fdata_types.VectorFieldData_3D:
    """Compute epsilon_ijk a_j b_k cellwise for two 3D VectorFieldData_3D."""
    varray_3d_a = fdata_types.as_3d_varray(
        vdata_3d=vdata_3d_a,
        param_name="<vdata_3d_a>",
    )
    varray_3d_b = fdata_types.as_3d_varray(
        vdata_3d=vdata_3d_b,
        param_name="<vdata_3d_b>",
    )
    array_checks.ensure_same_shape(
        array_a=varray_3d_a,
        array_b=varray_3d_b,
        param_name_a="<vdata_3d_a.farray>",
        param_name_b="<vdata_3d_b.farray>",
    )
    varray_3d_axb = farray_operators.compute_varray_cross_product(
        varray_3d_a=varray_3d_a,
        varray_3d_b=varray_3d_b,
        varray_3d_out=varray_3d_out,
        sarray_3d_tmp=sarray_3d_tmp,
    )
    fdata_types.ensure_3d_varray(
        varray_3d=varray_3d_axb,
        param_name="<vdata_3d_axb.farray>",
    )
    return fdata_types.VectorFieldData_3D(
        farray=varray_3d_axb,
        param_name="<vdata_3d_axb>",
    )


def compute_vdata_curl(
    *,
    vdata_3d: fdata_types.VectorFieldData_3D,
    cell_widths: tuple[float, float, float] | list[float],
    varray_3d_out: numpy.ndarray | None = None,
    grad_order: int = 2,
) -> fdata_types.VectorFieldData_3D:
    """Compute curl epsilon_ijk d_j f_k for a 3D VectorFieldData_3D."""
    varray_3d = fdata_types.as_3d_varray(
        vdata_3d=vdata_3d,
        param_name="<vdata_3d>",
    )
    varray_3d_curl = farray_operators.compute_varray_curl(
        varray_3d=varray_3d,
        cell_widths=cell_widths,
        varray_3d_out=varray_3d_out,
        grad_order=grad_order,
    )
    fdata_types.ensure_3d_varray(
        varray_3d=varray_3d_curl,
        param_name="<vdata_3d_curl.farray>",
    )
    return fdata_types.VectorFieldData_3D(
        farray=varray_3d_curl,
        param_name="<vdata_3d_curl>",
    )


def compute_vdata_divergence(
    *,
    vdata_3d: fdata_types.VectorFieldData_3D,
    cell_widths: tuple[float, float, float] | list[float],
    sarray_3d_out: numpy.ndarray | None = None,
    grad_order: int = 2,
) -> fdata_types.ScalarFieldData_3D:
    """Compute divergence d_i f_i for a 3D VectorFieldData_3D."""
    varray_3d = fdata_types.as_3d_varray(
        vdata_3d=vdata_3d,
        param_name="<vdata_3d>",
    )
    sarray_3d_div = farray_operators.compute_varray_divergence(
        varray_3d=varray_3d,
        cell_widths=cell_widths,
        sarray_3d_out=sarray_3d_out,
        grad_order=grad_order,
    )
    fdata_types.ensure_3d_sarray(
        sarray_3d=sarray_3d_div,
        param_name="<sdata_3d_div.farray>",
    )
    return fdata_types.ScalarFieldData_3D(
        farray=sarray_3d_div,
        param_name="<sdata_3d_div>",
    )


def compute_vdata_magnitude(
    *,
    vdata_3d: fdata_types.VectorFieldData_3D,
    sarray_3d_out: numpy.ndarray | None = None,
    sarray_3d_tmp: numpy.ndarray | None = None,
) -> fdata_types.ScalarFieldData_3D:
    """Compute |v| = sqrt(v_i v_i) per cell for a 3D VectorFieldData_3D."""
    varray_3d = fdata_types.as_3d_varray(
        vdata_3d=vdata_3d,
        param_name="<vdata_3d>",
    )
    sarray_3d_vmagn = farray_operators.compute_varray_magnitude(
        varray_3d=varray_3d,
        sarray_3d_out=sarray_3d_out,
        sarray_3d_tmp=sarray_3d_tmp,
    )
    fdata_types.ensure_3d_sarray(
        sarray_3d=sarray_3d_vmagn,
        param_name="<sdata_3d_vmagn.farray>",
    )
    return fdata_types.ScalarFieldData_3D(
        farray=sarray_3d_vmagn,
        param_name="<sdata_3d_vmagn>",
    )


def compute_magnetic_energy_density_sdata(
    vdata_3d_b: fdata_types.VectorFieldData_3D,
    *,
    energy_prefactor: float = 0.5,
) -> fdata_types.ScalarFieldData_3D:
    """
    Compute magnetic energy density from a 3D magnetic field (proportional to b_i b_i),
    returning ScalarFieldData_3D.
    """
    sdata_3d_sum_b_sq = sum_of_vdata_comps_squared(
        vdata_3d=vdata_3d_b,
    )
    sarray_3d_sum_b_sq = fdata_types.as_3d_sarray(
        sdata_3d=sdata_3d_sum_b_sq,
        param_name="<sdata_3d_sum_b_sq>",
    )
    farray_operators.scale_sarray_inplace(
        sarray_3d=sarray_3d_sum_b_sq,
        scale=energy_prefactor,
    )
    return fdata_types.ScalarFieldData_3D(
        farray=sarray_3d_sum_b_sq,
        param_name="<sdata_3d_sum_b_sq>",
    )


## } MODULE
