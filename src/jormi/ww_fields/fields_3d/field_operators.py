## { MODULE

##
## === DEPENDENCIES
##

## third-party
from typing import Any

from numpy.typing import NDArray

## local
from jormi.ww_arrays.farrays_3d import farray_operators
from jormi.ww_fields.fields_3d import field_models
from jormi.ww_validation import validate_types

##
## === OPERATORS WORKING ON SCALAR FIELDS
##


def compute_sfield_rms(
    sfield_3d: field_models.ScalarField_3D,
) -> float:
    """Compute the RMS of a 3D scalar field."""
    sarray_3d = field_models.extract_3d_sarray(
        sfield_3d=sfield_3d,
        param_name="<sfield_3d>",
    )
    return farray_operators.compute_sarray_rms(
        sarray_3d=sarray_3d,
    )


def compute_sfield_volume_integral(
    sfield_3d: field_models.ScalarField_3D,
) -> float:
    """Compute the volume integral of a 3D scalar field."""
    sarray_3d = field_models.extract_3d_sarray(
        sfield_3d=sfield_3d,
        param_name="<sfield_3d>",
    )
    udomain_3d = sfield_3d.udomain
    return farray_operators.compute_sarray_volume_integral(
        sarray_3d=sarray_3d,
        cell_volume=udomain_3d.cell_volume,
    )


def compute_sfield_gradient(
    sfield_3d: field_models.ScalarField_3D,
    *,
    out_varray_3d: NDArray[Any] | None = None,
    field_name: str,
    latex_label: str,
    grad_order: int = 2,
) -> field_models.VectorField_3D:
    """Compute the gradient d_i f of a 3D scalar field."""
    validate_types.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    sarray_3d = field_models.extract_3d_sarray(
        sfield_3d=sfield_3d,
        param_name="<sfield_3d>",
    )
    udomain_3d = sfield_3d.udomain
    sim_time = sfield_3d.sim_time
    grad_f_varray_3d = farray_operators.compute_sarray_grad(
        sarray_3d=sarray_3d,
        cell_widths_3d=udomain_3d.cell_widths,
        out_varray_3d=out_varray_3d,
        grad_order=grad_order,
    )
    return field_models.VectorField_3D.from_3d_varray(
        varray_3d=grad_f_varray_3d,
        udomain_3d=udomain_3d,
        field_name=field_name,
        latex_label=latex_label,
        sim_time=sim_time,
    )


##
## === OPERATORS WORKING ON VECTOR FIELDS
##


def compute_vfield_magnitude(
    vfield_3d: field_models.VectorField_3D,
    *,
    field_name: str,
    latex_label: str,
) -> field_models.ScalarField_3D:
    """Compute the magnitude sqrt(f_i f_i) of a 3D vector field."""
    varray_3d = field_models.extract_3d_varray(
        vfield_3d=vfield_3d,
        param_name="<vfield_3d>",
    )
    udomain_3d = vfield_3d.udomain
    sim_time = vfield_3d.sim_time
    v_magn_sarray_3d = farray_operators.compute_varray_magnitude(
        varray_3d=varray_3d,
    )
    return field_models.ScalarField_3D.from_3d_sarray(
        sarray_3d=v_magn_sarray_3d,
        udomain_3d=udomain_3d,
        field_name=field_name,
        latex_label=latex_label,
        sim_time=sim_time,
    )


def compute_vfield_dot_product(
    *,
    f_vfield_3d: field_models.VectorField_3D,
    g_vfield_3d: field_models.VectorField_3D,
    field_name: str,
    latex_label: str,
) -> field_models.ScalarField_3D:
    """Compute the dot product a_i b_i cellwise for two 3D vector field_models."""
    field_models.ensure_same_3d_field_udomains(
        field_3d_a=f_vfield_3d,
        field_3d_b=g_vfield_3d,
        field_name_a="<f_vfield_3d>",
        field_name_b="<g_vfield_3d>",
    )
    f_varray_3d = field_models.extract_3d_varray(
        vfield_3d=f_vfield_3d,
        param_name="<f_vfield_3d>",
    )
    g_varray_3d = field_models.extract_3d_varray(
        vfield_3d=g_vfield_3d,
        param_name="<g_vfield_3d>",
    )
    adotb_sarray_3d = farray_operators.compute_dot_over_varray_comps(
        f_varray_3d=f_varray_3d,
        g_varray_3d=g_varray_3d,
    )
    return field_models.ScalarField_3D.from_3d_sarray(
        sarray_3d=adotb_sarray_3d,
        udomain_3d=f_vfield_3d.udomain,
        field_name=field_name,
        latex_label=latex_label,
        sim_time=f_vfield_3d.sim_time,
    )


def compute_vfield_cross_product(
    *,
    f_vfield_3d: field_models.VectorField_3D,
    g_vfield_3d: field_models.VectorField_3D,
    out_varray_3d: NDArray[Any] | None = None,
    tmp_sarray_3d: NDArray[Any] | None = None,
    field_name: str,
    latex_label: str,
) -> field_models.VectorField_3D:
    """Compute the cross product epsilon_ijk a_j b_k cellwise for two 3D vector field_models."""
    field_models.ensure_same_3d_field_udomains(
        field_3d_a=f_vfield_3d,
        field_3d_b=g_vfield_3d,
        field_name_a="<f_vfield_3d>",
        field_name_b="<g_vfield_3d>",
    )
    f_varray_3d = field_models.extract_3d_varray(
        vfield_3d=f_vfield_3d,
        param_name="<f_vfield_3d>",
    )
    g_varray_3d = field_models.extract_3d_varray(
        vfield_3d=g_vfield_3d,
        param_name="<g_vfield_3d>",
    )
    axb_varray_3d = farray_operators.compute_varray_cross_product(
        f_varray_3d=f_varray_3d,
        g_varray_3d=g_varray_3d,
        out_varray_3d=out_varray_3d,
        tmp_sarray_3d=tmp_sarray_3d,
    )
    return field_models.VectorField_3D.from_3d_varray(
        varray_3d=axb_varray_3d,
        udomain_3d=f_vfield_3d.udomain,
        field_name=field_name,
        latex_label=latex_label,
        sim_time=f_vfield_3d.sim_time,
    )


def compute_vfield_curl(
    vfield_3d: field_models.VectorField_3D,
    *,
    out_varray_3d: NDArray[Any] | None = None,
    field_name: str,
    latex_label: str,
    grad_order: int = 2,
) -> field_models.VectorField_3D:
    """Compute the curl epsilon_ijk d_j f_k of a 3D vector field."""
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
    curl_varray_3d = farray_operators.compute_varray_curl(
        varray_3d=varray_3d,
        cell_widths_3d=udomain_3d.cell_widths,
        out_varray_3d=out_varray_3d,
        grad_order=grad_order,
    )
    return field_models.VectorField_3D.from_3d_varray(
        varray_3d=curl_varray_3d,
        udomain_3d=udomain_3d,
        field_name=field_name,
        latex_label=latex_label,
        sim_time=sim_time,
    )


def compute_vfield_divergence(
    vfield_3d: field_models.VectorField_3D,
    *,
    out_sarray_3d: NDArray[Any] | None = None,
    field_name: str,
    latex_label: str,
    grad_order: int = 2,
) -> field_models.ScalarField_3D:
    """Compute the divergence d_i f_i of a 3D vector field."""
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
    div_sarray_3d = farray_operators.compute_varray_divergence(
        varray_3d=varray_3d,
        cell_widths_3d=udomain_3d.cell_widths,
        out_sarray_3d=out_sarray_3d,
        grad_order=grad_order,
    )
    return field_models.ScalarField_3D.from_3d_sarray(
        sarray_3d=div_sarray_3d,
        udomain_3d=udomain_3d,
        field_name=field_name,
        latex_label=latex_label,
        sim_time=sim_time,
    )


## } MODULE
