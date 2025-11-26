## { MODULE

##
## === DEPENDENCIES
##

import numpy

from jormi.ww_types import type_manager
from jormi.ww_fields.fields_3d import (
    _farray_operators,
    field_types,
)

##
## === OPTIMISED OPERATORS WORKING ON FIELDS
##


def compute_sfield_rms(
    sfield_3d: field_types.ScalarField_3D,
) -> float:
    """Compute the RMS of a 3D scalar field."""
    sarray_3d = field_types.extract_3d_sarray(
        sfield_3d=sfield_3d,
        param_name="<sfield_3d>",
    )
    return _farray_operators.compute_sarray_rms(
        sarray_3d=sarray_3d,
    )


def compute_sfield_volume_integral(
    sfield_3d: field_types.ScalarField_3D,
) -> float:
    """Compute the volume integral of a 3D scalar field."""
    sarray_3d = field_types.extract_3d_sarray(
        sfield_3d=sfield_3d,
        param_name="<sfield_3d>",
    )
    udomain_3d = sfield_3d.udomain
    return _farray_operators.compute_sarray_volume_integral(
        sarray_3d=sarray_3d,
        cell_volume=udomain_3d.cell_volume,
    )


def compute_sfield_gradient(
    *,
    sfield_3d: field_types.ScalarField_3D,
    varray_3d_out: numpy.ndarray | None = None,
    field_label: str = "d_i f",
    grad_order: int = 2,
) -> field_types.VectorField_3D:
    """Compute the gradient d_i f of a 3D scalar field."""
    type_manager.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    sarray_3d = field_types.extract_3d_sarray(
        sfield_3d=sfield_3d,
        param_name="<sfield_3d>",
    )
    udomain_3d = sfield_3d.udomain
    sim_time = sfield_3d.sim_time
    varray_3d_gradf = _farray_operators.compute_sarray_grad(
        sarray_3d=sarray_3d,
        cell_widths_3d=udomain_3d.cell_widths,
        varray_3d_out=varray_3d_out,
        grad_order=grad_order,
    )
    return field_types.VectorField_3D.from_3d_varray(
        varray_3d=varray_3d_gradf,
        udomain_3d=udomain_3d,
        field_label=field_label,
        sim_time=sim_time,
    )


def compute_vfield_magnitude(
    vfield_3d: field_types.VectorField_3D,
    *,
    field_label: str = "sqrt(f_i f_i)",
) -> field_types.ScalarField_3D:
    """Compute the magnitude sqrt(f_i f_i) of a 3D vector field."""
    varray_3d = field_types.extract_3d_varray(
        vfield_3d=vfield_3d,
        param_name="<vfield_3d>",
    )
    udomain_3d = vfield_3d.udomain
    sim_time = vfield_3d.sim_time
    sarray_3d_vmagn = _farray_operators.compute_varray_magnitude(
        varray_3d=varray_3d,
    )
    return field_types.ScalarField_3D.from_3d_sarray(
        sarray_3d=sarray_3d_vmagn,
        udomain_3d=udomain_3d,
        field_label=field_label,
        sim_time=sim_time,
    )


def compute_vfield_dot_product(
    *,
    vfield_3d_a: field_types.VectorField_3D,
    vfield_3d_b: field_types.VectorField_3D,
    field_label: str = "a_i b_i",
) -> field_types.ScalarField_3D:
    """Compute the dot product a_i b_i cellwise for two 3D vector fields."""
    field_types.ensure_same_3d_field_udomains(
        field_3d_a=vfield_3d_a,
        field_3d_b=vfield_3d_b,
        field_name_a="<vfield_3d_a>",
        field_name_b="<vfield_3d_b>",
    )
    varray_3d_a = field_types.extract_3d_varray(
        vfield_3d=vfield_3d_a,
        param_name="<vfield_3d_a>",
    )
    varray_3d_b = field_types.extract_3d_varray(
        vfield_3d=vfield_3d_b,
        param_name="<vfield_3d_b>",
    )
    sarray_3d_adotb = _farray_operators.dot_over_varray_comps(
        varray_3d_a=varray_3d_a,
        varray_3d_b=varray_3d_b,
    )
    return field_types.ScalarField_3D.from_3d_sarray(
        sarray_3d=sarray_3d_adotb,
        udomain_3d=vfield_3d_a.udomain,
        field_label=field_label,
        sim_time=vfield_3d_a.sim_time,
    )


def compute_vfield_cross_product(
    *,
    vfield_3d_a: field_types.VectorField_3D,
    vfield_3d_b: field_types.VectorField_3D,
    varray_3d_out: numpy.ndarray | None = None,
    sarray_3d_tmp: numpy.ndarray | None = None,
    field_label: str = "epsilon_ijk a_j b_k",
) -> field_types.VectorField_3D:
    """Compute the cross product epsilon_ijk a_j b_k cellwise for two 3D vector fields."""
    field_types.ensure_same_3d_field_udomains(
        field_3d_a=vfield_3d_a,
        field_3d_b=vfield_3d_b,
        field_name_a="<vfield_3d_a>",
        field_name_b="<vfield_3d_b>",
    )
    varray_3d_a = field_types.extract_3d_varray(
        vfield_3d=vfield_3d_a,
        param_name="<vfield_3d_a>",
    )
    varray_3d_b = field_types.extract_3d_varray(
        vfield_3d=vfield_3d_b,
        param_name="<vfield_3d_b>",
    )
    varray_3d_axb = _farray_operators.compute_varray_cross_product(
        varray_3d_a=varray_3d_a,
        varray_3d_b=varray_3d_b,
        varray_3d_out=varray_3d_out,
        sarray_3d_tmp=sarray_3d_tmp,
    )
    return field_types.VectorField_3D.from_3d_varray(
        varray_3d=varray_3d_axb,
        udomain_3d=vfield_3d_a.udomain,
        field_label=field_label,
        sim_time=vfield_3d_a.sim_time,
    )


def compute_vfield_curl(
    vfield_3d: field_types.VectorField_3D,
    *,
    varray_3d_out: numpy.ndarray | None = None,
    field_label: str = "epsilon_ijk d_j f_k",
    grad_order: int = 2,
) -> field_types.VectorField_3D:
    """Compute the curl epsilon_ijk d_j f_k of a 3D vector field."""
    type_manager.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    varray_3d = field_types.extract_3d_varray(
        vfield_3d=vfield_3d,
        param_name="<vfield_3d>",
    )
    udomain_3d = vfield_3d.udomain
    sim_time = vfield_3d.sim_time
    varray_3d_curl = _farray_operators.compute_varray_curl(
        varray_3d=varray_3d,
        cell_widths_3d=udomain_3d.cell_widths,
        varray_3d_out=varray_3d_out,
        grad_order=grad_order,
    )
    return field_types.VectorField_3D.from_3d_varray(
        varray_3d=varray_3d_curl,
        udomain_3d=udomain_3d,
        field_label=field_label,
        sim_time=sim_time,
    )


def compute_vfield_divergence(
    vfield_3d: field_types.VectorField_3D,
    *,
    sarray_3d_out: numpy.ndarray | None = None,
    field_label: str = "d_i f_i",
    grad_order: int = 2,
) -> field_types.ScalarField_3D:
    """Compute the divergence d_i f_i of a 3D vector field."""
    type_manager.ensure_finite_int(
        param=grad_order,
        param_name="<grad_order>",
        allow_none=False,
        require_positive=True,
    )
    varray_3d = field_types.extract_3d_varray(
        vfield_3d=vfield_3d,
        param_name="<vfield_3d>",
    )
    udomain_3d = vfield_3d.udomain
    sim_time = vfield_3d.sim_time
    sarray_3d_div = _farray_operators.compute_varray_divergence(
        varray_3d=varray_3d,
        cell_widths_3d=udomain_3d.cell_widths,
        sarray_3d_out=sarray_3d_out,
        grad_order=grad_order,
    )
    return field_types.ScalarField_3D.from_3d_sarray(
        sarray_3d=sarray_3d_div,
        udomain_3d=udomain_3d,
        field_label=field_label,
        sim_time=sim_time,
    )


## } MODULE
