## { MODULE

##
## === DEPENDENCIES
##

import numpy

from jormi.ww_types import fdata_types, field_types
from jormi.ww_fields import fdata_operators, finite_difference

##
## === OPTIMISED OPERATORS WORKING ON FIELDS
##


def compute_sfield_rms(
    sfield: field_types.ScalarField,
) -> float:
    field_types.ensure_sfield(sfield)
    return fdata_operators.compute_sdata_rms(
        sdata=sfield.fdata,
    )


def compute_sfield_volume_integral(
    sfield: field_types.ScalarField,
) -> float:
    field_types.ensure_sfield(sfield)
    udomain = sfield.udomain
    return fdata_operators.compute_sdata_volume_integral(
        sdata=sfield.fdata,
        cell_volume=udomain.cell_volume,
    )


def compute_sfield_gradient(
    *,
    sfield: field_types.ScalarField,
    out_varray: numpy.ndarray | None = None,
    field_label: str = r"$\nabla f$",
    grad_order: int = 2,
) -> field_types.VectorField:
    field_types.ensure_sfield(sfield)
    udomain = sfield.udomain
    sim_time = sfield.sim_time
    grad_varray = fdata_operators.compute_sdata_grad(
        sdata=sfield.fdata,
        cell_widths=udomain.cell_widths,
        out_varray=out_varray,
        grad_order=grad_order,
    )
    grad_fdata = fdata_types.VectorFieldData(
        farray=grad_varray,
        param_name="<grad.fdata>",
    )
    return field_types.VectorField(
        fdata=grad_fdata,
        udomain=udomain,
        field_label=field_label,
        sim_time=sim_time,
    )


def compute_vfield_magnitude(
    vfield: field_types.VectorField,
    *,
    field_label: str = r"$|\vec{f}|$",
) -> field_types.ScalarField:
    field_types.ensure_vfield(vfield)
    udomain = vfield.udomain
    sim_time = vfield.sim_time
    magn_sarray = fdata_operators.sum_of_squared_components(
        vdata=vfield.fdata,
    )
    numpy.sqrt(magn_sarray, out=magn_sarray)
    magn_fdata = fdata_types.ScalarFieldData(
        farray=magn_sarray,
        param_name="<vfield_magn.fdata>",
    )
    return field_types.ScalarField(
        fdata=magn_fdata,
        udomain=udomain,
        field_label=field_label,
        sim_time=sim_time,
    )


def compute_vfield_dot_product(
    *,
    vfield_a: field_types.VectorField,
    vfield_b: field_types.VectorField,
    field_label: str = r"$\vec{a}\cdot\vec{b}$",
) -> field_types.ScalarField:
    field_types.ensure_vfield(vfield_a)
    field_types.ensure_vfield(vfield_b)
    if vfield_a.udomain != vfield_b.udomain:
        raise ValueError("`vfield_a.udomain` must match `vfield_b.udomain`.")
    field_types.ensure_same_field_shape(
        field_a=vfield_a,
        field_b=vfield_b,
    )
    dot_sarray = fdata_operators.dot_over_components(
        vdata_a=vfield_a.fdata,
        vdata_b=vfield_b.fdata,
    )
    dot_fdata = fdata_types.ScalarFieldData(
        farray=dot_sarray,
        param_name="<vfield_dot.fdata>",
    )
    return field_types.ScalarField(
        fdata=dot_fdata,
        udomain=vfield_a.udomain,
        field_label=field_label,
        sim_time=vfield_a.sim_time,
    )


def compute_vfield_cross_product(
    *,
    vfield_a: field_types.VectorField,
    vfield_b: field_types.VectorField,
    out_varray: numpy.ndarray | None = None,
    tmp_sarray: numpy.ndarray | None = None,
    field_label: str = r"$\vec{a}\times\vec{b}$",
) -> field_types.VectorField:
    field_types.ensure_vfield(vfield_a)
    field_types.ensure_vfield(vfield_b)
    if vfield_a.udomain != vfield_b.udomain:
        raise ValueError("`vfield_a.udomain` must match `vfield_b.udomain`.")
    field_types.ensure_same_field_shape(
        field_a=vfield_a,
        field_b=vfield_b,
    )
    varray_a = vfield_a.fdata.farray
    varray_b = vfield_b.fdata.farray
    domain_shape = varray_a.shape[1:]
    dtype = numpy.result_type(varray_a.dtype, varray_b.dtype)
    cross_varray = fdata_operators.ensure_properties(
        farray_shape=varray_a.shape,
        dtype=dtype,
        farray=out_varray,
    )
    tmp_sarray = fdata_operators.ensure_properties(
        farray_shape=domain_shape,
        dtype=dtype,
        farray=tmp_sarray,
    )
    ## cross_x = a_y * b_z - a_z * b_y
    numpy.multiply(varray_a[1], varray_b[2], out=cross_varray[0])
    numpy.multiply(varray_a[2], varray_b[1], out=tmp_sarray)
    numpy.subtract(cross_varray[0], tmp_sarray, out=cross_varray[0])
    ## cross_y = -a_x * b_z + a_z * b_x
    numpy.multiply(varray_a[2], varray_b[0], out=cross_varray[1])
    numpy.multiply(varray_a[0], varray_b[2], out=tmp_sarray)
    numpy.subtract(cross_varray[1], tmp_sarray, out=cross_varray[1])
    ## cross_z = a_x * b_y - a_y * b_x
    numpy.multiply(varray_a[0], varray_b[1], out=cross_varray[2])
    numpy.multiply(varray_a[1], varray_b[0], out=tmp_sarray)
    numpy.subtract(cross_varray[2], tmp_sarray, out=cross_varray[2])
    cross_fdata = fdata_types.VectorFieldData(
        farray=cross_varray,
        param_name="<vfield_cross.fdata>",
    )
    return field_types.VectorField(
        fdata=cross_fdata,
        udomain=vfield_a.udomain,
        field_label=field_label,
        sim_time=vfield_a.sim_time,
    )


def compute_vfield_curl(
    vfield: field_types.VectorField,
    *,
    out_varray: numpy.ndarray | None = None,
    field_label: str = r"$\nabla\times\vec{f}$",
    grad_order: int = 2,
) -> field_types.VectorField:
    field_types.ensure_vfield(vfield)
    varray = vfield.fdata.farray
    udomain = vfield.udomain
    sim_time = vfield.sim_time
    nabla = finite_difference.get_grad_fn(grad_order)
    cell_width_x, cell_width_y, cell_width_z = udomain.cell_widths
    curl_varray = fdata_operators.ensure_properties(
        farray_shape=varray.shape,
        dtype=varray.dtype,
        farray=out_varray,
    )
    ## curl_x = d_y v_z - d_z v_y
    numpy.subtract(
        nabla(
            sarray=varray[2],
            cell_width=cell_width_y,
            grad_axis=1,
        ),
        nabla(
            sarray=varray[1],
            cell_width=cell_width_z,
            grad_axis=2,
        ),
        out=curl_varray[0],
    )
    ## curl_y = d_z v_x - d_x v_z
    numpy.subtract(
        nabla(
            sarray=varray[0],
            cell_width=cell_width_z,
            grad_axis=2,
        ),
        nabla(
            sarray=varray[2],
            cell_width=cell_width_x,
            grad_axis=0,
        ),
        out=curl_varray[1],
    )
    ## curl_z = d_x v_y - d_y v_x
    numpy.subtract(
        nabla(
            sarray=varray[1],
            cell_width=cell_width_x,
            grad_axis=0,
        ),
        nabla(
            sarray=varray[0],
            cell_width=cell_width_y,
            grad_axis=1,
        ),
        out=curl_varray[2],
    )
    curl_fdata = fdata_types.VectorFieldData(
        farray=curl_varray,
        param_name="<vfield_curl.fdata>",
    )
    return field_types.VectorField(
        fdata=curl_fdata,
        udomain=udomain,
        field_label=field_label,
        sim_time=sim_time,
    )


def compute_vfield_divergence(
    vfield: field_types.VectorField,
    *,
    out_sarray: numpy.ndarray | None = None,
    field_label: str = r"$\nabla\cdot\vec{f}$",
    grad_order: int = 2,
) -> field_types.ScalarField:
    field_types.ensure_vfield(vfield)
    varray = vfield.fdata.farray
    udomain = vfield.udomain
    sim_time = vfield.sim_time
    nabla = finite_difference.get_grad_fn(grad_order)
    cell_width_x, cell_width_y, cell_width_z = udomain.cell_widths
    domain_shape = varray.shape[1:]
    div_sarray = fdata_operators.ensure_properties(
        farray_shape=domain_shape,
        dtype=varray.dtype,
        farray=out_sarray,
    )
    ## start with d_i f_i for i=1, then add i=2 and i=3 in-place
    div_sarray[...] = nabla(
        sarray=varray[0],
        cell_width=cell_width_x,
        grad_axis=0,
    )
    numpy.add(
        div_sarray,
        nabla(
            sarray=varray[1],
            cell_width=cell_width_y,
            grad_axis=1,
        ),
        out=div_sarray,
    )
    numpy.add(
        div_sarray,
        nabla(
            sarray=varray[2],
            cell_width=cell_width_z,
            grad_axis=2,
        ),
        out=div_sarray,
    )
    div_fdata = fdata_types.ScalarFieldData(
        farray=div_sarray,
        param_name="<vfield_div.fdata>",
    )
    return field_types.ScalarField(
        fdata=div_fdata,
        udomain=udomain,
        field_label=field_label,
        sim_time=sim_time,
    )


##
## === COMPUTING FIELD QUANTITIES
##


def compute_magnetic_energy_density(
    b_vfield: field_types.VectorField,
    *,
    energy_prefactor: float = 0.5,
    field_label: str = r"$E_\mathrm{mag}$",
) -> field_types.ScalarField:
    field_types.ensure_vfield(b_vfield)
    udomain = b_vfield.udomain
    sim_time = b_vfield.sim_time
    Emag_sarray = fdata_operators.sum_of_squared_components(
        vdata=b_vfield.fdata,
    )
    Emag_sarray *= numpy.asarray(energy_prefactor, dtype=Emag_sarray.dtype)
    Emag_fdata = fdata_types.ScalarFieldData(
        farray=Emag_sarray,
        param_name="<Emag.fdata>",
    )
    return field_types.ScalarField(
        fdata=Emag_fdata,
        udomain=udomain,
        field_label=field_label,
        sim_time=sim_time,
    )


def compute_total_magnetic_energy(
    b_vfield: field_types.VectorField,
    *,
    energy_prefactor: float = 0.5,
) -> float:
    field_types.ensure_vfield(b_vfield)
    Emag_sfield = compute_magnetic_energy_density(
        b_vfield=b_vfield,
        energy_prefactor=energy_prefactor,
    )
    return compute_sfield_volume_integral(
        sfield=Emag_sfield,
    )


## } MODULE
