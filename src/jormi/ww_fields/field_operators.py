## { MODULE

##
## === DEPENDENCIES
##

import numpy
from jormi.utils import array_utils
from jormi.ww_fields import farray_types, farray_operators, finite_difference, field_types

##
## === OPTIMISED OPERATORS WORKING ON FIELDS
##


def compute_sfield_rms(
    sfield: field_types.ScalarField,
) -> float:
    field_types.ensure_sfield(sfield)
    return farray_operators.compute_sarray_rms(sfield.data)


def compute_sfield_volume_integral(
    sfield: field_types.ScalarField,
    uniform_domain: field_types.UniformDomain,
) -> float:
    field_types.ensure_sfield(sfield)
    field_types.ensure_uniform_domain(uniform_domain)
    field_types.ensure_domain_matches_sfield(
        uniform_domain=uniform_domain,
        sfield=sfield,
    )
    return farray_operators.compute_sarray_volume_integral(
        sarray=sfield.data,
        cell_volume=uniform_domain.cell_volume,
    )


def compute_sfield_gradient(
    sfield: field_types.ScalarField,
    uniform_domain: field_types.UniformDomain,
    out_varray: numpy.ndarray | None = None,
    field_label: str = r"$\nabla f$",
    grad_order: int = 2,
) -> field_types.VectorField:
    field_types.ensure_sfield(sfield)
    field_types.ensure_uniform_domain(uniform_domain)
    field_types.ensure_domain_matches_sfield(
        uniform_domain=uniform_domain,
        sfield=sfield,
    )
    sim_time = sfield.sim_time
    sarray = sfield.data
    farray_types.ensure_sarray(sarray)
    grad_varray = farray_operators.compute_sarray_grad(
        sarray=sarray,
        cell_widths=uniform_domain.cell_widths,
        grad_order=grad_order,
        out_varray=out_varray,
    )
    return field_types.VectorField(
        sim_time=sim_time,
        data=grad_varray,
        field_label=field_label,
    )


def compute_vfield_magnitude(
    vfield: field_types.VectorField,
    field_label: str = r"$|\vec{f}|$",
) -> field_types.ScalarField:
    field_types.ensure_vfield(vfield)
    sim_time = vfield.sim_time
    varray = vfield.data
    farray_types.ensure_varray(varray)
    field_magn = farray_operators.sum_of_squared_components(varray)  # allocates output (reused below)
    numpy.sqrt(field_magn, out=field_magn)  # in-place transform
    return field_types.ScalarField(
        sim_time=sim_time,
        data=field_magn,
        field_label=field_label,
    )


def compute_vfield_dot_product(
    vfield_a: field_types.VectorField,
    vfield_b: field_types.VectorField,
    field_label: str = r"$\vec{a}\cdot\vec{b}$",
) -> field_types.ScalarField:
    field_types.ensure_vfield(vfield_a)
    field_types.ensure_vfield(vfield_b)
    varray_a = vfield_a.data
    varray_b = vfield_b.data
    farray_types.ensure_varray(varray_a)
    farray_types.ensure_varray(varray_b)
    array_utils.ensure_same_shape(
        array_a=varray_a,
        array_b=varray_b,
    )
    dot_sarray = farray_operators.dot_over_components(
        varray_a=varray_a,
        varray_b=varray_b,
    )
    return field_types.ScalarField(
        sim_time=vfield_a.sim_time,
        data=dot_sarray,
        field_label=field_label,
    )


def compute_vfield_cross_product(
    vfield_a: field_types.VectorField,
    vfield_b: field_types.VectorField,
    out_varray: numpy.ndarray | None = None,
    tmp_sarray: numpy.ndarray | None = None,
    field_label: str = r"$\vec{a}\times\vec{b}$",
) -> field_types.VectorField:
    field_types.ensure_vfield(vfield_a)
    field_types.ensure_vfield(vfield_b)
    varray_a = vfield_a.data
    varray_b = vfield_b.data
    farray_types.ensure_varray(varray_a)
    farray_types.ensure_varray(varray_b)
    array_utils.ensure_same_shape(
        array_a=varray_a,
        array_b=varray_b,
    )
    domain_shape = varray_a.shape[1:]
    dtype = numpy.result_type(varray_a.dtype, varray_b.dtype)
    cross_varray = farray_operators.ensure_properties(
        array_shape=varray_a.shape,
        dtype=dtype,
        array=out_varray,
    )
    farray_types.ensure_varray(cross_varray)
    tmp_sarray = farray_operators.ensure_properties(
        array_shape=domain_shape,
        dtype=dtype,
        array=tmp_sarray,
    )
    farray_types.ensure_sarray(tmp_sarray)
    ## cross_x = a_y * b_z - a_z * b_y
    numpy.multiply(varray_a[1], varray_b[2], out=cross_varray[0])  # out[0] = a_y * b_z
    numpy.multiply(varray_a[2], varray_b[1], out=tmp_sarray)  # tmp = a_z * b_y
    numpy.subtract(cross_varray[0], tmp_sarray, out=cross_varray[0])  # out[0] = a_y * b_z - a_z * b_y
    ## cross_y = -a_x * b_z + a_z * b_x
    numpy.multiply(varray_a[2], varray_b[0], out=cross_varray[1])  # out[1] = a_z * b_x
    numpy.multiply(varray_a[0], varray_b[2], out=tmp_sarray)  # tmp = a_x * b_z
    numpy.subtract(cross_varray[1], tmp_sarray, out=cross_varray[1])  # out[1] = a_z * b_x - a_x * b_z
    ## cross_z = a_x * b_y - a_y * b_x
    numpy.multiply(varray_a[0], varray_b[1], out=cross_varray[2])  # out[2] = a_x * b_y
    numpy.multiply(varray_a[1], varray_b[0], out=tmp_sarray)  # tmp = a_y * b_x
    numpy.subtract(cross_varray[2], tmp_sarray, out=cross_varray[2])  # out[2] = a_x * b_y - a_y * b_x
    return field_types.VectorField(
        sim_time=vfield_a.sim_time,
        data=cross_varray,
        field_label=field_label,
    )


def compute_vfield_curl(
    vfield: field_types.VectorField,
    uniform_domain: field_types.UniformDomain,
    out_varray: numpy.ndarray | None = None,
    field_label: str = r"$\nabla\times\vec{f}$",
    grad_order: int = 2,
) -> field_types.VectorField:
    field_types.ensure_vfield(vfield)
    field_types.ensure_uniform_domain(uniform_domain)
    field_types.ensure_domain_matches_vfield(
        uniform_domain=uniform_domain,
        vfield=vfield,
    )
    sim_time = vfield.sim_time
    varray = vfield.data
    farray_types.ensure_varray(varray)
    nabla = finite_difference.get_grad_func(grad_order)
    cell_width_x, cell_width_y, cell_width_z = uniform_domain.cell_widths
    curl_varray = farray_operators.ensure_properties(
        array_shape=varray.shape,
        dtype=varray.dtype,
        array=out_varray,
    )
    farray_types.ensure_varray(curl_varray)
    ## curl_x = dv_z/dy - dv_y/dz
    numpy.subtract(
        nabla(sarray=varray[2], cell_width=cell_width_y, grad_axis=1),
        nabla(sarray=varray[1], cell_width=cell_width_z, grad_axis=2),
        out=curl_varray[0],
    )
    ## curl_y = dv_x/dz - dv_z/dx
    numpy.subtract(
        nabla(sarray=varray[0], cell_width=cell_width_z, grad_axis=2),
        nabla(sarray=varray[2], cell_width=cell_width_x, grad_axis=0),
        out=curl_varray[1],
    )
    ## curl_z = dv_y/dx - dv_x/dy
    numpy.subtract(
        nabla(sarray=varray[1], cell_width=cell_width_x, grad_axis=0),
        nabla(sarray=varray[0], cell_width=cell_width_y, grad_axis=1),
        out=curl_varray[2],
    )
    return field_types.VectorField(
        sim_time=sim_time,
        data=curl_varray,
        field_label=field_label,
    )


def compute_vfield_divergence(
    vfield: field_types.VectorField,
    uniform_domain: field_types.UniformDomain,
    out_sarray: numpy.ndarray | None = None,
    field_label: str = r"$\nabla\cdot\vec{f}$",
    grad_order: int = 2,
) -> field_types.ScalarField:
    field_types.ensure_vfield(vfield)
    field_types.ensure_uniform_domain(uniform_domain)
    field_types.ensure_domain_matches_vfield(
        uniform_domain=uniform_domain,
        vfield=vfield,
    )
    sim_time = vfield.sim_time
    varray = vfield.data
    farray_types.ensure_varray(varray)
    nabla = finite_difference.get_grad_func(grad_order)
    cell_width_x, cell_width_y, cell_width_z = uniform_domain.cell_widths
    domain_shape = varray.shape[1:]
    div_sarray = farray_operators.ensure_properties(
        array_shape=domain_shape,
        dtype=varray.dtype,
        array=out_sarray,
    )
    farray_types.ensure_sarray(div_sarray)
    ## start with dv_x/dx, then add others in-place to avoid an extra tmp_sarray
    div_sarray[...] = nabla(sarray=varray[0], cell_width=cell_width_x, grad_axis=0)
    numpy.add(div_sarray, nabla(sarray=varray[1], cell_width=cell_width_y, grad_axis=1), out=div_sarray)
    numpy.add(div_sarray, nabla(sarray=varray[2], cell_width=cell_width_z, grad_axis=2), out=div_sarray)
    return field_types.ScalarField(
        sim_time=sim_time,
        data=div_sarray,
        field_label=field_label,
    )


##
## === COMPUTING FIELD QUANTITIES
##


def compute_magnetic_energy_density(
    vfield: field_types.VectorField,
    energy_prefactor: float = 0.5,
    field_label: str = r"$E_\mathrm{mag}$",
) -> field_types.ScalarField:
    field_types.ensure_vfield(vfield)
    sim_time = vfield.sim_time
    varray = vfield.data
    farray_types.ensure_varray(varray)
    Emag_sarray = farray_operators.sum_of_squared_components(varray)  # allocates output (reused below)
    Emag_sarray *= numpy.asarray(energy_prefactor, dtype=Emag_sarray.dtype)  # scale in-place
    return field_types.ScalarField(
        sim_time=sim_time,
        data=Emag_sarray,
        field_label=field_label,
    )


def compute_total_magnetic_energy(
    vfield: field_types.VectorField,
    uniform_domain: field_types.UniformDomain,
    energy_prefactor: float = 0.5,
) -> float:
    field_types.ensure_vfield(vfield)
    field_types.ensure_uniform_domain(uniform_domain)
    field_types.ensure_domain_matches_vfield(
        uniform_domain=uniform_domain,
        vfield=vfield,
    )
    Emag_sfield = compute_magnetic_energy_density(
        vfield=vfield,
        energy_prefactor=energy_prefactor,
    )
    return compute_sfield_volume_integral(
        sfield=Emag_sfield,
        uniform_domain=uniform_domain,
    )


## } MODULE
