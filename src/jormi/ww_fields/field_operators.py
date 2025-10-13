## { MODULE

##
## === DEPENDENCIES
##

import numpy
from jormi.ww_data import array_types, array_operators, finite_difference
from jormi.ww_fields import field_types

##
## === OPTIMISED OPERATORS WORKING ON FIELDS
##


def compute_sfield_rms(
    sfield: field_types.ScalarField,
) -> float:
    field_types.ensure_sfield(sfield)
    return array_operators.compute_sarray_rms(sfield.data)


def compute_sfield_volume_integral(
    sfield: field_types.ScalarField,
    domain_details: field_types.UniformDomain,
) -> float:
    field_types.ensure_sfield(sfield)
    field_types.ensure_uniform_domain(domain_details)
    field_types.ensure_domain_matches_sfield(
        domain_details=domain_details,
        sfield=sfield,
    )
    return array_operators.compute_sarray_volume_integral(
        sarray=sfield.data,
        cell_volume=domain_details.cell_volume,
    )


def compute_sfield_gradient(
    sfield: field_types.ScalarField,
    domain_details: field_types.UniformDomain,
    out_varray: numpy.ndarray | None = None,
    labels: tuple[str, str, str] = (
        r"$\partial f/\partial x$",
        r"$\partial f/\partial y$",
        r"$\partial f/\partial z$",
    ),
    grad_order: int = 2,
) -> field_types.VectorField:
    field_types.ensure_sfield(sfield)
    field_types.ensure_uniform_domain(domain_details)
    field_types.ensure_domain_matches_sfield(
        domain_details=domain_details,
        sfield=sfield,
    )
    sarray = sfield.data
    array_types.ensure_sarray(sarray)
    grad_varray = array_operators.compute_sarray_grad(
        sarray=sarray,
        cell_widths=domain_details.cell_widths,
        grad_order=grad_order,
        out_varray=out_varray,
    )
    return field_types.VectorField(
        sim_time=sfield.sim_time,
        data=grad_varray,
        labels=labels,
    )


def compute_vfield_magnitude(
    vfield: field_types.VectorField,
    label: str = r"$|\vec(f)|$",
) -> field_types.ScalarField:
    field_types.ensure_vfield(vfield)
    varray = vfield.data
    array_types.ensure_varray(varray)
    field_magn = array_operators.sum_of_component_squares(varray)  # allocates output (reused below)
    numpy.sqrt(field_magn, out=field_magn)  # in-place transform
    return field_types.ScalarField(
        sim_time=vfield.sim_time,
        data=field_magn,
        label=label,
    )


def compute_vfield_dot_product(
    vfield_a: field_types.VectorField,
    vfield_b: field_types.VectorField,
    label: str = r"$\vec{a}\cdot\vec{b}$",
) -> field_types.ScalarField:
    field_types.ensure_vfield(vfield_a)
    field_types.ensure_vfield(vfield_b)
    varray_a = vfield_a.data
    varray_b = vfield_b.data
    array_types.ensure_varray(varray_a)
    array_types.ensure_varray(varray_b)
    array_types.ensure_same_shape(
        array_a=varray_a,
        array_b=varray_b,
    )
    dot_sarray = array_operators.dot_over_components(
        varray_a=varray_a,
        varray_b=varray_b,
    )
    return field_types.ScalarField(
        sim_time=vfield_a.sim_time,
        data=dot_sarray,
        label=label,
    )


def compute_vfield_cross_product(
    vfield_a: field_types.VectorField,
    vfield_b: field_types.VectorField,
    out_varray: numpy.ndarray | None = None,
    tmp_sarray: numpy.ndarray | None = None,
    labels: tuple[str, str, str] = (
        r"$(\vec{a}\times\vec{b})_x$",
        r"$(\vec{a}\times\vec{b})_y$",
        r"$(\vec{a}\times\vec{b})_z$",
    ),
) -> field_types.VectorField:
    field_types.ensure_vfield(vfield_a)
    field_types.ensure_vfield(vfield_b)
    varray_a = vfield_a.data
    varray_b = vfield_b.data
    array_types.ensure_varray(varray_a)
    array_types.ensure_varray(varray_b)
    array_types.ensure_same_shape(
        array_a=varray_a,
        array_b=varray_b,
    )
    domain_shape = varray_a.shape[1:]
    dtype = numpy.result_type(varray_a.dtype, varray_b.dtype)
    cross_varray = array_operators.ensure_array_properties(
        array_shape=varray_a.shape,
        dtype=dtype,
        array=out_varray,
    )
    array_types.ensure_varray(cross_varray)
    tmp_sarray = array_operators.ensure_array_properties(
        array_shape=domain_shape,
        dtype=dtype,
        array=tmp_sarray,
    )
    array_types.ensure_sarray(tmp_sarray)
    ## cross_x = a_y * b_z - a_z * b_y
    numpy.multiply(varray_a[1], varray_b[2], out=cross_varray[0])  # a_y * b_z
    numpy.multiply(varray_a[2], varray_b[1], out=tmp_sarray)  # a_z * b_y
    numpy.subtract(cross_varray[0], tmp_sarray, out=cross_varray[0])
    ## cross_y = -a_x * b_z + a_z * b_x
    numpy.multiply(varray_a[2], varray_b[0], out=cross_varray[1])  # a_z * b_x
    numpy.multiply(varray_a[0], varray_b[2], out=tmp_sarray)  # a_x * b_z
    numpy.subtract(cross_varray[1], tmp_sarray, out=cross_varray[1])
    ## cross_z = a_x * by - a_y * b_x
    numpy.multiply(varray_a[0], varray_b[1], out=cross_varray[2])  # a_x * b_y
    numpy.multiply(varray_a[1], varray_b[0], out=tmp_sarray)  # a_y * b_x
    numpy.subtract(cross_varray[2], tmp_sarray, out=cross_varray[2])
    return field_types.VectorField(
        sim_time=vfield_a.sim_time,
        data=cross_varray,
        labels=labels,
    )


def compute_vfield_curl(
    vfield: field_types.VectorField,
    domain_details: field_types.UniformDomain,
    grad_order: int = 2,
    out_varray: numpy.ndarray | None = None,
    labels: tuple[str, str, str] = (
        r"$(\nabla\times\vec{f})_x$",
        r"$(\nabla\times\vec{f})_y$",
        r"$(\nabla\times\vec{f})_z$",
    ),
) -> field_types.VectorField:
    field_types.ensure_vfield(vfield)
    field_types.ensure_uniform_domain(domain_details)
    field_types.ensure_domain_matches_vfield(
        domain_details=domain_details,
        vfield=vfield,
    )
    varray = vfield.data
    array_types.ensure_varray(varray)
    nabla = finite_difference.get_grad_func(grad_order)
    cell_width_x, cell_width_y, cell_width_z = domain_details.cell_widths
    curl_varray = array_operators.ensure_array_properties(
        array_shape=varray.shape,
        dtype=varray.dtype,
        array=out_varray,
    )
    array_types.ensure_varray(curl_varray)
    ## curl_x = dv_z/dy - dv_y/dz
    numpy.subtract(
        nabla(varray[2], cell_width_y, grad_axis=1),
        nabla(varray[1], cell_width_z, grad_axis=2),
        out=curl_varray[0],
    )
    ## curl_y = dv_x/dz - dv_z/dx
    numpy.subtract(
        nabla(varray[0], cell_width_z, grad_axis=2),
        nabla(varray[2], cell_width_x, grad_axis=0),
        out=curl_varray[1],
    )
    ## curl_z = dv_y/dx - dv_x/dy
    numpy.subtract(
        nabla(varray[1], cell_width_x, grad_axis=0),
        nabla(varray[0], cell_width_y, grad_axis=1),
        out=curl_varray[2],
    )
    return field_types.VectorField(
        sim_time=vfield.sim_time,
        data=curl_varray,
        labels=labels,
    )


def compute_vfield_divergence(
    vfield: field_types.VectorField,
    domain_details: field_types.UniformDomain,
    out_sarray: numpy.ndarray | None = None,
    label: str = r"$\nabla\cdot\vec{f}$",
    grad_order: int = 2,
) -> field_types.ScalarField:
    field_types.ensure_vfield(vfield)
    field_types.ensure_uniform_domain(domain_details)
    field_types.ensure_domain_matches_vfield(
        domain_details=domain_details,
        vfield=vfield,
    )
    varray = vfield.data
    array_types.ensure_varray(varray)
    nabla = finite_difference.get_grad_func(grad_order)
    cell_width_x, cell_width_y, cell_width_z = domain_details.cell_widths
    domain_shape = varray.shape[1:]
    div_sarray = array_operators.ensure_array_properties(
        array_shape=domain_shape,
        dtype=varray.dtype,
        array=out_sarray,
    )
    array_types.ensure_sarray(div_sarray)
    ## start with dv_x/dx, then add others in-place to avoid an extra tmp_sarray
    div_sarray[...] = nabla(varray[0], cell_width_x, grad_axis=0)
    numpy.add(div_sarray, nabla(varray[1], cell_width_y, grad_axis=1), out=div_sarray)
    numpy.add(div_sarray, nabla(varray[2], cell_width_z, grad_axis=2), out=div_sarray)
    return field_types.ScalarField(
        sim_time=vfield.sim_time,
        data=div_sarray,
        label=label,
    )


##
## === COMPUTING FIELD QUANTITIES
##


def compute_magnetic_energy_density(
    vfield: field_types.VectorField,
    energy_prefactor: float = 0.5,
    label: str = r"$E_\mathrm{mag}$",
) -> field_types.ScalarField:
    field_types.ensure_vfield(vfield)
    varray = vfield.data
    array_types.ensure_varray(varray)
    Emag_sarray = array_operators.sum_of_component_squares(varray)  # allocates output (reused below)
    Emag_sarray *= numpy.asarray(energy_prefactor, dtype=Emag_sarray.dtype)  # scale in-place
    return field_types.ScalarField(
        sim_time=vfield.sim_time,
        data=Emag_sarray,
        label=label,
    )


def compute_total_magnetic_energy(
    vfield: field_types.VectorField,
    domain_details: field_types.UniformDomain,
    energy_prefactor: float = 0.5,
) -> float:
    field_types.ensure_vfield(vfield)
    field_types.ensure_uniform_domain(domain_details)
    field_types.ensure_domain_matches_vfield(
        domain_details=domain_details,
        vfield=vfield,
    )
    Emag_sfield = compute_magnetic_energy_density(
        vfield=vfield,
        energy_prefactor=energy_prefactor,
    )
    return compute_sfield_volume_integral(
        sfield=Emag_sfield,
        domain_details=domain_details,
    )


## } MODULE
