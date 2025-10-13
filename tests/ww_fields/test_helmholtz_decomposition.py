## { TEST

##
## === DEPENDENCIES
##

import numpy
from jormi.utils import list_utils
from jormi.ww_io import io_manager
from jormi.ww_plots import plot_manager
from jormi.ww_fields import field_types, field_operators, decompose_fields

##
## === EXAMPLE VECTOR FIELDS
##


def generate_curl_free_vfield(domain_bounds, num_cells):
    """Generate a curl-free (irrotational) vector field."""
    domain = numpy.linspace(domain_bounds[0], domain_bounds[1], int(num_cells))
    grid_x, grid_y, grid_z = numpy.meshgrid(domain, domain, domain, indexing="ij")
    sfield_qx = 2 * grid_x
    sfield_qy = 2 * grid_y
    sfield_qz = 2 * grid_z
    return field_types.VectorField(
        data=numpy.stack([sfield_qx, sfield_qy, sfield_qz]),
        labels=("q_x", "q_y", "q_z"),
    )


def generate_div_free_vfield(domain_bounds, num_cells):
    """Generate a solenoidal (divergence-free) vector field."""
    domain_length = domain_bounds[1] - domain_bounds[0]
    domain = numpy.linspace(domain_bounds[0], domain_bounds[1], int(num_cells))
    k = 2 * numpy.pi / domain_length
    grid_x, grid_y, grid_z = numpy.meshgrid(domain, domain, domain, indexing="ij")
    sfield_qx = -k * grid_x * numpy.sin(k * grid_x * grid_y)
    sfield_qy =  k * grid_y * numpy.sin(k * grid_x * grid_y)
    sfield_qz = numpy.zeros_like(grid_z)
    return field_types.VectorField(
        data=numpy.stack([sfield_qx, sfield_qy, sfield_qz]),
        labels=("q_x", "q_y", "q_z"),
    )


def generate_mixed_vfield(domain_bounds, num_cells):
    v_div = generate_curl_free_vfield(domain_bounds, num_cells)
    v_sol = generate_div_free_vfield(domain_bounds, num_cells)
    return field_types.VectorField(
        data=v_div.data + v_sol.data,
        labels=("q_x", "q_y", "q_z"),
    )


##
## === HELPER FUNCTIONS
##


def compute_field_fraction(bin_edges, pdf):
    nonzero_indices = numpy.where(pdf > 0)[0]
    if len(nonzero_indices) > 0:
        first_percent = bin_edges[nonzero_indices[0]]
        last_percent = bin_edges[nonzero_indices[-1]]
        return first_percent if first_percent == last_percent else (last_percent - first_percent)
    return 0.0


def plot_vfield_sliceSlice(ax, vfield: field_types.VectorField, domain_bounds):
    _, num_cells_x, num_cells_y, num_cells_z = vfield.data.shape
    index_z = num_cells_z // 2  ## middle slice in the z-direction
    grid_x, grid_y = numpy.meshgrid(
        numpy.linspace(domain_bounds[0], domain_bounds[1], num_cells_x),
        numpy.linspace(domain_bounds[0], domain_bounds[1], num_cells_y),
        indexing="xy",
    )
    sfield_q_magn = field_operators.compute_vfield_magnitude(vfield)
    sfield_q_magn_slice = sfield_q_magn.data[:, :, index_z]
    sfield_q_magn_min = float(numpy.min(sfield_q_magn_slice))
    sfield_q_magn_max = float(numpy.max(sfield_q_magn_slice))
    ax.imshow(
        sfield_q_magn_slice.T,
        origin="lower",
        extent=[domain_bounds[0], domain_bounds[1], domain_bounds[0], domain_bounds[1]],
        cmap="viridis",
        alpha=0.7,
    )
    ax.streamplot(
        grid_x,
        grid_y,
        vfield.data[0, :, :, index_z],
        vfield.data[1, :, :, index_z],
        color="black",
        arrowstyle="->",
        linewidth=2.0,
        density=1.0,
        arrowsize=1.0,
        broken_streamlines=False,
    )
    ax.text(
        0.5,
        0.95,
        f"magnitude: [{sfield_q_magn_min:.2e}, {sfield_q_magn_max:.2e}]",
        va="top",
        ha="center",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
    )
    ax.set_xlim([domain_bounds[0], domain_bounds[1]])
    ax.set_ylim([domain_bounds[0], domain_bounds[1]])
    ax.set_xticks([])
    ax.set_yticks([])


##
## === ORTHOGONAL DECOMPOSITION TEST
##


def main():
    num_cells = 50
    domain_bounds = (-1.0, 1.0)
    resolution = (num_cells, num_cells, num_cells)
    domain_details = field_types.UniformDomain(
        periodicity=(True, True, True),
        resolution=resolution,
        domain_bounds=(domain_bounds, domain_bounds, domain_bounds),
    )
    list_vfields = [
        {
            "label": "divergence",
            "vfield": generate_curl_free_vfield(domain_bounds, num_cells),
        },
        {
            "label": "solenoidal",
            "vfield": generate_div_free_vfield(domain_bounds, num_cells),
        },
        {
            "label": "mixed",
            "vfield": generate_mixed_vfield(domain_bounds, num_cells),
        },
    ]
    fig, axs = plot_manager.create_figure(num_rows=3, num_cols=3, axis_shape=(7, 7))
    failed_vfields = []
    for vfield_index, vfield_entry in enumerate(list_vfields):
        vfield_name = vfield_entry["label"]
        vfield = vfield_entry["vfield"]
        print(f"input: {vfield_name} field")
        decomp = decompose_fields.compute_helmholtz_decomposition(
            vfield=vfield,
            domain_details=domain_details,
        )
        vfield_div = decomp.div_vfield
        vfield_sol = decomp.sol_vfield
        ## q - (q_div + q_sol)
        residual = field_types.VectorField(
            data=vfield.data - (vfield_div.data + vfield_sol.data),
            labels=vfield.labels,
        )
        sfield_check_q_diff = field_operators.compute_vfield_magnitude(residual)
        ## curl(q_div)
        curl_div = field_operators.compute_vfield_curl(vfield=vfield_div, domain_details=domain_details)
        sfield_check_div_is_sol_free = field_operators.compute_vfield_magnitude(curl_div)
        ## div(q_sol)
        sfield_check_sol_is_div_free = field_operators.compute_vfield_divergence(
            vfield=vfield_sol,
            domain_details=domain_details,
        )
        ## stats
        abs_q_diff = numpy.abs(sfield_check_q_diff.data)
        abs_sol_in_div = numpy.abs(sfield_check_div_is_sol_free.data)
        abs_div_in_sol = numpy.abs(sfield_check_sol_is_div_free.data)
        ave_q_diff = numpy.median(abs_q_diff)
        ave_sol_in_div = numpy.median(abs_sol_in_div)
        ave_div_in_sol = numpy.median(abs_div_in_sol)
        std_q_diff = numpy.std(abs_q_diff)
        std_sol_in_div = numpy.std(abs_sol_in_div)
        std_div_in_sol = numpy.std(abs_div_in_sol)
        ## simple thresholds (unchanged)
        bool_q_returned = ave_q_diff < 0.5
        bool_div_is_sol_free = ave_sol_in_div < 0.5
        bool_sol_is_div_free = ave_div_in_sol < 0.5
        print(f"|q - (q_div + q_sol)| median = {ave_q_diff:.2e} +/- {std_q_diff:.2e}")
        print(f"|curl(q_div)| median = {ave_sol_in_div:.2e} +/- {std_sol_in_div:.2e}")
        print(f"|div(q_sol)| median = {ave_div_in_sol:.2e} +/- {std_div_in_sol:.2e}")
        ## plots
        plot_vfield_sliceSlice(axs[vfield_index, 0], vfield, domain_bounds)
        plot_vfield_sliceSlice(axs[vfield_index, 1], vfield_div, domain_bounds)
        plot_vfield_sliceSlice(axs[vfield_index, 2], vfield_sol, domain_bounds)
        axs[vfield_index, 0].text(
            0.5,
            0.05,
            f"input: {vfield_name} field",
            va="bottom",
            ha="center",
            transform=axs[vfield_index, 0].transAxes,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )
        axs[vfield_index, 1].text(
            0.5,
            0.05,
            "measured: divergence component",
            va="bottom",
            ha="center",
            transform=axs[vfield_index, 1].transAxes,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )
        axs[vfield_index, 2].text(
            0.5,
            0.05,
            "measured: solenoidal component",
            va="bottom",
            ha="center",
            transform=axs[vfield_index, 2].transAxes,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )
        if not (bool_q_returned and bool_div_is_sol_free and bool_sol_is_div_free):
            if not bool_q_returned:
                print("Failed test. q_div + q_sol != q")
            if not bool_div_is_sol_free:
                print("Failed test. |curl(q_div)| > threshold")
            if not bool_sol_is_div_free:
                print("Failed test. |div(q_sol)| > threshold")
            failed_vfields.append(vfield_name)
        else:
            print("Test passed successfully!")
        print(" ")
    directory = io_manager.get_caller_directory()
    file_name = "helmholtz_decomposition.png"
    file_path = io_manager.combine_file_path_parts([directory, file_name])
    plot_manager.save_figure(fig, file_path)
    assert len(failed_vfields) == 0, (
        f"Test failed for the following vector field(s): "
        f"{list_utils.cast_to_string(failed_vfields)}"
    )
    print("All tests passed successfully!")


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } TEST
