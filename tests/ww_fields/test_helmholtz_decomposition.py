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


def generate_div_vfield(domain_bounds, num_cells):
    """Generate a curl-free (irrotational) vector field."""
    domain = numpy.linspace(domain_bounds[0], domain_bounds[1], int(num_cells))
    grid_x, grid_y, grid_z = numpy.meshgrid(domain, domain, domain, indexing="ij")
    sfield_qx = 2 * grid_x
    sfield_qy = 2 * grid_y
    sfield_qz = 2 * grid_z
    return field_types.VectorField(
        data=numpy.stack([sfield_qx, sfield_qy, sfield_qz]),
        field_label=r"$\vec{q}$",
    )


def generate_sol_vfield(domain_bounds, num_cells):
    """Generate a solenoidal (divergence-free) vector field."""
    domain_length = domain_bounds[1] - domain_bounds[0]
    domain = numpy.linspace(domain_bounds[0], domain_bounds[1], int(num_cells))
    k = 2 * numpy.pi / domain_length
    grid_x, grid_y, grid_z = numpy.meshgrid(domain, domain, domain, indexing="ij")
    sfield_qx = -k * grid_x * numpy.sin(k * grid_x * grid_y)
    sfield_qy = k * grid_y * numpy.sin(k * grid_x * grid_y)
    sfield_qz = numpy.zeros_like(grid_z)
    return field_types.VectorField(
        data=numpy.stack([sfield_qx, sfield_qy, sfield_qz]),
        field_label=r"$\vec{q}$",
    )


def generate_uniform_vfield(const_vector, num_cells):
    """Generate a uniform (bulk-only) vector field with constant components."""
    sfield_qx = numpy.full((num_cells, num_cells, num_cells), float(const_vector[0]))
    sfield_qy = numpy.full((num_cells, num_cells, num_cells), float(const_vector[1]))
    sfield_qz = numpy.full((num_cells, num_cells, num_cells), float(const_vector[2]))
    return field_types.VectorField(
        data=numpy.stack([sfield_qx, sfield_qy, sfield_qz]),
        field_label=r"$\vec{q}$",
    )


def generate_mixed_vfield(domain_bounds, num_cells, bulk_vector=(0.0, 0.0, 0.0)):
    """Generate a mixed field: div + sol (+ optional uniform bulk)."""
    v_div = generate_div_vfield(domain_bounds, num_cells)
    v_sol = generate_sol_vfield(domain_bounds, num_cells)
    if any(float(b) != 0.0 for b in bulk_vector):
        v_bulk = generate_uniform_vfield(bulk_vector, num_cells)
        data = v_div.data + v_sol.data + v_bulk.data
    else:
        data = v_div.data + v_sol.data
    return field_types.VectorField(
        data=data,
        field_label=r"$\vec{q}$",
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
    uniform_domain = field_types.UniformDomain(
        periodicity=(True, True, True),
        resolution=resolution,
        domain_bounds=(domain_bounds, domain_bounds, domain_bounds),
    )
    ## include a pure-bulk test: should recover bulk in bulk_vfield, and near-zero div/sol
    bulk_vector = (0.3, -0.1, 0.2)
    list_vfields = [
        {
            "label": "mixed",
            "vfield": generate_mixed_vfield(domain_bounds, num_cells, bulk_vector=bulk_vector),
        },
        {
            "label": "divergence",
            "vfield": generate_div_vfield(domain_bounds, num_cells),
        },
        {
            "label": "solenoidal",
            "vfield": generate_sol_vfield(domain_bounds, num_cells),
        },
        {
            "label": "bulk",
            "vfield": generate_uniform_vfield(bulk_vector, num_cells),
        },
    ]
    ## 4 rows (input + 3 decomps/combined) x 4 cols (combined, div-only, sol-only, bulk-only)
    fig, axs_grid = plot_manager.create_figure(
        num_rows=4,
        num_cols=4,
        axis_shape=(7, 7),
    )
    failed_vfields = []
    for vfield_index, vfield_entry in enumerate(list_vfields):
        vfield_name = vfield_entry["label"]
        vfield = vfield_entry["vfield"]
        print(f"input: {vfield_name} field")
        ## decompose
        decomp = decompose_fields.compute_helmholtz_decomposition(
            vfield=vfield,
            uniform_domain=uniform_domain,
        )
        vfield_div = decomp.div_vfield
        vfield_sol = decomp.sol_vfield
        vfield_bulk = decomp.bulk_vfield
        ## reconstructed field: q_rec = q_div + q_sol + q_bulk
        vfield_rec = field_types.VectorField(
            data=vfield_div.data + vfield_sol.data + vfield_bulk.data,
            field_label=vfield.field_label,
        )
        ## residual: q - q_rec (should be ~0)
        residual = field_types.VectorField(
            data=vfield.data - vfield_rec.data,
            field_label=vfield.field_label,
        )
        ## checks
        sfield_check_q_diff = field_operators.compute_vfield_magnitude(residual)
        curl_div = field_operators.compute_vfield_curl(
            vfield=vfield_div,
            uniform_domain=uniform_domain,
        )
        sfield_check_div_is_sol_free = field_operators.compute_vfield_magnitude(
            vfield=curl_div,
        )
        sfield_check_sol_is_div_free = field_operators.compute_vfield_divergence(
            vfield=vfield_sol,
            uniform_domain=uniform_domain,
        )
        curl_bulk = field_operators.compute_vfield_curl(vfield=vfield_bulk, uniform_domain=uniform_domain)
        div_bulk = field_operators.compute_vfield_divergence(
            vfield=vfield_bulk, uniform_domain=uniform_domain
        )
        sfield_check_bulk_curl = field_operators.compute_vfield_magnitude(curl_bulk)
        sfield_check_bulk_div = div_bulk  ## already scalar field
        ## stats
        abs_q_diff = numpy.abs(sfield_check_q_diff.data)
        abs_sol_in_div = numpy.abs(sfield_check_div_is_sol_free.data)
        abs_div_in_sol = numpy.abs(sfield_check_sol_is_div_free.data)
        abs_curl_bulk = numpy.abs(sfield_check_bulk_curl.data)
        abs_div_bulk = numpy.abs(sfield_check_bulk_div.data)
        ave_q_diff = numpy.median(abs_q_diff)
        ave_sol_in_div = numpy.median(abs_sol_in_div)
        ave_div_in_sol = numpy.median(abs_div_in_sol)
        ave_curl_bulk = numpy.median(abs_curl_bulk)
        ave_div_bulk = numpy.median(abs_div_bulk)
        std_q_diff = numpy.std(abs_q_diff)
        std_sol_in_div = numpy.std(abs_sol_in_div)
        std_div_in_sol = numpy.std(abs_div_in_sol)
        std_curl_bulk = numpy.std(abs_curl_bulk)
        std_div_bulk = numpy.std(abs_div_bulk)
        ## simple thresholds (tolerant; these can be tightened)
        bool_q_returned = ave_q_diff < 0.5
        bool_div_is_sol_free = ave_sol_in_div < 0.5
        bool_sol_is_div_free = ave_div_in_sol < 0.5
        bool_bulk_uniform_curl = ave_curl_bulk < 1e-12
        bool_bulk_uniform_div = ave_div_bulk < 1e-12
        print(f"|q - (q_div + q_sol + q_bulk)| median = {ave_q_diff:.2e} +/- {std_q_diff:.2e}")
        print(f"|curl(q_div)| median = {ave_sol_in_div:.2e} +/- {std_sol_in_div:.2e}")
        print(f"|div(q_sol)| median = {ave_div_in_sol:.2e} +/- {std_div_in_sol:.2e}")
        print(f"|curl(q_bulk)| median = {ave_curl_bulk:.2e} +/- {std_curl_bulk:.2e}")
        print(f"|div(q_bulk)| median = {ave_div_bulk:.2e} +/- {std_div_bulk:.2e}")
        ## plots: for each input field, fill a 4x4 block column-wise:
        col = vfield_index  ## one column per input case
        plot_vfield_sliceSlice(axs_grid[0, col], vfield_rec, domain_bounds)
        plot_vfield_sliceSlice(axs_grid[1, col], vfield_div, domain_bounds)
        plot_vfield_sliceSlice(axs_grid[2, col], vfield_sol, domain_bounds)
        plot_vfield_sliceSlice(axs_grid[3, col], vfield_bulk, domain_bounds)
        axs_grid[0, col].text(
            0.5,
            0.05,
            f"reconstructed: {vfield_name}",
            va="bottom",
            ha="center",
            transform=axs_grid[0, col].transAxes,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )
        axs_grid[1, col].text(
            0.5,
            0.05,
            "measured: divergence component",
            va="bottom",
            ha="center",
            transform=axs_grid[1, col].transAxes,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )
        axs_grid[2, col].text(
            0.5,
            0.05,
            "measured: solenoidal component",
            va="bottom",
            ha="center",
            transform=axs_grid[2, col].transAxes,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )
        axs_grid[3, col].text(
            0.5,
            0.05,
            "measured: bulk component",
            va="bottom",
            ha="center",
            transform=axs_grid[3, col].transAxes,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )
        if not (bool_q_returned and bool_div_is_sol_free and bool_sol_is_div_free and bool_bulk_uniform_curl
                and bool_bulk_uniform_div):
            if not bool_q_returned: print("Failed test. q_div + q_sol + q_bulk != q")
            if not bool_div_is_sol_free: print("Failed test. |curl(q_div)| > threshold")
            if not bool_sol_is_div_free: print("Failed test. |div(q_sol)| > threshold")
            if not bool_bulk_uniform_curl: print("Failed test. |curl(q_bulk)| not ~ 0")
            if not bool_bulk_uniform_div: print("Failed test. |div(q_bulk)| not ~ 0")
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
