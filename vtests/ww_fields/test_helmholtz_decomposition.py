## { V-TEST

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path
from typing import Any, TypedDict

## third-party
import numpy
from matplotlib.axes import Axes as mpl_Axes

## local
from jormi import ww_lists
from jormi.ww_fields.fields_3d import (
    decompose_fields,
    domain_types,
    field_operators,
    field_types,
)
from jormi.ww_plots import manage_plots
from jormi.ww_types import check_types


class _VFieldEntry(TypedDict):
    label: str
    vfield: field_types.VectorField_3D


##
## === EXAMPLE VECTOR FIELDS
##


def generate_div_vfield(
    udomain_3d: domain_types.UniformDomain_3D,
) -> field_types.VectorField_3D:
    """Generate a curl-free (irrotational) vector field."""
    x0_centers, x1_centers, x2_centers = udomain_3d.cell_centers
    grid_x0, grid_x1, grid_x2 = numpy.meshgrid(
        x0_centers,
        x1_centers,
        x2_centers,
        indexing="ij",
    )
    varray = numpy.stack([2 * grid_x0, 2 * grid_x1, 2 * grid_x2])
    return field_types.VectorField_3D.from_3d_varray(
        varray_3d=varray,
        udomain_3d=udomain_3d,
        field_label=r"\vec{q}",
    )


def generate_sol_vfield(
    udomain_3d: domain_types.UniformDomain_3D,
) -> field_types.VectorField_3D:
    """Generate a solenoidal (divergence-free) vector field."""
    x0_centers, x1_centers, x2_centers = udomain_3d.cell_centers
    domain_length = udomain_3d.domain_lengths[0]
    k = 2 * numpy.pi / domain_length
    grid_x0, grid_x1, grid_x2 = numpy.meshgrid(
        x0_centers,
        x1_centers,
        x2_centers,
        indexing="ij",
    )
    vcomp_x0 = -k * grid_x0 * numpy.sin(k * grid_x0 * grid_x1)
    vcomp_x1 = k * grid_x1 * numpy.sin(k * grid_x0 * grid_x1)
    vcomp_x2 = numpy.zeros_like(grid_x2)
    varray = numpy.stack([vcomp_x0, vcomp_x1, vcomp_x2])
    return field_types.VectorField_3D.from_3d_varray(
        varray_3d=varray,
        udomain_3d=udomain_3d,
        field_label=r"\vec{q}",
    )


def generate_uniform_vfield(
    const_vector: tuple[float, float, float],
    udomain_3d: domain_types.UniformDomain_3D,
) -> field_types.VectorField_3D:
    """Generate a uniform (bulk-only) vector field with constant components."""
    check_types.ensure_sequence(
        param=const_vector,
        param_name="const_vector",
        allow_none=False,
        seq_length=3,
        valid_seq_types=check_types.RuntimeTypes.Sequences.SequenceLike,
        valid_elem_types=check_types.RuntimeTypes.Numerics.FloatLike,
    )
    resolution = udomain_3d.resolution
    varray = numpy.stack(
        [
            numpy.full(resolution, float(const_vector[0])),
            numpy.full(resolution, float(const_vector[1])),
            numpy.full(resolution, float(const_vector[2])),
        ],
    )
    return field_types.VectorField_3D.from_3d_varray(
        varray_3d=varray,
        udomain_3d=udomain_3d,
        field_label=r"\vec{q}",
    )


def generate_mixed_vfield(
    udomain_3d: domain_types.UniformDomain_3D,
    bulk_vector: tuple[float, float, float] | None = None,
) -> field_types.VectorField_3D:
    """Generate a mixed field: div + sol (+ optional uniform bulk)."""
    check_types.ensure_sequence(
        param=bulk_vector,
        param_name="bulk_vector",
        allow_none=True,
        seq_length=3,
        valid_seq_types=check_types.RuntimeTypes.Sequences.SequenceLike,
        valid_elem_types=check_types.RuntimeTypes.Numerics.FloatLike,
    )
    varray_div = field_types.extract_3d_varray(generate_div_vfield(udomain_3d))
    varray_sol = field_types.extract_3d_varray(generate_sol_vfield(udomain_3d))
    if bulk_vector is not None:
        varray_bulk = field_types.extract_3d_varray(
            vfield_3d=generate_uniform_vfield(
                const_vector=bulk_vector,
                udomain_3d=udomain_3d,
            ),
        )
        varray = varray_div + varray_sol + varray_bulk
    else:
        varray = varray_div + varray_sol
    return field_types.VectorField_3D.from_3d_varray(
        varray_3d=varray,
        udomain_3d=udomain_3d,
        field_label=r"\vec{q}",
    )


##
## === HELPER FUNCTIONS
##


def _sfield_abs_median_std(
    sfield: field_types.ScalarField_3D,
) -> tuple[float, float]:
    arr = numpy.abs(field_types.extract_3d_sarray(sfield))
    return float(numpy.median(arr)), float(numpy.std(arr))


def compute_field_fraction(
    bin_edges: numpy.ndarray[Any, numpy.dtype[Any]],
    pdf: numpy.ndarray[Any, numpy.dtype[Any]],
) -> float:
    nonzero_indices = numpy.where(pdf > 0)[0]
    if len(nonzero_indices) > 0:
        first_percent = float(bin_edges[nonzero_indices[0]])
        last_percent = float(bin_edges[nonzero_indices[-1]])
        return (first_percent if first_percent == last_percent else (last_percent - first_percent))
    return 0.0


def plot_vfield_slice(
    ax: mpl_Axes,
    vfield: field_types.VectorField_3D,
    domain_bounds: tuple[float, float],
) -> None:
    varray = field_types.extract_3d_varray(vfield)
    num_cells_x0, num_cells_x1, num_cells_x2 = varray.shape[1:]
    index_x2 = num_cells_x2 // 2  # middle slice in the z-direction
    grid_x0, grid_x1 = numpy.meshgrid(
        numpy.linspace(domain_bounds[0], domain_bounds[1], num_cells_x0),
        numpy.linspace(domain_bounds[0], domain_bounds[1], num_cells_x1),
        indexing="xy",
    )
    sfield_q_magn = field_operators.compute_vfield_magnitude(vfield)
    sfield_q_magn_array = field_types.extract_3d_sarray(sfield_q_magn)
    sfield_q_magn_slice = sfield_q_magn_array[:, :, index_x2]
    sfield_q_magn_min = float(numpy.min(sfield_q_magn_slice))
    sfield_q_magn_max = float(numpy.max(sfield_q_magn_slice))
    ax.imshow(
        sfield_q_magn_slice.T,
        origin="lower",
        extent=(domain_bounds[0], domain_bounds[1], domain_bounds[0], domain_bounds[1]),
        cmap="viridis",
        alpha=0.7,
    )
    ax.streamplot(
        grid_x0,
        grid_x1,
        varray[0, :, :, index_x2],
        varray[1, :, :, index_x2],
        color="black",
        arrowstyle="->",
        linewidth=2.0,
        density=1.0,
        arrowsize=1.0,
        broken_streamlines=False,
    )
    min_label = f"min: {sfield_q_magn_min:.2e}"
    max_label = f"max: {sfield_q_magn_max:.2e}"
    ax.text(
        0.05,
        0.05,
        f"{min_label}\n{max_label}",
        va="bottom",
        ha="left",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
    )
    ax.set_xlim((domain_bounds[0], domain_bounds[1]))
    ax.set_ylim((domain_bounds[0], domain_bounds[1]))
    ax.set_xticks([])
    ax.set_yticks([])


##
## === ORTHOGONAL DECOMPOSITION TEST
##


def main():
    num_cells = 50
    domain_bounds = (-1.0, 1.0)
    resolution = (num_cells, num_cells, num_cells)
    udomain_3d = domain_types.UniformDomain_3D(
        periodicity=(True, True, True),
        resolution=resolution,
        domain_bounds=(domain_bounds, domain_bounds, domain_bounds),
    )
    ## include a pure-bulk test: should recover bulk in bulk_vfield, and near-zero div/sol
    bulk_vector = (0.3, -0.1, 0.2)
    list_vfields: list[_VFieldEntry] = [
        {
            "label": "div. + sol. + bulk",
            "vfield": generate_mixed_vfield(
                udomain_3d=udomain_3d,
                bulk_vector=bulk_vector,
            ),
        },
        {
            "label": "purely div.",
            "vfield": generate_div_vfield(udomain_3d),
        },
        {
            "label": "purely sol.",
            "vfield": generate_sol_vfield(udomain_3d),
        },
        {
            "label": "purely bulk",
            "vfield": generate_uniform_vfield(bulk_vector, udomain_3d),
        },
    ]
    ## 4 rows (input + 3 meaured) x 4 cols (scenarios: combined, div-only, sol-only, bulk-only)
    fig, axs_grid = manage_plots.create_figure(
        num_rows=4,
        num_cols=4,
        axis_shape=(7, 8),
    )
    failed_vfields = []
    for vfield_index, vfield_entry in enumerate(list_vfields):
        vfield_name = vfield_entry["label"]
        vfield = vfield_entry["vfield"]
        print(f"input: {vfield_name} field")
        ## decompose
        dfields = decompose_fields.compute_helmholtz_decomposed_fields(
            vfield_3d_q=vfield,
        )
        vfield_3d_div = dfields.vfield_3d_div
        vfield_3d_sol = dfields.vfield_3d_sol
        vfield_3d_bulk = dfields.vfield_3d_bulk
        ## reconstructed field: q_rec = q_div + q_sol + q_bulk
        vfield_rec = field_types.VectorField_3D.from_3d_varray(
            varray_3d=(
                field_types.extract_3d_varray(vfield_3d_div) + field_types.extract_3d_varray(vfield_3d_sol) +
                field_types.extract_3d_varray(vfield_3d_bulk)
            ),
            udomain_3d=udomain_3d,
            field_label=vfield.field_label,
        )
        ## residual: q - q_rec (should be ~0)
        vfield_residual = field_types.VectorField_3D.from_3d_varray(
            varray_3d=(field_types.extract_3d_varray(vfield) - field_types.extract_3d_varray(vfield_rec)),
            udomain_3d=udomain_3d,
            field_label=vfield.field_label,
        )
        ## checks
        sfield_check_q_diff = field_operators.compute_vfield_magnitude(vfield_residual)
        curl_div = field_operators.compute_vfield_curl(vfield_3d_div)
        sfield_check_div_is_sol_free = field_operators.compute_vfield_magnitude(
            curl_div,
        )
        sfield_check_sol_is_div_free = field_operators.compute_vfield_divergence(
            vfield_3d_sol,
        )
        curl_bulk = field_operators.compute_vfield_curl(vfield_3d_bulk)
        sfield_check_bulk_div = field_operators.compute_vfield_divergence(
            vfield_3d_bulk,
        )
        sfield_check_bulk_curl = field_operators.compute_vfield_magnitude(curl_bulk)
        ## stats and thresholds (tolerant; these can be tightened)
        check_items = [
            (
                "|q - (q_div + q_sol + q_bulk)|",
                sfield_check_q_diff,
                0.5,
                "q_div + q_sol + q_bulk != q",
            ),
            (
                "|curl(q_div)|",
                sfield_check_div_is_sol_free,
                0.5,
                "|curl(q_div)| > threshold",
            ),
            (
                "|div(q_sol)|",
                sfield_check_sol_is_div_free,
                0.5,
                "|div(q_sol)| > threshold",
            ),
            (
                "|curl(q_bulk)|",
                sfield_check_bulk_curl,
                1e-12,
                "|curl(q_bulk)| not ~ 0",
            ),
            (
                "|div(q_bulk)|",
                sfield_check_bulk_div,
                1e-12,
                "|div(q_bulk)| not ~ 0",
            ),
        ]
        failed_checks = []
        for label, sfield, threshold, fail_msg in check_items:
            median, std = _sfield_abs_median_std(sfield)
            print(f"{label} median = {median:.2e} +/- {std:.2e}")
            if median >= threshold:
                failed_checks.append(fail_msg)
        ## plots: for each input field, fill a 4x4 block column-wise:
        index_col = vfield_index
        plot_vfield_slice(
            ax=axs_grid[0, index_col],
            vfield=vfield_rec,
            domain_bounds=domain_bounds,
        )
        plot_vfield_slice(
            ax=axs_grid[1, index_col],
            vfield=vfield_3d_div,
            domain_bounds=domain_bounds,
        )
        plot_vfield_slice(
            ax=axs_grid[2, index_col],
            vfield=vfield_3d_sol,
            domain_bounds=domain_bounds,
        )
        plot_vfield_slice(
            ax=axs_grid[3, index_col],
            vfield=vfield_3d_bulk,
            domain_bounds=domain_bounds,
        )
        axs_grid[0, index_col].text(
            0.5,
            0.95,
            f"input: {vfield_name}",
            va="top",
            ha="center",
            transform=axs_grid[0, index_col].transAxes,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )
        axs_grid[1, index_col].text(
            0.5,
            0.95,
            "measured: div. comp.",
            va="top",
            ha="center",
            transform=axs_grid[1, index_col].transAxes,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )
        axs_grid[2, index_col].text(
            0.5,
            0.95,
            "measured: sol. comp.",
            va="top",
            ha="center",
            transform=axs_grid[2, index_col].transAxes,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )
        axs_grid[3, index_col].text(
            0.5,
            0.95,
            "measured: bulk comp.",
            va="top",
            ha="center",
            transform=axs_grid[3, index_col].transAxes,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )
        if failed_checks:
            for check_msg in failed_checks:
                print(f"Failed test. {check_msg}")
            failed_vfields.append(vfield_name)
        else:
            print("Test passed successfully!")
        print(" ")
    file_name = "helmholtz_decomposition.png"
    file_path = Path(__file__).parent / file_name
    manage_plots.save_figure(fig, file_path)
    assert len(failed_vfields) == 0, (
        f"Test failed for the following vector field(s): "
        f"{ww_lists.as_string(failed_vfields)}"
    )
    print("All tests passed successfully!")


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } V-TEST
