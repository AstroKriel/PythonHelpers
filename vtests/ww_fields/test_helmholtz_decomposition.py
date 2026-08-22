## { V-TEST

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

## third-party
import numpy

## local
from jormi import ww_lists
from jormi.ww_fields.fields_3d import (
    decompose_fields,
    domain_models,
    field_models,
    field_operators,
)
from jormi.ww_io import manage_log
from jormi.ww_plots import annotate_panel, manage_figure, style_plots
from jormi.ww_types import box_positions
from jormi.ww_validation import validate_types

##
## === TYPE ALIASES
##


@dataclass(frozen=True)
class VFieldEntry:
    label: str
    vfield_3d: field_models.VectorField_3D


@dataclass(frozen=True)
class DecomposedVFields:
    sum_vfield_3d: field_models.VectorField_3D
    div_vfield_3d: field_models.VectorField_3D
    sol_vfield_3d: field_models.VectorField_3D
    bulk_vfield_3d: field_models.VectorField_3D


##
## === EXAMPLE VECTOR FIELDS
##


def generate_div_vfield(
    uniform_domain_3d: domain_models.UniformDomain_3D,
) -> field_models.VectorField_3D:
    """Generate a curl-free (irrotational) vector field."""
    x0_centers, x1_centers, x2_centers = uniform_domain_3d.cell_centers
    grid_x0, grid_x1, grid_x2 = numpy.meshgrid(
        x0_centers,
        x1_centers,
        x2_centers,
        indexing="ij",
    )
    varray = numpy.stack([2 * grid_x0, 2 * grid_x1, 2 * grid_x2])
    return field_models.VectorField_3D.from_3d_varray(
        varray_3d=varray,
        uniform_domain_3d=uniform_domain_3d,
        field_name="purely_div",
        latex_label=r"\vec{q}_\mathrm{div}",
    )


def generate_sol_vfield(
    uniform_domain_3d: domain_models.UniformDomain_3D,
) -> field_models.VectorField_3D:
    """Generate a solenoidal (divergence-free) vector field."""
    x0_centers, x1_centers, x2_centers = uniform_domain_3d.cell_centers
    domain_length = uniform_domain_3d.domain_lengths[0]
    wavenumber = 2 * numpy.pi / domain_length
    grid_x0, grid_x1, grid_x2 = numpy.meshgrid(
        x0_centers,
        x1_centers,
        x2_centers,
        indexing="ij",
    )
    vcomp_x0 = -wavenumber * grid_x0 * numpy.sin(wavenumber * grid_x0 * grid_x1)
    vcomp_x1 = wavenumber * grid_x1 * numpy.sin(wavenumber * grid_x0 * grid_x1)
    vcomp_x2 = numpy.zeros_like(grid_x2)
    varray = numpy.stack([vcomp_x0, vcomp_x1, vcomp_x2])
    return field_models.VectorField_3D.from_3d_varray(
        varray_3d=varray,
        uniform_domain_3d=uniform_domain_3d,
        field_name="purely_sol",
        latex_label=r"\vec{q}_\mathrm{sol}",
    )


def generate_uniform_vfield(
    bulk_vector: tuple[float, float, float],
    uniform_domain_3d: domain_models.UniformDomain_3D,
) -> field_models.VectorField_3D:
    """Generate a uniform (bulk-only) vector field with constant components."""
    validate_types.ensure_sequence(
        param=bulk_vector,
        param_name="bulk_vector",
        allow_none=False,
        seq_length=3,
        valid_seq_types=validate_types.RuntimeTypes.Sequences.SequenceLike,
        valid_elem_types=validate_types.RuntimeTypes.Numerics.FloatLike,
    )
    resolution = uniform_domain_3d.resolution
    varray = numpy.stack(
        [
            numpy.full(resolution, float(bulk_vector[0])),
            numpy.full(resolution, float(bulk_vector[1])),
            numpy.full(resolution, float(bulk_vector[2])),
        ],
    )
    return field_models.VectorField_3D.from_3d_varray(
        varray_3d=varray,
        uniform_domain_3d=uniform_domain_3d,
        field_name="purely_bulk",
        latex_label=r"\vec{q}_\mathrm{bulk}",
    )


def generate_mixed_vfield(
    uniform_domain_3d: domain_models.UniformDomain_3D,
    bulk_vector: tuple[float, float, float] | None = None,
) -> field_models.VectorField_3D:
    """Generate a mixed field: div + sol (+ optional uniform bulk)."""
    validate_types.ensure_sequence(
        param=bulk_vector,
        param_name="bulk_vector",
        allow_none=True,
        seq_length=3,
        valid_seq_types=validate_types.RuntimeTypes.Sequences.SequenceLike,
        valid_elem_types=validate_types.RuntimeTypes.Numerics.FloatLike,
    )
    varray_div = field_models.extract_3d_varray(
        generate_div_vfield(
            uniform_domain_3d,
        ),
    )
    varray_sol = field_models.extract_3d_varray(
        generate_sol_vfield(
            uniform_domain_3d,
        ),
    )
    if bulk_vector is not None:
        varray_bulk = field_models.extract_3d_varray(
            vfield_3d=generate_uniform_vfield(
                bulk_vector=bulk_vector,
                uniform_domain_3d=uniform_domain_3d,
            ),
        )
        varray = varray_div + varray_sol + varray_bulk
    else:
        varray = varray_div + varray_sol
    return field_models.VectorField_3D.from_3d_varray(
        varray_3d=varray,
        uniform_domain_3d=uniform_domain_3d,
        field_name="mixed",
        latex_label=r"\vec{q}",
    )


##
## === HELPER FUNCTIONS
##


def _sfield_abs_median_std(
    sfield_3d: field_models.ScalarField_3D,
) -> tuple[float, float]:
    sarray_3d = numpy.abs(
        field_models.extract_3d_sarray(
            sfield_3d,
        ),
    )
    return float(
        numpy.median(
            sarray_3d,
        ),
    ), float(
        numpy.std(
            sarray_3d,
        ),
    )


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
    panel: manage_figure.PlotPanel,
    vfield_3d: field_models.VectorField_3D,
    domain_bounds: tuple[float, float],
) -> None:
    varray = field_models.extract_3d_varray(vfield_3d)
    num_cells_x0, num_cells_x1, num_cells_x2 = varray.shape[1:]
    index_x2 = num_cells_x2 // 2  # middle slice in the z-direction
    grid_x0, grid_x1 = numpy.meshgrid(
        numpy.linspace(domain_bounds[0], domain_bounds[1], num_cells_x0),
        numpy.linspace(domain_bounds[0], domain_bounds[1], num_cells_x1),
        indexing="xy",
    )
    sfield_q_magn_3d = field_operators.compute_vfield_magnitude(
        vfield_3d,
        field_name="q_magnitude",
        latex_label=r"|\vec{q}|",
    )
    sfield_q_magn_array = field_models.extract_3d_sarray(sfield_q_magn_3d)
    sfield_q_magn_slice = sfield_q_magn_array[:, :, index_x2]
    sfield_q_magn_min = float(
        numpy.min(
            sfield_q_magn_slice,
        ),
    )
    sfield_q_magn_max = float(
        numpy.max(
            sfield_q_magn_slice,
        ),
    )
    panel.imshow(
        sfield_q_magn_slice.T,
        origin="lower",
        extent=(domain_bounds[0], domain_bounds[1], domain_bounds[0], domain_bounds[1]),
        cmap="viridis",
        alpha=0.7,
    )
    panel.streamplot(
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
    annotate_panel.add_text(
        panel=panel,
        x_pos=0.05,
        y_pos=0.05,
        label=f"{min_label}\n{max_label}",
        x_alignment=box_positions.Positions.Side.Left,
        y_alignment=box_positions.Positions.Side.Bottom,
        box_alpha=1.0,
    )
    panel.set_xlim((domain_bounds[0], domain_bounds[1]))
    panel.set_ylim((domain_bounds[0], domain_bounds[1]))
    panel.set_xticks([])
    panel.set_yticks([])


def annotate_panel_title(
    *,
    panel: manage_figure.PlotPanel,
    text: str,
) -> None:
    annotate_panel.add_text(
        panel=panel,
        x_pos=0.5,
        y_pos=0.95,
        label=text,
        x_alignment=box_positions.Positions.Center.Center,
        y_alignment=box_positions.Positions.Side.Top,
        box_alpha=1.0,
    )


##
## === ORTHOGONAL DECOMPOSITION TEST
##


class TestHelmholtzDecomposition:
    """
    Helmholtz-decompose a set of input vector fields and check that each
    reconstructs to its input and that the div/sol/bulk components carry the
    expected curl-free, divergence-free, and constant-field properties.
    """

    def __init__(
        self,
    ):
        ## domain parameters
        self.num_cells: int = 50
        self.domain_bounds: tuple[float, float] = (-1.0, 1.0)
        ## bulk component included in the mixed and pure-bulk inputs
        self.bulk_vector: tuple[float, float, float] = (0.3, -0.1, 0.2)
        ## pass criteria: per-check thresholds on the abs-median of each diagnostic
        self.check_thresholds: dict[str, float] = {
            "|q - (q_div + q_sol + q_bulk)|": 0.5,
            "|curl(q_div)|": 0.5,
            "|div(q_sol)|": 0.5,
            "|curl(q_bulk)|": 1e-12,
            "|div(q_bulk)|": 1e-12,
        }

    def run(
        self,
    ) -> None:
        uniform_domain_3d = self._build_domain()
        input_vfields = self._build_input_vfields(uniform_domain_3d)
        ## 4 rows (input + 3 measured) x 4 cols (combined, div-only, sol-only, bulk-only)
        figure, panels_grid = manage_figure.create_figure(
            num_panel_rows=4,
            num_panel_columns=4,
            panel_shape=manage_figure.BoxShape(
                width_cm=20.0,
                height_cm=17.5,
            ),
        )
        failed_vfields: list[str] = []
        for vfield_index, vfield_entry in enumerate(input_vfields):
            vfield_name = vfield_entry.label
            vfield_3d = vfield_entry.vfield_3d
            manage_log.log_task(text=f"Input: {vfield_name} field")
            decomposed_vfields, failed_checks = self._decompose_and_check(
                vfield_3d=vfield_3d,
                uniform_domain_3d=uniform_domain_3d,
            )
            self._plot_vfield_column(
                panels_grid=panels_grid,
                index_col=vfield_index,
                vfield_name=vfield_name,
                decomposed_vfields=decomposed_vfields,
            )
            if failed_checks:
                for check_msg in failed_checks:
                    manage_log.log_outcome(
                        text=check_msg,
                        outcome=manage_log.ActionOutcome.FAILURE,
                    )
                failed_vfields.append(vfield_name)
            else:
                manage_log.log_outcome(
                    text=vfield_name,
                    outcome=manage_log.ActionOutcome.SUCCESS,
                )
            manage_log.log_empty_lines()
        ## always save even on failure, so a fail stays inspectable
        figure_path = Path(__file__).parent / "helmholtz_decomposition.png"
        manage_figure.save_figure(
            figure=figure,
            figure_path=figure_path,
        )
        assert not failed_vfields, (
            f"Test failed for the following vector field(s): "
            f"{ww_lists.as_string(elems=failed_vfields)}"
        )
        manage_log.log_action(
            title="Helmholtz decomposition",
            outcome=manage_log.ActionOutcome.SUCCESS,
            message="All checks passed.",
        )

    def _build_domain(
        self,
    ) -> domain_models.UniformDomain_3D:
        resolution = (self.num_cells, self.num_cells, self.num_cells)
        return domain_models.UniformDomain_3D(
            periodicity=(True, True, True),
            resolution=resolution,
            domain_bounds=(self.domain_bounds, self.domain_bounds, self.domain_bounds),
        )

    def _build_input_vfields(
        self,
        uniform_domain_3d: domain_models.UniformDomain_3D,
    ) -> list[VFieldEntry]:
        return [
            VFieldEntry(
                label="div. + sol. + bulk",
                vfield_3d=generate_mixed_vfield(
                    uniform_domain_3d=uniform_domain_3d,
                    bulk_vector=self.bulk_vector,
                ),
            ),
            VFieldEntry(
                label="purely div.",
                vfield_3d=generate_div_vfield(uniform_domain_3d),
            ),
            VFieldEntry(
                label="purely sol.",
                vfield_3d=generate_sol_vfield(uniform_domain_3d),
            ),
            VFieldEntry(
                label="purely bulk",
                vfield_3d=generate_uniform_vfield(
                    bulk_vector=self.bulk_vector,
                    uniform_domain_3d=uniform_domain_3d,
                ),
            ),
        ]

    def _decompose_and_check(
        self,
        *,
        vfield_3d: field_models.VectorField_3D,
        uniform_domain_3d: domain_models.UniformDomain_3D,
    ) -> tuple[DecomposedVFields, list[str]]:
        decomposed_fields = decompose_fields.compute_helmholtz_decomposed_fields(
            vfield_3d=vfield_3d,
        )
        div_vfield_3d = decomposed_fields.div_vfield_3d
        sol_vfield_3d = decomposed_fields.sol_vfield_3d
        bulk_vfield_3d = decomposed_fields.bulk_vfield_3d
        ## q_sum = q_div + q_sol + q_bulk
        sum_vfield_3d = field_models.VectorField_3D.from_3d_varray(
            varray_3d=(
                field_models.extract_3d_varray(div_vfield_3d) +
                field_models.extract_3d_varray(sol_vfield_3d) +
                field_models.extract_3d_varray(bulk_vfield_3d)
            ),
            uniform_domain_3d=uniform_domain_3d,
            field_name="q_sum",
            latex_label=r"\vec{q}_\mathrm{sum}",
        )
        ## residual: q - q_sum (should be ~0)
        residual_vfield_3d = field_models.VectorField_3D.from_3d_varray(
            varray_3d=(field_models.extract_3d_varray(vfield_3d) - field_models.extract_3d_varray(sum_vfield_3d)),
            uniform_domain_3d=uniform_domain_3d,
            field_name="q_residual",
            latex_label=r"\vec{q} - \vec{q}_\mathrm{sum}",
        )
        check_residual_sfield_3d = field_operators.compute_vfield_magnitude(
            residual_vfield_3d,
            field_name="q_residual_magnitude",
            latex_label=r"|\vec{q} - \vec{q}_\mathrm{sum}|",
        )
        curl_div_vfield_3d = field_operators.compute_vfield_curl(
            div_vfield_3d,
            field_name="curl_q_div",
            latex_label=r"\nabla\times\vec{q}_\mathrm{div}",
        )
        check_curl_div_sfield_3d = field_operators.compute_vfield_magnitude(
            curl_div_vfield_3d,
            field_name="curl_q_div_magnitude",
            latex_label=r"|\nabla\times\vec{q}_\mathrm{div}|",
        )
        check_div_sol_sfield_3d = field_operators.compute_vfield_divergence(
            sol_vfield_3d,
            field_name="div_q_sol",
            latex_label=r"\nabla\cdot\vec{q}_\mathrm{sol}",
        )
        curl_bulk_vfield_3d = field_operators.compute_vfield_curl(
            bulk_vfield_3d,
            field_name="curl_q_bulk",
            latex_label=r"\nabla\times\vec{q}_\mathrm{bulk}",
        )
        check_div_bulk_sfield_3d = field_operators.compute_vfield_divergence(
            bulk_vfield_3d,
            field_name="div_q_bulk",
            latex_label=r"\nabla\cdot\vec{q}_\mathrm{bulk}",
        )
        check_curl_bulk_sfield_3d = field_operators.compute_vfield_magnitude(
            curl_bulk_vfield_3d,
            field_name="curl_q_bulk_magnitude",
            latex_label=r"|\nabla\times\vec{q}_\mathrm{bulk}|",
        )
        check_items: list[tuple[str, field_models.ScalarField_3D]] = [
            ("|q - (q_div + q_sol + q_bulk)|", check_residual_sfield_3d),
            ("|curl(q_div)|", check_curl_div_sfield_3d),
            ("|div(q_sol)|", check_div_sol_sfield_3d),
            ("|curl(q_bulk)|", check_curl_bulk_sfield_3d),
            ("|div(q_bulk)|", check_div_bulk_sfield_3d),
        ]
        failed_checks: list[str] = []
        for check_label, check_sfield_error_3d in check_items:
            error_median, error_std = _sfield_abs_median_std(check_sfield_error_3d)
            manage_log.log_note(text=f"{check_label} median = {error_median:.2e} +/- {error_std:.2e}")
            error_threshold = self.check_thresholds[check_label]
            if error_median >= error_threshold:
                failed_checks.append(f"{check_label}: median {error_median:.2e} >= threshold {error_threshold:.2e}")
        decomposed_vfields = DecomposedVFields(
            sum_vfield_3d=sum_vfield_3d,
            div_vfield_3d=div_vfield_3d,
            sol_vfield_3d=sol_vfield_3d,
            bulk_vfield_3d=bulk_vfield_3d,
        )
        return decomposed_vfields, failed_checks

    def _plot_vfield_column(
        self,
        *,
        panels_grid: manage_figure.PlotPanelGrid,
        index_col: int,
        vfield_name: str,
        decomposed_vfields: DecomposedVFields,
    ) -> None:
        plot_vfields = [
            (decomposed_vfields.sum_vfield_3d, f"input: {vfield_name}"),
            (decomposed_vfields.div_vfield_3d, "measured: div. comp."),
            (decomposed_vfields.sol_vfield_3d, "measured: sol. comp."),
            (decomposed_vfields.bulk_vfield_3d, "measured: bulk comp."),
        ]
        for plot_index, (plot_vfield_3d, plot_annotation) in enumerate(plot_vfields):
            panel = panels_grid[plot_index, index_col]
            plot_vfield_slice(
                panel=panel,
                vfield_3d=plot_vfield_3d,
                domain_bounds=self.domain_bounds,
            )
            annotate_panel_title(
                panel=panel,
                text=plot_annotation,
            )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    style_plots.set_theme()
    test = TestHelmholtzDecomposition()
    test.run()

## } V-TEST
