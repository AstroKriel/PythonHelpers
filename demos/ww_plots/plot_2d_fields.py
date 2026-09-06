## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path
from typing import Any

## third-party
import numpy

from matplotlib.colorbar import Colorbar as mpl_Colorbar
from numpy.typing import NDArray

## local
from jormi.ww_io import manage_log
from jormi.ww_plots import (
    add_color,
    annotate_panel,
    manage_figure,
    plot_data,
)
from jormi.ww_types import box_positions

##
## === CONSTANTS
##

NUM_CELLS = 192
PIXELS_PER_CM = 160.0
DOMAIN_RANGE = (-1.0, 1.0)
AXIS_RANGES = (DOMAIN_RANGE, DOMAIN_RANGE)
## the panels sit side by side, so a tick label on the domain edge would collide with its
## neighbour's; nothing sits above or below, so the y axis keeps its endpoints
X_TICKS = (
    -0.5,
    0.0,
    0.5,
)
Y_TICKS = (
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
)

## one gap for the whole figure: between neighbouring panels, and between the panels and
## the colorbar above them
PANEL_GAP_PT = 4.0

## the vorticity of the vortex below peaks at 2 pi, so pinning the range here lets the
## panels and the shared colorbar build the same palette from the same numbers
VORTICITY_RANGE = (-2.0 * numpy.pi, 2.0 * numpy.pi)
PALETTE_CONFIG = add_color.DivergingPaletteConfig(mid_value=0.0)

OVERLAY_COLOR = "black"

##
## === DEMO DATA
##


def generate_vortex() -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """
    A Taylor-Green vortex, returned as its two velocity components and its vorticity.

    Every array is indexed [rows, cols], which is what the plot functions call "ij".
    """
    axis_values = numpy.linspace(DOMAIN_RANGE[0], DOMAIN_RANGE[1], NUM_CELLS)
    grid_x, grid_y = numpy.meshgrid(axis_values, axis_values, indexing="xy")
    velocity_x = numpy.sin(numpy.pi * grid_x) * numpy.cos(numpy.pi * grid_y)
    velocity_y = -numpy.cos(numpy.pi * grid_x) * numpy.sin(numpy.pi * grid_y)
    vorticity = 2.0 * numpy.pi * numpy.sin(numpy.pi * grid_x) * numpy.sin(numpy.pi * grid_y)
    return velocity_x, velocity_y, vorticity


##
## === HELPER FUNCTIONS
##


def draw_backdrop(
    *,
    panel: manage_figure.Panel,
    vorticity: NDArray[Any],
) -> None:
    """Draw the vorticity every panel shares, leaving its colorbar to the caller."""
    plot_data.plot_2d_array(
        panel=panel,
        array_2d=vorticity,
        data_format="ij",
        axis_ranges=AXIS_RANGES,
        colorbar_range=VORTICITY_RANGE,
        palette_config=PALETTE_CONFIG,
        add_colorbar=False,
    )
    panel.set_xticks(X_TICKS)
    panel.set_yticks(Y_TICKS)
    panel.set_xlabel(r"$x$")


def add_shared_colorbar(
    *,
    panel_row: manage_figure.PanelGrid,
    label: str,
) -> mpl_Colorbar:
    """Put one colorbar above the row, spanning it."""
    return add_color.add_colorbar(
        panels=panel_row,
        palette=add_color.make_palette(
            config=PALETTE_CONFIG,
            value_range=VORTICITY_RANGE,
        ),
        label=label,
        colorbar_side=box_positions.Positions.Side.Top,
        ## this bar spans three panels, so keeping the style's proportions would make it as
        ## thick as it is long here; it is set slimmer against its own length instead
        colorbar_aspect_ratio=36.0,
    )


##
## === PROGRAM MAIN
##


def main() -> None:
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    velocity_x, velocity_y, vorticity = generate_vortex()
    ## the top margin holds the shared colorbar, its tick labels and its label; the gaps
    ## given here also place the colorbar, since it neighbours the panels
    figure, panel_grid = manage_figure.create_figure(
        num_panel_rows=1,
        num_panel_cols=3,
        ## square, matching the square domain; the figure is fitted around it, so the room
        ## its labels and its colorbar need is measured rather than named here
        panel_aspect_ratio=1.0,
        panel_row_gap_pt=PANEL_GAP_PT,
        panel_col_gap_pt=PANEL_GAP_PT,
    )
    ## a grid keeps its row axis unless it is 1x1, so a single row arrives shaped (1, 3)
    panel_row = panel_grid[0]
    for panel in panel_row:
        draw_backdrop(
            panel=panel,
            vorticity=vorticity,
        )
    panel_row[0].set_ylabel(r"$y$")
    plot_data.plot_2d_quiver(
        panel=panel_row[0],
        array_2d_rows=velocity_y,
        array_2d_cols=velocity_x,
        axis_ranges=AXIS_RANGES,
        num_quivers=14,
        color=OVERLAY_COLOR,
    )
    plot_data.plot_2d_streamlines(
        panel=panel_row[1],
        array_2d_rows=velocity_y,
        array_2d_cols=velocity_x,
        axis_ranges=AXIS_RANGES,
        streamline_width_pt=0.6,
        streamline_density=1.1,
        color=OVERLAY_COLOR,
    )
    plot_data.plot_2d_contours(
        panel=panel_row[2],
        array_2d=vorticity,
        data_format="ij",
        axis_ranges=AXIS_RANGES,
        levels=9,
        color=OVERLAY_COLOR,
    )
    panel_labels = (
        "quiver",
        "streamlines",
        "contours",
    )
    for panel_index, (panel, panel_label) in enumerate(zip(
            panel_row,
            panel_labels,
    ), ):
        annotate_panel.add_text(
            panel=panel,
            x_pos_fraction=0.5,
            y_pos_fraction=0.05,
            label=rf"\texttt{{{panel_label}}}",
            x_alignment=box_positions.Positions.Center.Center,
            y_alignment=box_positions.Positions.Side.Bottom,
            box_alpha=1.0,
        )
        ## only the leftmost panel carries y tick labels; the three share a y axis
        if panel_index > 0:
            panel.set_yticklabels([])
    add_shared_colorbar(
        panel_row=panel_row,
        label=r"$\omega_z$",
    )
    manage_figure.save_figure(
        figure=figure,
        figure_path=Path(__file__).parent / "2d-fields.png",
        pixels_per_cm=PIXELS_PER_CM,
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
