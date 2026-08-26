## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path
from typing import Any

## third-party
import numpy

from numpy.typing import NDArray

## local
from jormi.ww_io import manage_log
from jormi.ww_plots import (
    add_color,
    annotate_panel,
    color_palettes,
    manage_figure,
    style_figure,
)
from jormi.ww_types import box_positions

##
## === CONSTANTS
##

NUM_CELLS = 128
PIXELS_PER_CM = 160.0
## the width over height of the panel as it is drawn
PANEL_ASPECT = 2.284
VALUE_RANGE = (-1.3, 1.3)
DIVERGING_MID_VALUE = 0.0
DISCRETE_BIN_EDGES = (-1.3, -0.6, -0.2, 0.2, 0.6, 1.3)

##
## === DEMO DATA
##


def generate_gradient() -> NDArray[Any]:
    """A plane sloping across the panel, so every colour in a palette is shown."""
    axis_values = numpy.linspace(-1.0, 1.0, NUM_CELLS)
    x_grid, y_grid = numpy.meshgrid(axis_values, axis_values, indexing="xy")
    return x_grid + 0.3 * y_grid


##
## === HELPER FUNCTIONS
##


def build_palettes() -> dict[str, tuple[color_palettes.ColorPalette, color_palettes.ColorPalette]]:
    """
    Each kind of palette, built both ways: from a registered name, and from your colours.

    Names come from two places: `cmr.arctic` is cmasher's, while `pink-white-green` is
    one of the palettes jormi registers itself.
    """
    return {
        "sequential": (
            color_palettes.SequentialPalette.from_name(
                value_range=VALUE_RANGE,
                palette_name="cmr.arctic",
            ),
            color_palettes.SequentialPalette.from_colors(
                value_range=VALUE_RANGE,
                colors=["#0b132b", "#3a506b", "#5bc0be", "#f2f7f5"],
            ),
        ),
        "diverging": (
            color_palettes.DivergingPalette.from_name(
                value_range=VALUE_RANGE,
                mid_value=DIVERGING_MID_VALUE,
                palette_name="pink-white-green",
            ),
            color_palettes.DivergingPalette.from_colors(
                value_range=VALUE_RANGE,
                mid_value=DIVERGING_MID_VALUE,
                colors=["#5b2c6f", "#c39bd3", "#ffffff", "#f5b041", "#7e5109"],
            ),
        ),
        "discrete": (
            color_palettes.DiscretePalette.from_name(
                bin_edges=DISCRETE_BIN_EDGES,
                palette_name="cmr.arctic",
            ),
            color_palettes.DiscretePalette.from_colors(
                bin_edges=DISCRETE_BIN_EDGES,
                colors=["#0b132b", "#3a506b", "#5bc0be", "#f2f7f5", "#f2b880"],
            ),
        ),
    }


def draw_palette(
    *,
    panel: manage_figure.Panel,
    array_2d: NDArray[Any],
    palette: color_palettes.ColorPalette,
    label: str,
    colorbar_side: box_positions.Positions.PositionLike,
) -> None:
    """Show the gradient through `palette`, with its colorbar on the figure's outer edge."""
    panel.imshow(
        array_2d,
        norm=palette.mpl_norm,
        cmap=palette.mpl_cmap,
        origin="lower",
        aspect="auto",
    )
    add_color.add_colorbar(
        panels=panel,
        palette=palette,
        colorbar_side=colorbar_side,
        ## these bars are short, running the height of one small panel, so the style's
        ## proportions would draw them too fine to read the palette off
        colorbar_aspect_ratio=10.0,
    )
    annotate_panel.add_text(
        panel=panel,
        x_pos=0.5,
        y_pos=0.92,
        label=label,
        x_alignment=box_positions.Positions.Center.Center,
        y_alignment=box_positions.Positions.Side.Top,
        box_alpha=1.0,
    )
    panel.set_xticks([])
    panel.set_yticks([])


##
## === PROGRAM MAIN
##


def main() -> None:
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    style_figure.set_figure_params()
    array_2d = generate_gradient()
    palettes = build_palettes()
    ## each column carries a colorbar on the figure's outer edge; the figure is fitted, so
    ## the room those bars and their tick labels need is measured rather than named here
    figure, panel_grid = manage_figure.create_figure(
        num_panel_rows=len(palettes),
        num_panel_columns=2,
        panel_aspect=PANEL_ASPECT,
        panel_column_gap=16.0,
        panel_row_gap=14.0,
    )
    column_sides = (
        box_positions.Positions.Side.Left,
        box_positions.Positions.Side.Right,
    )
    for row_index, (kind, (named_palette, custom_palette)) in enumerate(palettes.items()):
        for column_index, palette in enumerate((named_palette, custom_palette)):
            route = "from_name" if column_index == 0 else "from_colors"
            draw_palette(
                panel=panel_grid[row_index, column_index],
                array_2d=array_2d,
                palette=palette,
                label=rf"{kind}: \texttt{{{route}}}",
                colorbar_side=column_sides[column_index],
            )
    manage_figure.save_figure(
        figure=figure,
        figure_path=Path(__file__).parent / "colour-palettes.png",
        pixels_per_cm=PIXELS_PER_CM,
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
