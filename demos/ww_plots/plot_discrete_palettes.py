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
    manage_figure,
    style_plots,
)
from jormi.ww_plots.color_palettes import DiscretePalette
from jormi.ww_types import box_positions

##
## === DEMO DATA
##


def _make_gradient(
    *,
    value_min: float,
    value_max: float,
) -> NDArray[Any]:
    x_values = numpy.linspace(value_min, value_max, 200)
    y_values = numpy.linspace(value_min, value_max, 200)
    x_grid, y_grid = numpy.meshgrid(x_values, y_values)
    return x_grid + y_grid * 0.0  # horizontal gradient


##
## === PROGRAM MAIN
##


def main() -> None:
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    style_plots.set_theme()
    value_min, value_max = 0.0, 1.0
    data = _make_gradient(
        value_min=value_min,
        value_max=value_max,
    )

    palettes = [
        (
            "uniform binning",
            DiscretePalette.from_uniform_range(
                value_range=(value_min, value_max),
                num_bins=5,
                palette_name="cmr.arctic",
                palette_range=(0.0, 1.0),
            ),
        ),
        (
            "custom bin edges",
            DiscretePalette.from_name(
                bin_edges=(0.0, 0.1, 0.3, 0.6, 0.8, 1.0),
                palette_name="cmr.arctic",
                palette_range=(0.0, 1.0),
            ),
        ),
        (
            "custom hex colors",
            DiscretePalette.from_colors(
                bin_edges=(0.0, 0.25, 0.5, 0.75, 1.0),
                colors=["#264653", "#2a9d8f", "#e9c46a", "#f4a261"],
                palette_range=(0.0, 1.0),
            ),
        ),
        (
            "modified color bins",
            DiscretePalette.from_uniform_range(
                value_range=(value_min, value_max),
                num_bins=5,
                palette_name="cmr.arctic",
                palette_range=(0.0, 1.0),
            ).with_bin_edges((0.0, 0.05, 0.2, 0.5, 0.9, 1.0)),
        ),
    ]

    num_panels = len(palettes)
    figure, panels = manage_figure.create_figure(
        num_panel_rows=num_panels,
        num_panel_columns=1,
        panel_shape=manage_figure.BoxShape(
            width_cm=10.0,
            height_cm=10.0,
        ),
        x_spacing=0.3,
    )

    for panel_index, (title, palette) in enumerate(palettes):
        panel = panels[panel_index, 0]
        panel.imshow(
            data,
            norm=palette.mpl_norm,
            cmap=palette.mpl_cmap,
            origin="lower",
            aspect="auto",
        )
        add_color.add_colorbar(
            panel=panel,
            palette=palette,
            colorbar_length=0.95,
            colorbar_thickness=0.1,
        )
        annotate_panel.add_text(
            panel=panel,
            x_pos=0.5,
            y_pos=0.95,
            label=title,
            x_alignment=box_positions.Positions.Center.Center,
            y_alignment=box_positions.Positions.Side.Top,
            text_size=14,
            box_alpha=1.0,
        )
        panel.set_xticks([])
        panel.set_yticks([])

    script_path = Path(__file__).parent
    manage_figure.save_figure(
        figure=figure,
        figure_path=script_path / "discrete_palettes.png",
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
