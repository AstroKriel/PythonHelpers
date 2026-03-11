## { SCRIPT

##
## === DEPENDENCIES
##

import numpy

from pathlib import Path

from jormi.ww_plots import plot_manager, add_color
from jormi.ww_plots.color_palette import SequentialPalette

##
## === DEMO DATA
##


def _make_gradient(value_min: float, value_max: float) -> numpy.ndarray:
    x_values = numpy.linspace(value_min, value_max, 200)
    y_values = numpy.linspace(value_min, value_max, 200)
    x_grid, y_grid = numpy.meshgrid(x_values, y_values)
    return x_grid + y_grid * 0.0  # horizontal gradient


##
## === PROGRAM MAIN
##


def main() -> None:
    value_min, value_max = 0.0, 1.0
    data = _make_gradient(value_min, value_max)

    palettes = [
        (
            "named colormap",
            SequentialPalette.from_name(
                value_range=(value_min, value_max),
                palette_name="cmr.arctic",
                palette_range=(0.0, 1.0),
            ),
        ),
        (
            "built from custom hex colors",
            SequentialPalette.from_colors(
                value_range=(value_min, value_max),
                colors=["#1a1aff", "#ffffff", "#ff1a1a"],
                palette_range=(0.0, 1.0),
            ),
        ),
        (
            "clipped color range: (0.2, 0.8)",
            SequentialPalette.from_name(
                value_range=(value_min, value_max),
                palette_name="cmr.arctic",
                palette_range=(0.0, 1.0),
            ).with_palette_range((0.2, 0.8)),
        ),
        (
            "clipped value range: (0.3, 0.7)",
            SequentialPalette.from_name(
                value_range=(value_min, value_max),
                palette_name="cmr.arctic",
                palette_range=(0.0, 1.0),
            ).with_value_range((0.3, 0.7)),
        ),
    ]

    num_panels = len(palettes)
    fig, axs = plot_manager.create_figure(
        num_rows=num_panels,
        num_cols=1,
        axis_shape=(4, 4),
        x_spacing=0.3,
    )

    for col_idx, (title, palette) in enumerate(palettes):
        ax = axs[col_idx, 0]
        ax.imshow(
            data,
            norm=palette.mpl_norm,
            cmap=palette.mpl_cmap,
            origin="lower",
            aspect="auto",
        )
        add_color.add_colorbar(ax, palette=palette)
        ax.text(
            0.5,
            0.95,
            title,
            fontsize=12,
            va="top",
            ha="center",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )
        ax.set_xticks([])
        ax.set_yticks([])

    script_path = Path(__file__).parent
    plot_manager.save_figure(fig, script_path / "sequential_palettes.png")


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
