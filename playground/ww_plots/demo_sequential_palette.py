## { SCRIPT

##
## === DEPENDENCIES
##

import numpy

from pathlib import Path

from jormi.ww_plots import plot_manager, add_colour
from jormi.ww_plots.colour_palette import SequentialPalette

##
## === DEMO DATA
##


def _make_gradient(vmin: float, vmax: float) -> numpy.ndarray:
    x = numpy.linspace(vmin, vmax, 200)
    y = numpy.linspace(vmin, vmax, 200)
    xx, yy = numpy.meshgrid(x, y)
    return xx + yy * 0.0  # horizontal gradient


##
## === PROGRAM MAIN
##


def main() -> None:
    vmin, vmax = 0.0, 1.0
    data = _make_gradient(vmin, vmax)

    palettes = [
        (
            "default: cmr.arctic",
            SequentialPalette(
                value_range=(vmin, vmax),
            ),
        ),
        (
            "built-in: white-brown",
            SequentialPalette(
                value_range=(vmin, vmax),
                palette_name="white-brown",
            ),
        ),
        (
            "from_colours",
            SequentialPalette.from_colours(
                value_range=(vmin, vmax),
                colours=["#1a1aff", "#ffffff", "#ff1a1a"],
                palette_name="custom-blue-white-red",
            ),
        ),
        (
            "with_palette_range (0.2, 0.8)",
            SequentialPalette(
                value_range=(vmin, vmax),
            ).with_palette_range((0.2, 0.8)),
        ),
    ]

    n_panels = len(palettes)
    fig, axs = plot_manager.create_figure(
        num_rows=n_panels,
        num_cols=1,
        axis_shape=(4, 4),
        x_spacing=0.3,
    )

    for col_idx, (title, palette) in enumerate(palettes):
        ax = axs[col_idx, 0]
        ax.imshow(
            data,
            norm=palette._mpl_norm,
            cmap=palette._mpl_colormap,
            origin="lower",
            aspect="auto",
        )
        add_colour.add_colourbar(ax, palette=palette)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    script_path = Path(__file__).parent
    plot_manager.save_figure(fig, script_path / "demo_sequential_palette.png")


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
