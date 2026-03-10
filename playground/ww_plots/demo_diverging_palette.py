## { SCRIPT

##
## === DEPENDENCIES
##

import numpy

from pathlib import Path

from jormi.ww_plots import plot_manager, add_color
from jormi.ww_plots.color_palette import DivergingPalette

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
    vmin, vmax, vmid = -1.0, 1.0, 0.0
    data = _make_gradient(vmin, vmax)

    palettes = [
        (
            "from_name: blue-red",
            DivergingPalette.from_name(
                value_range=(vmin, vmax),
                mid_value=vmid,
            ),
        ),
        (
            "from_name: purple-green",
            DivergingPalette.from_name(
                value_range=(vmin, vmax),
                mid_value=vmid,
                palette_name="purple-green",
            ),
        ),
        (
            "from_colors",
            DivergingPalette.from_colors(
                value_range=(vmin, vmax),
                mid_value=vmid,
                colors=["#0a3d62", "#d6eaf8", "#f9f9f9", "#fadbd8", "#922b21"],
            ),
        ),
        (
            "with_mid_value (-0.3)",
            DivergingPalette.from_name(
                value_range=(vmin, vmax),
                mid_value=vmid,
            ).with_mid_value(-0.3),
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
        add_color.add_colorbar(ax, palette=palette)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    script_path = Path(__file__).parent
    plot_manager.save_figure(fig, script_path / "demo_diverging_palette.png")


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
