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


def _make_gradient(value_min: float, value_max: float) -> numpy.ndarray:
    x_values = numpy.linspace(value_min, value_max, 200)
    y_values = numpy.linspace(value_min, value_max, 200)
    x_grid, y_grid = numpy.meshgrid(x_values, y_values)
    return x_grid + y_grid * 0.0  # horizontal gradient


##
## === PROGRAM MAIN
##


def main() -> None:
    value_min, value_max, value_mid = -1.0, 1.0, 0.0
    data = _make_gradient(value_min, value_max)

    palettes = [
        (
            "named colormap",
            DivergingPalette.from_name(
                value_range=(value_min, value_max),
                mid_value=value_mid,
                palette_name="blue-white-red",
                palette_range=(0.0, 1.0),
            ),
        ),
        (
            "custom hex colors",
            DivergingPalette.from_colors(
                value_range=(value_min, value_max),
                mid_value=value_mid,
                colors=["#0a3d62", "#d6eaf8", "#f9f9f9", "#fadbd8", "#922b21"],
                palette_range=(0.0, 1.0),
            ),
        ),
        (
            "shifted midpoint value",
            DivergingPalette.from_name(
                value_range=(value_min, value_max),
                mid_value=value_mid,
                palette_name="blue-white-red",
                palette_range=(0.0, 1.0),
            ).with_mid_value(-0.3),
        ),
        (
            "clipped value range",
            DivergingPalette.from_name(
                value_range=(value_min, value_max),
                mid_value=value_mid,
                palette_name="blue-white-red",
                palette_range=(0.0, 1.0),
            ).with_value_range((-0.5, 1.0)),
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
        add_color.add_colorbar(
            ax=ax,
            palette=palette,
            cbar_length=0.95,
            cbar_thickness=0.1,
        )
        ax.text(
            0.5,
            0.95,
            title,
            fontsize=14,
            va="top",
            ha="center",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )
        ax.set_xticks([])
        ax.set_yticks([])

    script_path = Path(__file__).parent
    plot_manager.save_figure(fig, script_path / "diverging_palettes.png")


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
