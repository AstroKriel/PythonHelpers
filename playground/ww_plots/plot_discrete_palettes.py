## { SCRIPT

##
## === DEPENDENCIES
##

import numpy

from pathlib import Path

from jormi.ww_plots import plot_manager, add_color
from jormi.ww_plots.color_palette import DiscretePalette

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
    plot_manager.save_figure(fig, script_path / "discrete_palettes.png")


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
