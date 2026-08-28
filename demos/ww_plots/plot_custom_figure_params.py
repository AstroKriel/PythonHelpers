## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## third-party
import numpy

## local
from jormi.ww_io import manage_log
from jormi.ww_plots import manage_figure, style_figure

##
## === CONSTANTS
##

LIGHT_PANEL_ASPECT_RATIO = 1.3
DARK_PANEL_ASPECT_RATIO = 1.6

FIGURE_DIR = Path(__file__).parent
LIGHT_FIGURE_PATH = FIGURE_DIR / "custom-figure-params-light.png"
DARK_FIGURE_PATH = FIGURE_DIR / "custom-figure-params-dark.png"

LIGHT_FIGURE_PARAMS = style_figure.FigureParams(
    theme=style_figure.Theme.LIGHT,
    latex_params=style_figure.LatexParams(),
    text_size_params=style_figure.TextSizeParams(largest_size_pt=9.0),
    artist_params=style_figure.ArtistParams(),
    frame_params=style_figure.FrameParams(),
    legend_params=style_figure.LegendParams(),
    save_params=style_figure.SaveParams(),
    figure_layout=style_figure.FigureLayout(
        figure_width=style_figure.FigureWidth(width_fraction=0.75),
    ),
    colorbar_layout=style_figure.ColorbarLayout(),
)

DARK_FIGURE_PARAMS = style_figure.FigureParams(
    theme=style_figure.Theme.DARK,
    latex_params=style_figure.LatexParams(),
    text_size_params=style_figure.TextSizeParams(largest_size_pt=16.0),
    artist_params=style_figure.ArtistParams(line_width_pt=1.8),
    frame_params=style_figure.FrameParams(),
    legend_params=style_figure.LegendParams(),
    save_params=style_figure.SaveParams(),
    figure_layout=style_figure.FigureLayout(
        figure_width=style_figure.FigureWidth(width_fraction=1.0),
    ),
    colorbar_layout=style_figure.ColorbarLayout(),
)

##
## === HELPER FUNCTIONS
##


def plot_demo_curve(
    *,
    panel: manage_figure.Panel,
) -> None:
    x_values = numpy.linspace(0.0, 4.0 * numpy.pi, 200)
    y_values = numpy.sin(x_values) * numpy.exp(-0.1 * x_values)
    panel.plot(x_values, y_values)
    panel.set_xlabel(r"$x$")
    panel.set_ylabel(r"$y$")


##
## === PROGRAM MAIN
##


def main() -> None:
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)

    light_figure, light_panel = manage_figure.create_figure(
        panel_aspect_ratio=LIGHT_PANEL_ASPECT_RATIO,
        figure_params=LIGHT_FIGURE_PARAMS,
    )
    plot_demo_curve(panel=light_panel)
    manage_figure.save_figure(
        figure=light_figure,
        figure_path=LIGHT_FIGURE_PATH,
    )

    ## without `figure_params` here, `create_figure` falls back to whatever style is
    ## still active, silently rendering this figure in the light style set above
    dark_figure, dark_panel = manage_figure.create_figure(
        panel_aspect_ratio=DARK_PANEL_ASPECT_RATIO,
        figure_params=DARK_FIGURE_PARAMS,
    )
    plot_demo_curve(panel=dark_panel)
    manage_figure.save_figure(
        figure=dark_figure,
        figure_path=DARK_FIGURE_PATH,
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
