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
    annotate_panel,
    manage_figure,
    style_plots,
)
from jormi.ww_types import box_positions

##
## === CONSTANTS
##

## a straight trend, scattered away from it in both coordinates
SEED = 42
NUM_POINTS = 40
TREND_SLOPE = 1.6
TREND_INTERCEPT = 2.0
X_SCATTER = 0.25
Y_SCATTER = 1.2

## the same density for both figures, so the half-page file is exactly half as wide in
## pixels and any difference in the text is real rather than resampling
PIXELS_PER_CM = 160.0
PANEL_ASPECT_RATIO = 1.4

##
## === DEMO DATA
##


def generate_scattered_series(
    *,
    seed: int,
    num_points: int,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Sample a straight trend, then scatter each point away from it in x and y."""
    rng = numpy.random.default_rng(seed=seed)
    x_trend_values = numpy.linspace(0.0, 10.0, num_points)
    y_trend_values = TREND_INTERCEPT + TREND_SLOPE * x_trend_values
    x_values = x_trend_values + rng.normal(0.0, X_SCATTER, num_points)
    y_values = y_trend_values + rng.normal(0.0, Y_SCATTER, num_points)
    return x_values, y_values


##
## === HELPER FUNCTIONS
##


def plot_series(
    *,
    panel: manage_figure.Panel,
    x_values: NDArray[Any],
    y_values: NDArray[Any],
) -> None:
    """Draw the series as markers, with the trend they were scattered from."""
    x_trend_values = numpy.array([x_values.min(), x_values.max()])
    panel.plot(
        x_values,
        y_values,
        marker="o",
        ls="",
        color="black",
        label="measured",
    )
    panel.plot(
        x_trend_values,
        TREND_INTERCEPT + TREND_SLOPE * x_trend_values,
        ls="--",
        color="black",
        label=rf"$y = {TREND_INTERCEPT:.1f} + {TREND_SLOPE:.1f}\,x$",
    )
    panel.set_xlabel(r"$x \; [\mathrm{arb.\,units}]$")
    panel.set_ylabel(r"$\langle y \rangle \; [\mathrm{arb.\,units}]$")
    panel.legend(loc=box_positions.MPLPositions.Anchor.Corner.TopLeft)


def report_drawn_sizes(
    *,
    label: str,
    figure: Any,
    panel: manage_figure.Panel,
) -> None:
    """Log the figure width and drawn text size, both in the units they were asked in."""
    figure_width_cm = float(figure.get_size_inches()[0]) * style_plots.CM_PER_INCH
    panel_width_cm = panel.get_position().width * figure_width_cm
    text_size_params = style_plots.get_text_size_params()
    manage_log.log_action(
        title=label,
        outcome=manage_log.ActionOutcome.SUCCESS,
        message="Text holds its size while the panel narrows.",
        notes={
            "figure width": f"{figure_width_cm:.2f} cm",
            "panel width": f"{panel_width_cm:.2f} cm",
            "axis label": f"{text_size_params.axis_label_size:.2f} pt",
            "tick label": f"{text_size_params.tick_label_size:.2f} pt",
        },
    )


##
## === PROGRAM MAIN
##


def main() -> None:
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    style_plots.set_theme()
    x_values, y_values = generate_scattered_series(
        seed=SEED,
        num_points=NUM_POINTS,
    )
    figures_dir = Path(__file__).parent
    page_layouts = {
        "full page": style_plots.FULL_PAGE_FIGURE_LAYOUT,
        "0.9 page": style_plots.FigureLayout(
            figure_width=style_plots.FigureWidth(width_fraction=0.9),
        ),
        "half page": style_plots.HALF_PAGE_FIGURE_LAYOUT,
    }
    for label, figure_layout in page_layouts.items():
        figure, panel = manage_figure.create_figure(
            figure_layout=figure_layout,
            panel_aspect_ratio=PANEL_ASPECT_RATIO,
        )
        plot_series(
            panel=panel,
            x_values=x_values,
            y_values=y_values,
        )
        annotate_panel.add_text(
            panel=panel,
            x_pos=0.95,
            y_pos=0.05,
            label=rf"{label}: $\sigma_x = {X_SCATTER:.2f},\; \sigma_y = {Y_SCATTER:.2f}$",
            x_alignment=box_positions.Positions.Side.Right,
            y_alignment=box_positions.Positions.Side.Bottom,
        )
        report_drawn_sizes(
            label=label,
            figure=figure,
            panel=panel,
        )
        manage_figure.save_figure(
            figure=figure,
            figure_path=figures_dir / f"page-width-{label.replace(' ', '-')}.png",
            pixels_per_cm=PIXELS_PER_CM,
            verbose=False,
        )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
