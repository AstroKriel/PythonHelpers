## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import dataclasses

from dataclasses import dataclass
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
    style_figure,
)
from jormi.ww_types import box_positions

##
## === CONSTANTS
##

SEED = 42
NUM_POINTS = 40
X_RANGE = (-5.0, 5.0)
X_SCATTER = 0.25

## the same density for both figures, so the half-page file is exactly half as wide in
## pixels and any difference in the text is real rather than resampling
PIXELS_PER_CM = 160.0
## the width over height of the panel as it is drawn
PANEL_ASPECT = 1.426

DATA_COLORS = (
    "royalblue",
    "orangered",
    "forestgreen",
)
## the half page carries a single series, so it has no colours to tell apart
SINGLE_SERIES_COLOR = "black"

##
## === DEMO DATA
##


@dataclass
class ScatteredSeries:
    """One series, and the two numbers it was drawn from."""

    color: str
    slope: float
    sigma: float
    x_values: NDArray[Any]
    y_values: NDArray[Any]


def generate_series(
    *,
    seed: int,
    slope: float,
    sigma: float,
    color: str,
) -> ScatteredSeries:
    """Sample `y = m x`, scattering each point away from it in both coordinates."""
    rng = numpy.random.default_rng(seed=seed)
    x_trend_values = numpy.linspace(X_RANGE[0], X_RANGE[1], NUM_POINTS)
    x_values = x_trend_values + rng.normal(0.0, X_SCATTER, NUM_POINTS)
    y_values = slope * x_values + rng.normal(0.0, sigma, NUM_POINTS)
    return ScatteredSeries(
        color=color,
        slope=slope,
        sigma=sigma,
        x_values=x_values,
        y_values=y_values,
    )


def generate_all_series() -> list[ScatteredSeries]:
    """
    Three series that differ in both slope and how tightly they hold to it.

    Every slope is positive, so the data fans through the lower-left and upper-right and
    leaves the other two corners clear for the legends and the inset.
    """
    settings = (
        (1.6, 1.2),
        (1.0, 0.6),
        (0.45, 0.3),
    )
    return [
        generate_series(
            seed=SEED + index,
            slope=slope,
            sigma=sigma,
            color=DATA_COLORS[index],
        )
        for index, (slope, sigma) in enumerate(settings)
    ]


##
## === HELPER FUNCTIONS
##


def plot_series(
    *,
    panel: manage_figure.Panel,
    series: ScatteredSeries,
) -> None:
    """Draw one series as markers, with the exact trend it was scattered from."""
    panel.plot(
        series.x_values,
        series.y_values,
        marker="o",
        ls="",
        color=series.color,
    )
    x_trend_values = numpy.array(X_RANGE)
    panel.plot(
        x_trend_values,
        series.slope * x_trend_values,
        ls="--",
        color=series.color,
    )


def label_axes(
    *,
    panel: manage_figure.Panel,
) -> None:
    panel.set_xlabel(r"$x$")
    panel.set_ylabel(r"$y$")


def format_trend(
    *,
    series: ScatteredSeries | None = None,
    include_scatter: bool = False,
) -> str:
    """
    The trend a series is drawn from, named in symbols or in its own numbers.

    `include_scatter` is what separates the two entries a legend carries: the markers
    scatter about the trend, while the line drawn through them does not.
    """
    if series is None:
        trend = r"m\,x"
        scatter = r"\sigma"
    else:
        trend = rf"{series.slope:.2f}\,x"
        scatter = f"{series.sigma:.2f}"
    if include_scatter:
        return rf"$y = {trend} + {scatter}$"
    return rf"$y = {trend}$"


def add_shape_legend(
    *,
    panel: manage_figure.Panel,
) -> None:
    """Key what the shapes mean, in one colour, since the colours are keyed separately."""
    annotate_panel.add_custom_legend(
        panel=panel,
        artists=[
            "o",
            "--",
        ],
        labels=[
            format_trend(include_scatter=True),
            format_trend(),
        ],
        colors=[
            "black",
            "black",
        ],
        anchor_point=(0.975, 0.2),
        anchor_at_corner=box_positions.Positions.Corner.BottomRight,
    )


def add_trend_legend(
    *,
    panel: manage_figure.Panel,
    all_series: list[ScatteredSeries],
) -> None:
    """Key each colour to the numbers its series was drawn from; the text carries the colour."""
    annotate_panel.add_custom_legend(
        panel=panel,
        artists=[None for _ in all_series],
        labels=[
            rf"$m = {series.slope:.2f}, \; \sigma = {series.sigma:.2f}$"
            for series in all_series
        ],
        colors=[series.color for series in all_series],
        anchor_point=(0.975, 0.025),
        anchor_at_corner=box_positions.Positions.Corner.BottomRight,
    )


def add_combined_legend(
    *,
    panel: manage_figure.Panel,
    series: ScatteredSeries,
) -> None:
    """With one series there are no colours to key, so the shapes carry its numbers."""
    annotate_panel.add_custom_legend(
        panel=panel,
        artists=[
            "o",
            "--",
        ],
        labels=[
            format_trend(
                series=series,
                include_scatter=True,
            ),
            format_trend(series=series),
        ],
        colors=[
            series.color,
            series.color,
        ],
        anchor_point=(0.0, 1.0),
        anchor_at_corner=box_positions.Positions.Corner.TopLeft,
    )


def add_residual_inset(
    *,
    panel: manage_figure.Panel,
    all_series: list[ScatteredSeries],
) -> None:
    """
    Show what the quoted uncertainties come from: the scatter about each exact trend.

    The y label sits on the right, away from the main panel's own, while the x label keeps
    to the bottom where an x label is read.
    """
    inset_panel = manage_figure.add_inset_panel(
        panel=panel,
        bounds=(0.035, 0.65, 0.35, 0.3),
        x_label=r"$x$",
        y_label=r"$y - m\,x$",
        x_label_alignment=box_positions.Positions.Side.Bottom,
        y_label_alignment=box_positions.Positions.Side.Right,
    )
    max_abs_residual = 0.0
    for series in all_series:
        residual = series.y_values - series.slope * series.x_values
        max_abs_residual = max(max_abs_residual, abs(residual).max())
        inset_panel.plot(
            series.x_values,
            residual,
            marker="o",
            ls="",
            color=series.color,
        )
    inset_panel.axhline(
        0.0,
        ls="--",
        color="black",
    )
    inset_panel.set_ylim(-1.1 * max_abs_residual, 1.1 * max_abs_residual)


def report_drawn_sizes(
    *,
    label: str,
    figure: Any,
    panel: manage_figure.Panel,
) -> None:
    """Log the figure width, the drawn text size, and the room left at either side edge."""
    figure_width_cm = float(figure.get_size_inches()[0]) * style_figure.CM_PER_INCH
    figure_width_pt = figure_width_cm * style_figure.PT_PER_CM
    panel_width_cm = panel.get_position().width * figure_width_cm
    figure_params = style_figure.get_figure_params()
    text_size_params = figure_params.text_size_params
    ink_bounds = panel.get_tightbbox()
    if ink_bounds is None:
        raise RuntimeError("the panel has no drawn extent to measure.")
    ink_box = ink_bounds.transformed(figure.dpi_scale_trans.inverted())
    left_clearance_pt = ink_box.x0 * style_figure.PT_PER_INCH
    right_clearance_pt = figure_width_pt - (ink_box.x1 * style_figure.PT_PER_INCH)
    manage_log.log_action(
        title=label,
        outcome=(
            manage_log.ActionOutcome.SUCCESS
            if min(left_clearance_pt, right_clearance_pt) >= 0.0
            else manage_log.ActionOutcome.FAILURE
        ),
        message="Text holds its size while the panel narrows.",
        notes={
            "figure width": f"{figure_width_cm:.2f} cm",
            "panel width": f"{panel_width_cm:.2f} cm",
            "axis label": f"{text_size_params.axis_label_size:.2f} pt",
            "tick label": f"{text_size_params.tick_label_size:.2f} pt",
            "clearance": f"{left_clearance_pt:.1f} pt left, {right_clearance_pt:.1f} pt right",
        },
    )


##
## === PROGRAM MAIN
##


def main() -> None:
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    style_figure.set_figure_params()
    all_series = generate_all_series()
    figures_dir = Path(__file__).parent
    ## the full page carries every series and the residual inset; the half page carries a
    ## single series in one colour, so the two can be compared on text size alone
    for label, figure_layout, is_full_page in (
        ("full page", style_figure.FULL_PAGE_FIGURE_LAYOUT, True),
        ("half page", style_figure.HALF_PAGE_FIGURE_LAYOUT, False),
    ):
        figure, panel = manage_figure.create_figure(
            figure_layout=figure_layout,
            panel_aspect=PANEL_ASPECT,
        )
        series_to_draw = (
            all_series if is_full_page
            else [dataclasses.replace(all_series[0], color=SINGLE_SERIES_COLOR)]
        )
        for series in series_to_draw:
            plot_series(
                panel=panel,
                series=series,
            )
        label_axes(panel=panel)
        if is_full_page:
            add_shape_legend(panel=panel)
            add_trend_legend(
                panel=panel,
                all_series=all_series,
            )
            add_residual_inset(
                panel=panel,
                all_series=all_series,
            )
        else:
            add_combined_legend(
                panel=panel,
                series=series_to_draw[0],
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
