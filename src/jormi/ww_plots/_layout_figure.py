## { MODULE

##
## === WORKSPACE SETUP
##

import matplotlib

matplotlib.use("Agg", force=True)

##
## === DEPENDENCIES
##

## stdlib
import dataclasses
import weakref

from dataclasses import dataclass
from collections.abc import Sequence
from typing import TypeAlias

## third-party
import numpy
from matplotlib import ticker as mpl_ticker
from matplotlib import transforms as mpl_transforms
from matplotlib.axes import Axes as mpl_Axes
from matplotlib.figure import Figure as mpl_Figure
from matplotlib.text import Text as mpl_Text
from numpy.typing import NDArray

## local
from jormi.ww_plots import style_figure
from jormi.ww_validation import validate_box_positions, validate_types
from jormi.ww_types import box_positions

##
## === TYPE ALIASES
##

Panel: TypeAlias = mpl_Axes
PanelGrid: TypeAlias = NDArray[numpy.object_]

##
## === BOX SHAPE
##


@dataclass(
    frozen=True,
    kw_only=True,
)
class BoxShape:
    """
    Width and height of a rectangle, in cm: a whole figure, or the share it gives one
    panel.

    Named rather than a bare pair, so which of the two is the height never has to be
    remembered; `as_mpl_shape` produces the pair for handing straight to Matplotlib.
    """

    width_cm: float
    height_cm: float

    def __post_init__(self) -> None:
        for param_name in (
            "width_cm",
            "height_cm",
        ):
            param_value = getattr(self, param_name)
            if not (param_value > 0):
                raise ValueError(f"`{param_name}` must be positive, but got {param_value}.")

    @property
    def as_mpl_shape(self) -> tuple[float, float]:
        """The pair, width first and in inches, as Matplotlib's `figsize` reads it."""
        return (
            self.width_cm / style_figure.CM_PER_INCH,
            self.height_cm / style_figure.CM_PER_INCH,
        )

    @property
    def aspect_ratio(self) -> float:
        """Width over height."""
        return self.width_cm / self.height_cm

## where the panels start out. A figure sized to a page is measured and given margins of its
## own when it is saved, so these only stand for a figure sized in its own terms.
DEFAULT_FIGURE_MARGINS: style_figure.FigureMargins = style_figure.FigureMargins()


def _set_figure_margins(
    *,
    figure: mpl_Figure,
    figure_shape: BoxShape,
    figure_margins: style_figure.FigureMargins,
) -> None:
    """
    Leave `figure_margins` clear around the panels in `figure`.

    Margins are in pt, while Matplotlib places panels as fractions of the figure, so
    `figure_shape` is what converts between the two.
    """
    figure_width_pt = style_figure.PT_PER_CM * figure_shape.width_cm
    figure_height_pt = style_figure.PT_PER_CM * figure_shape.height_cm
    if (figure_margins.left + figure_margins.right) >= figure_width_pt:
        raise ValueError(
            f"margins `left` + `right` ({figure_margins.left + figure_margins.right} pt)"
            f" leave no room for the panels in a figure {figure_width_pt:.1f} pt wide.",
        )
    if (figure_margins.bottom + figure_margins.top) >= figure_height_pt:
        raise ValueError(
            f"margins `bottom` + `top` ({figure_margins.bottom + figure_margins.top} pt)"
            f" leave no room for the panels in a figure {figure_height_pt:.1f} pt tall.",
        )
    ## Matplotlib positions panels from the figure's left and bottom edges, so the right
    ## and top margins are measured back from the far edge
    left_position = figure_margins.left / figure_width_pt
    right_position = 1.0 - (figure_margins.right / figure_width_pt)
    bottom_position = figure_margins.bottom / figure_height_pt
    top_position = 1.0 - (figure_margins.top / figure_height_pt)
    figure.subplots_adjust(
        left=left_position,
        right=right_position,
        bottom=bottom_position,
        top=top_position,
    )


def _compute_mpl_panel_gap(
    *,
    gap_length_pt: float,
    figure_length_pt: float,
    start_margin_pt: float,
    end_margin_pt: float,
    num_panels: int,
    param_name: str,
) -> float:
    """
    Convert the gap between two panels (pt) into the fraction of a panel Matplotlib wants.

    Matplotlib measures a gap against the panel beside it, and the panels share whatever
    the margins leave, so the gaps come out of that total length before the panels do.
    """
    total_length_pt = figure_length_pt - start_margin_pt - end_margin_pt
    total_gap_length_pt = (num_panels - 1) * gap_length_pt
    if total_gap_length_pt >= total_length_pt:
        raise ValueError(
            f"`{param_name}` ({gap_length_pt} pt) leaves no room for {num_panels} panels"
            f" in the {total_length_pt:.1f} pt the margins leave.",
        )
    panel_length_pt = (total_length_pt - total_gap_length_pt) / num_panels
    return gap_length_pt / panel_length_pt


def _set_panel_gaps(
    *,
    figure: mpl_Figure,
    figure_shape: BoxShape,
    figure_margins: style_figure.FigureMargins,
    num_panel_rows: int,
    num_panel_columns: int,
    panel_column_gap: float,
    panel_row_gap: float,
) -> None:
    """
    Leave `panel_column_gap` and `panel_row_gap` (pt) between each pair of panels in `figure`.

    Gaps are in pt like the margins, since a gap holds the neighbouring panel's tick and
    axis labels; `figure_shape` and `figure_margins` give the length they are measured in.
    """
    figure_width_pt = style_figure.PT_PER_CM * figure_shape.width_cm
    figure_height_pt = style_figure.PT_PER_CM * figure_shape.height_cm
    figure.subplots_adjust(
        wspace=_compute_mpl_panel_gap(
            gap_length_pt=panel_column_gap,
            figure_length_pt=figure_width_pt,
            start_margin_pt=figure_margins.left,
            end_margin_pt=figure_margins.right,
            num_panels=num_panel_columns,
            param_name="panel_column_gap",
        ),
        hspace=_compute_mpl_panel_gap(
            gap_length_pt=panel_row_gap,
            figure_length_pt=figure_height_pt,
            start_margin_pt=figure_margins.bottom,
            end_margin_pt=figure_margins.top,
            num_panels=num_panel_rows,
            param_name="panel_row_gap",
        ),
    )
##
## === FITTING A FIGURE TO WHAT IT DRAWS
##

_Side = box_positions.Positions.Side


@dataclasses.dataclass(kw_only=True)
class ColorbarPlacement:
    """Where one colorbar sits, in the terms its bounds were computed from."""

    colorbar_panel: Panel
    panels: list[Panel]
    side: _Side
    aspect_ratio: float
    length: float
    gap_pt: float
    ## the bar is shaped by its own proportions, so its thickness follows from how long it
    ## is, and how long it is follows from how many panels it spans
    span_rows: int
    span_columns: int


@dataclasses.dataclass(kw_only=True)
class SharedLabelPlacement:
    """
    A label naming an axis that several panels share.

    It sits outside those panels' own tick and axis labels, so where it goes is not known
    until they have been measured; only its text is fixed.
    """

    text: mpl_Text
    panels: list[Panel]
    side: _Side
    gap_pt: float


@dataclasses.dataclass(kw_only=True)
class FigureFit:
    """
    What a figure needs to be fitted to its contents.

    `panel_aspect` is the width / height of the panel as it is drawn, so what the margins
    hold is taken out of the figure around it rather than out of the panel.
    """

    num_panel_rows: int
    num_panel_columns: int
    panel_aspect: float
    figure_padding: style_figure.FigurePadding
    panel_column_gap: float
    panel_row_gap: float
    colorbars: list[ColorbarPlacement] = dataclasses.field(default_factory=list)
    shared_labels: list[SharedLabelPlacement] = dataclasses.field(default_factory=list)


## a figure is fitted from what it holds, so what it holds is recorded as it is built; kept
## beside the figure rather than on it, so nothing is attached to Matplotlib's own objects
_FIGURE_FITS: weakref.WeakKeyDictionary[mpl_Figure, FigureFit] = weakref.WeakKeyDictionary()


def as_panel_list(
    *,
    panels: Panel | PanelGrid | Sequence[Panel],
) -> list[Panel]:
    """Flatten however a caller names one panel or many into a plain list."""
    if isinstance(panels, mpl_Axes):
        return [panels]
    if isinstance(panels, numpy.ndarray):
        return [panel for panel in panels.flatten().tolist()]
    return list(panels)


def register_colorbar(
    *,
    colorbar_panel: Panel,
    panels: list[Panel],
    side: _Side,
    aspect_ratio: float,
    length: float,
    gap_pt: float,
) -> None:
    """Record a colorbar against its figure, so a later fit can measure and replace it."""
    figure_fit = _get_figure_fit(panels=panels)
    if figure_fit is None:
        return
    span_rows, span_columns = _count_panel_span(panels=panels)
    figure_fit.colorbars.append(
        ColorbarPlacement(
            colorbar_panel=colorbar_panel,
            panels=panels,
            side=side,
            aspect_ratio=aspect_ratio,
            length=length,
            gap_pt=gap_pt,
            span_rows=span_rows,
            span_columns=span_columns,
        ),
    )


def compute_colorbar_thickness_share(
    *,
    panels: Panel | PanelGrid | Sequence[Panel],
    side: _Side,
    length: float,
    aspect_ratio: float,
) -> float:
    """
    Convert a bar's length-over-thickness into the share of what it neighbours that
    Matplotlib measures a neighbouring panel's thickness by.

    Length runs along the bar and thickness across it, so which dimension each is taken
    from swaps with the side the bar sits on.
    """
    described_panels = as_panel_list(panels=panels)
    figure = described_panels[0].get_figure(root=True)
    if figure is None:
        raise ValueError("`panels` do not belong to a figure, so they have no size to measure against.")
    figure_width_pt = float(figure.get_size_inches()[0]) * style_figure.PT_PER_INCH
    figure_height_pt = float(figure.get_size_inches()[1]) * style_figure.PT_PER_INCH
    box = mpl_transforms.Bbox.union([panel.get_position() for panel in described_panels])
    described_width_pt = box.width * figure_width_pt
    described_height_pt = box.height * figure_height_pt
    if side in (_Side.Left, _Side.Right):
        length_pt, across_pt = length * described_height_pt, described_width_pt
    else:
        length_pt, across_pt = length * described_width_pt, described_height_pt
    return (length_pt / aspect_ratio) / across_pt


def _count_panel_span(
    *,
    panels: list[Panel],
) -> tuple[int, int]:
    """How many rows and columns of the grid `panels` covers between them."""
    rows: set[int] = set()
    columns: set[int] = set()
    for panel in panels:
        subplot_spec = panel.get_subplotspec()
        if subplot_spec is None:
            continue
        rows.update(range(subplot_spec.rowspan.start, subplot_spec.rowspan.stop))
        columns.update(range(subplot_spec.colspan.start, subplot_spec.colspan.stop))
    return max(len(rows), 1), max(len(columns), 1)


def register_shared_label(
    *,
    text: mpl_Text,
    panels: list[Panel],
    side: _Side,
    gap_pt: float,
) -> None:
    """Record a shared label against its figure, so a later fit can place it outside the panels."""
    figure_fit = _get_figure_fit(panels=panels)
    if figure_fit is None:
        return
    figure_fit.shared_labels.append(
        SharedLabelPlacement(
            text=text,
            panels=panels,
            side=side,
            gap_pt=gap_pt,
        ),
    )


def _get_figure_fit(
    *,
    panels: list[Panel],
) -> FigureFit | None:
    """The fit recorded for the figure `panels` belong to, if it was built to be fitted."""
    if not panels:
        return None
    figure = panels[0].get_figure(root=True)
    if figure is None:
        return None
    return _FIGURE_FITS.get(figure)


def _measure_content_beyond_panels(
    *,
    figure: mpl_Figure,
    figure_fit: FigureFit,
) -> dict[str, float]:
    """
    Measure how far the labels reach past the panels, per side, in pt.

    What is measured is text, drawn at absolute pt sizes, so what comes back does not change
    when the panels are resized. A colorbar's own box is taken off first, since that part is
    proportional to the panel and is solved for rather than measured.
    """
    ## an automatic locator picks its ticks from the size of the panel, so resizing a panel can
    ## relabel it and change what its labels take up; pinning them to what is measured here is
    ## what lets one measurement stand, rather than measuring and resizing in turn
    for panel in figure.axes:
        for axis in (panel.xaxis, panel.yaxis):
            axis.set_major_locator(mpl_ticker.FixedLocator(list(axis.get_majorticklocs())))
    renderer = figure.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
    pt_per_pixel = style_figure.PT_PER_INCH / figure.dpi
    panels = [
        panel for panel in figure.axes
        if all(placement.colorbar_panel is not panel for placement in figure_fit.colorbars)
    ]
    panel_block = mpl_transforms.Bbox.union([panel.get_window_extent(renderer) for panel in panels])
    ## text belonging to the figure rather than to a panel is drawn outside every panel, so it
    ## is measured alongside them; a shared label is left out, since where it sits is computed
    ## from this measurement rather than being an input to it
    shared_label_texts = [placement.text for placement in figure_fit.shared_labels]
    figure_artists = [
        artist for artist in (*figure.texts, *figure.legends) if artist not in shared_label_texts
    ]
    content = mpl_transforms.Bbox.union(
        [panel.get_tightbbox(renderer) for panel in figure.axes] +
        [artist.get_window_extent(renderer) for artist in figure_artists],
    )
    beyond = {
        "left": (panel_block.x0 - content.x0) * pt_per_pixel,
        "right": (content.x1 - panel_block.x1) * pt_per_pixel,
        "bottom": (panel_block.y0 - content.y0) * pt_per_pixel,
        "top": (content.y1 - panel_block.y1) * pt_per_pixel,
    }
    ## a colorbar's box and its gap are geometry, not text, so take them off what was measured;
    ## the deepest bar per side, since bars on one side of different panels sit beside each other
    colorbar_reach = dict.fromkeys(beyond, 0.0)
    for placement in figure_fit.colorbars:
        box = mpl_transforms.Bbox.union(
            [panel.get_window_extent(renderer) for panel in placement.panels],
        )
        side_name = placement.side.name.lower()
        is_beside_panels = placement.side in (_Side.Left, _Side.Right)
        along_bar = box.height if is_beside_panels else box.width
        thickness = (placement.length * along_bar) / placement.aspect_ratio
        reach = placement.gap_pt + thickness * pt_per_pixel
        colorbar_reach[side_name] = max(colorbar_reach[side_name], reach)
    return {side_name: extent - colorbar_reach[side_name] for side_name, extent in beyond.items()}


def _colorbar_reach_on_sides(
    *,
    figure_fit: FigureFit,
    sides: tuple[_Side, ...],
) -> tuple[float, float]:
    """
    Return how far the colorbars on `sides` reach past the panels, split into the part that
    scales with one panel and the part that does not (pt).

    A bar spanning several panels is a share of all of them together, so its thickness
    scales with that many panels and with the gaps between them. The deepest bar is taken
    rather than the total, since bars on one side of different panels sit beside each other.
    """
    is_beside_panels = _Side.Left in sides or _Side.Right in sides
    ## the bar runs across the panels it sits beside, so its length is spanned in the other
    ## direction: a bar on the left is as long as the rows are tall
    span_gap = figure_fit.panel_row_gap if is_beside_panels else figure_fit.panel_column_gap
    panel_coefficient = 0.0
    constant_pt = 0.0
    for placement in figure_fit.colorbars:
        if placement.side not in sides:
            continue
        span = placement.span_rows if is_beside_panels else placement.span_columns
        ## length runs along the bar, and a bar beside the panels is as long as they are tall,
        ## which is the panel width over the panel's aspect
        panel_length_coefficient = 1.0 / figure_fit.panel_aspect if is_beside_panels else 1.0
        reach_coefficient = placement.length * span * panel_length_coefficient / placement.aspect_ratio
        reach_constant = (
            placement.gap_pt + placement.length * (span - 1) * span_gap / placement.aspect_ratio
        )
        if reach_coefficient + reach_constant > panel_coefficient + constant_pt:
            panel_coefficient, constant_pt = reach_coefficient, reach_constant
    return panel_coefficient, constant_pt


def _shared_label_reach_per_side(
    *,
    figure: mpl_Figure,
    figure_fit: FigureFit,
) -> dict[str, float]:
    """How far the shared labels reach past the panels' own labels, per side, in pt."""
    renderer = figure.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
    pt_per_pixel = style_figure.PT_PER_INCH / figure.dpi
    reach = {
        "left": 0.0,
        "right": 0.0,
        "bottom": 0.0,
        "top": 0.0,
    }
    for placement in figure_fit.shared_labels:
        box = placement.text.get_window_extent(renderer)
        is_beside_panels = placement.side in (_Side.Left, _Side.Right)
        extent_pt = (box.width if is_beside_panels else box.height) * pt_per_pixel
        side_name = placement.side.name.lower()
        reach[side_name] = max(reach[side_name], placement.gap_pt + extent_pt)
    return reach


def _place_shared_labels(
    *,
    figure: mpl_Figure,
    figure_fit: FigureFit,
    panel_reach: dict[str, float],
) -> None:
    """Put each shared label just outside the panels' own labels, centred on the panels."""
    renderer = figure.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
    figure_width_pt = float(figure.get_size_inches()[0]) * style_figure.PT_PER_INCH
    figure_height_pt = float(figure.get_size_inches()[1]) * style_figure.PT_PER_INCH
    pt_per_pixel = style_figure.PT_PER_INCH / figure.dpi
    for placement in figure_fit.shared_labels:
        block = mpl_transforms.Bbox.union(
            [panel.get_window_extent(renderer) for panel in placement.panels],
        )
        box = placement.text.get_window_extent(renderer)
        is_beside_panels = placement.side in (_Side.Left, _Side.Right)
        extent_pt = (box.width if is_beside_panels else box.height) * pt_per_pixel
        offset_pt = panel_reach[placement.side.name.lower()] + placement.gap_pt + extent_pt / 2.0
        if is_beside_panels:
            edge_pt = (block.x0 if placement.side is _Side.Left else block.x1) * pt_per_pixel
            sign = -1.0 if placement.side is _Side.Left else 1.0
            x_position = (edge_pt + sign * offset_pt) / figure_width_pt
            y_position = 0.5 * (block.y0 + block.y1) * pt_per_pixel / figure_height_pt
        else:
            edge_pt = (block.y0 if placement.side is _Side.Bottom else block.y1) * pt_per_pixel
            sign = -1.0 if placement.side is _Side.Bottom else 1.0
            x_position = 0.5 * (block.x0 + block.x1) * pt_per_pixel / figure_width_pt
            y_position = (edge_pt + sign * offset_pt) / figure_height_pt
        placement.text.set_position((x_position, y_position))


def fit_figure_to_content(
    *,
    figure: mpl_Figure,
) -> None:
    """
    Resize `figure` so its labels sit inside it, with the padding the style asks for.

    The page fixes the width, so the panels are what give horizontally: the panel width is
    solved for. Nothing fixes the height, so the figure is what gives vertically, and its
    height follows. Does nothing to a figure that was not built to be fitted.
    """
    figure_fit = _FIGURE_FITS.get(figure)
    if figure_fit is None:
        return
    padding = figure_fit.figure_padding
    figure_width_pt = float(figure.get_size_inches()[0]) * style_figure.PT_PER_INCH
    panel_reach = _measure_content_beyond_panels(
        figure=figure,
        figure_fit=figure_fit,
    )
    ## a shared label sits beyond the panels' own labels, so it adds to what each side holds
    shared_reach = _shared_label_reach_per_side(
        figure=figure,
        figure_fit=figure_fit,
    )
    beyond = {side_name: panel_reach[side_name] + shared_reach[side_name] for side_name in panel_reach}
    ## horizontal: the width is fixed, so solve for the panel width
    side_thickness, side_gap_pt = _colorbar_reach_on_sides(
        figure_fit=figure_fit,
        sides=(_Side.Left, _Side.Right),
    )
    column_gaps_pt = (figure_fit.num_panel_columns - 1) * figure_fit.panel_column_gap
    horizontal_constants_pt = (
        padding.left + padding.right + beyond["left"] + beyond["right"] + column_gaps_pt + side_gap_pt
    )
    panel_width_pt = (figure_width_pt - horizontal_constants_pt) / (
        figure_fit.num_panel_columns + side_thickness
    )
    if panel_width_pt <= 0:
        raise ValueError(
            f"the labels and padding need {horizontal_constants_pt:.1f} pt of a figure only"
            f" {figure_width_pt:.1f} pt wide, so there is no room left for the panels.",
        )
    ## vertical: the height is free, so it follows from the panels
    panel_height_pt = panel_width_pt / figure_fit.panel_aspect
    stacked_thickness, stacked_gap_pt = _colorbar_reach_on_sides(
        figure_fit=figure_fit,
        sides=(_Side.Bottom, _Side.Top),
    )
    row_gaps_pt = (figure_fit.num_panel_rows - 1) * figure_fit.panel_row_gap
    ## a bar above or below is as long as the panels are wide, so its thickness scales with
    ## the panel width that was just solved for, not with the panel height it eats into
    figure_height_pt = (
        padding.bottom + padding.top + beyond["bottom"] + beyond["top"] + row_gaps_pt + stacked_gap_pt +
        panel_height_pt * figure_fit.num_panel_rows + stacked_thickness * panel_width_pt
    )
    ## margins hold the labels, the padding, and any colorbar sitting outside the panels
    left_thickness, left_gap_pt = _colorbar_reach_on_sides(figure_fit=figure_fit, sides=(_Side.Left,))
    right_thickness, right_gap_pt = _colorbar_reach_on_sides(figure_fit=figure_fit, sides=(_Side.Right,))
    bottom_thickness, bottom_gap_pt = _colorbar_reach_on_sides(figure_fit=figure_fit, sides=(_Side.Bottom,))
    top_thickness, top_gap_pt = _colorbar_reach_on_sides(figure_fit=figure_fit, sides=(_Side.Top,))
    figure_margins = style_figure.FigureMargins(
        left=padding.left + beyond["left"] + left_gap_pt + left_thickness * panel_width_pt,
        right=padding.right + beyond["right"] + right_gap_pt + right_thickness * panel_width_pt,
        bottom=padding.bottom + beyond["bottom"] + bottom_gap_pt + bottom_thickness * panel_width_pt,
        top=padding.top + beyond["top"] + top_gap_pt + top_thickness * panel_width_pt,
    )
    figure_shape = BoxShape(
        width_cm=figure_width_pt / style_figure.PT_PER_CM,
        height_cm=figure_height_pt / style_figure.PT_PER_CM,
    )
    figure.set_size_inches(figure_shape.as_mpl_shape)
    _set_figure_margins(
        figure=figure,
        figure_shape=figure_shape,
        figure_margins=figure_margins,
    )
    _set_panel_gaps(
        figure=figure,
        figure_shape=figure_shape,
        figure_margins=figure_margins,
        num_panel_rows=figure_fit.num_panel_rows,
        num_panel_columns=figure_fit.num_panel_columns,
        panel_column_gap=figure_fit.panel_column_gap,
        panel_row_gap=figure_fit.panel_row_gap,
    )
    ## the panels have moved, so each shared label is placed again just outside their own
    _place_shared_labels(
        figure=figure,
        figure_fit=figure_fit,
        panel_reach=panel_reach,
    )
    ## the panels have moved, so every colorbar is placed again against where they are now
    for placement in figure_fit.colorbars:
        bounds = compute_neighbouring_panel_bounds(
            panels=placement.panels,
            side=placement.side,
            thickness=compute_colorbar_thickness_share(
                panels=placement.panels,
                side=placement.side,
                length=placement.length,
                aspect_ratio=placement.aspect_ratio,
            ),
            length=placement.length,
            gap=placement.gap_pt / (
                figure_width_pt if placement.side in (_Side.Left, _Side.Right) else figure_height_pt
            ),
        )
        placement.colorbar_panel.set_position(
            (
                bounds.x_min,
                bounds.y_min,
                bounds.x_width,
                bounds.y_width,
            ),
        )

##


@dataclass(frozen=True)
class PanelBounds:
    """Bounding box for a panel in figure coordinates."""

    x_min: float
    y_min: float
    x_width: float
    y_width: float


def compute_neighbouring_panel_bounds(
    *,
    panels: Panel | PanelGrid | Sequence[Panel],
    side: _Side = box_positions.Positions.Side.Right,
    gap: float = 0.1,
    thickness: float = 1.0,
    length: float = 1.0,
) -> PanelBounds:
    """
    Compute figure bounds for a panel placed neighbouring `panels`.

    The new panel sits on the `side` of `panels`, offset by `gap` (in figure coordinates).
    `thickness` sets its extent perpendicular to `side`, as a fraction of the corresponding
    dimension of what it neighbours. `length` sets its extent parallel to `side`, also as a
    fraction, centered on that edge. Given several panels it neighbours all of them, which
    is how one colorbar comes to describe a whole grid.
    """
    for param_name, param_value in (
        ("thickness", thickness),
        ("length", length),
    ):
        validate_types.ensure_finite_float(
            param=param_value,
            param_name=param_name,
            allow_none=False,
            require_positive=True,
            allow_zero=False,
        )
    ## a panel may sit flush against the one it neighbours, so no gap is a gap of zero
    validate_types.ensure_finite_float(
        param=gap,
        param_name="gap",
        allow_none=False,
        require_positive=True,
        allow_zero=True,
    )
    box = mpl_transforms.Bbox.union([panel.get_position() for panel in as_panel_list(panels=panels)])
    if side in (_Side.Left, _Side.Right):
        x_width = box.width * thickness
        y_width = box.height * length
        if side == _Side.Right:
            return PanelBounds(
                x_min=box.x1 + gap,
                y_min=box.y0 + (box.height - y_width) / 2,
                x_width=x_width,
                y_width=y_width,
            )
        else:
            return PanelBounds(
                x_min=box.x0 - x_width - gap,
                y_min=box.y0 + (box.height - y_width) / 2,
                x_width=x_width,
                y_width=y_width,
            )
    elif side in (_Side.Top, _Side.Bottom):
        x_width = box.width * length
        y_width = box.height * thickness
        if side == _Side.Top:
            return PanelBounds(
                x_min=box.x0 + (box.width - x_width) / 2,
                y_min=box.y1 + gap,
                x_width=x_width,
                y_width=y_width,
            )
        else:
            return PanelBounds(
                x_min=box.x0 + (box.width - x_width) / 2,
                y_min=box.y0 - y_width - gap,
                x_width=x_width,
                y_width=y_width,
            )
    else:
        raise ValueError(f"unexpected side: {side!r}.")  # pyright: ignore[reportUnreachable]


## } MODULE
