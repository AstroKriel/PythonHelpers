## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import dataclasses
import typing
import weakref

from collections import abc as collections_abc

## third-party
import numpy

from matplotlib import artist as mpl_artist
from matplotlib import axes as mpl_axes
from matplotlib import backend_bases as mpl_backend_bases
from matplotlib import figure as mpl_figure
from matplotlib import text as mpl_text
from matplotlib import ticker as mpl_ticker
from matplotlib import transforms as mpl_transforms
from numpy import typing as numpy_typing

## local
from jormi.ww_plots import style_figure
from jormi.ww_validation import validate_types
from jormi.ww_types import box_positions

##
## === TYPE ALIASES
##

Panel: typing.TypeAlias = mpl_axes.Axes
PanelGrid: typing.TypeAlias = numpy_typing.NDArray[numpy.object_]
PanelLike: typing.TypeAlias = Panel | PanelGrid | collections_abc.Sequence[Panel]

##
## === BOX SHAPE
##


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class BoxShape:
    """Width and height of a rectangle (in cm)."""

    width_cm: float
    height_cm: float

    def __post_init__(
        self,
    ) -> None:
        for param_name in (
                "width_cm",
                "height_cm",
        ):
            param_value = getattr(self, param_name)
            if not (param_value > 0):
                raise ValueError(f"`{param_name}` must be positive, but got {param_value}.")

    @property
    def as_mpl_shape(
        self,
    ) -> tuple[float, float]:
        """The pair, width first and in inches, in the order the Matplotlib `figsize` argument wants."""
        return (
            self.width_cm / style_figure.CM_PER_INCH,
            self.height_cm / style_figure.CM_PER_INCH,
        )

    @property
    def aspect_ratio(
        self,
    ) -> float:
        """Width over height."""
        return self.width_cm / self.height_cm


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class FigureSize:
    """The current size of a figure (in pt)."""

    figure: mpl_figure.Figure

    @property
    def width_pt(
        self,
    ) -> float:
        return style_figure.PT_PER_INCH * float(self.figure.get_size_inches()[0])

    @property
    def height_pt(
        self,
    ) -> float:
        return style_figure.PT_PER_INCH * float(self.figure.get_size_inches()[1])


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class FigureMargins:
    """Space between the figure edge and the axes (in pt; 1 pt = 1/72 inch)."""

    left_pt: float
    right_pt: float
    bottom_pt: float
    top_pt: float

    def __post_init__(
        self,
    ) -> None:
        for param_name in (
                "left_pt",
                "right_pt",
                "bottom_pt",
                "top_pt",
        ):
            param_value = getattr(self, param_name)
            if param_value < 0:
                raise ValueError(f"`{param_name}` must not be negative, but got {param_value}.")


## zero margin lets every panel start as large as possible, so its ticks come from a fixed size
INITIAL_FIGURE_MARGINS: FigureMargins = FigureMargins(
    left_pt=0.0,
    right_pt=0.0,
    bottom_pt=0.0,
    top_pt=0.0,
)


def set_figure_margins(
    *,
    figure: mpl_figure.Figure,
    figure_shape: BoxShape,
    figure_margins: FigureMargins,
) -> None:
    """
    Leave `figure_margins` clear around the panels in `figure`.

    Margins are in pt; `figure_shape` converts them to the fractions Matplotlib places
    panels with.
    """
    figure_width_pt = style_figure.PT_PER_CM * figure_shape.width_cm
    figure_height_pt = style_figure.PT_PER_CM * figure_shape.height_cm
    if (figure_margins.left_pt + figure_margins.right_pt) >= figure_width_pt:
        raise ValueError(
            f"margins `left_pt` + `right_pt` ({figure_margins.left_pt + figure_margins.right_pt} pt)"
            f" leave no room for the panels in a figure {figure_width_pt:.1f} pt wide.",
        )
    if (figure_margins.bottom_pt + figure_margins.top_pt) >= figure_height_pt:
        raise ValueError(
            f"margins `bottom_pt` + `top_pt` ({figure_margins.bottom_pt + figure_margins.top_pt} pt)"
            f" leave no room for the panels in a figure {figure_height_pt:.1f} pt tall.",
        )
    ## Matplotlib positions panels from the left and bottom edges of the figure; the right
    ## and top margins are measured back from the far edge
    left_position = figure_margins.left_pt / figure_width_pt
    right_position = 1.0 - (figure_margins.right_pt / figure_width_pt)
    bottom_position = figure_margins.bottom_pt / figure_height_pt
    top_position = 1.0 - (figure_margins.top_pt / figure_height_pt)
    figure.subplots_adjust(
        left=left_position,
        right=right_position,
        bottom=bottom_position,
        top=top_position,
    )


def _compute_mpl_panel_gap(
    *,
    param_name: str,
    panel_gap_pt: float,
    figure_length_pt: float,
    margin_lo_pt: float,
    margin_hi_pt: float,
    num_panels: int,
) -> float:
    """
    Convert the gap between two panels (in pt) into the fraction of a panel Matplotlib wants.

    Matplotlib measures a gap against the panel beside it. The panels share whatever the
    margins leave, so a gap comes out of that length before the panels do.
    """
    total_gap_pt = panel_gap_pt * (num_panels - 1)
    total_length_pt = figure_length_pt - margin_lo_pt - margin_hi_pt
    if total_gap_pt >= total_length_pt:
        raise ValueError(
            f"`{param_name}` ({panel_gap_pt} pt) leaves no room for {num_panels} panels"
            f" in the {total_length_pt:.1f} pt the margins leave.",
        )
    panel_length_pt = (total_length_pt - total_gap_pt) / num_panels
    return panel_gap_pt / panel_length_pt


def set_panel_gaps(
    *,
    figure: mpl_figure.Figure,
    figure_shape: BoxShape,
    figure_margins: FigureMargins,
    num_panel_rows: int,
    num_panel_cols: int,
    panel_row_gap_pt: float,
    panel_col_gap_pt: float,
) -> None:
    """
    Leave `panel_row_gap_pt` and `panel_col_gap_pt` between each pair of panels in `figure`.

    Gaps are in pt like the margins, since a gap holds the tick and axis labels of the
    neighbouring panel; `figure_shape` and `figure_margins` give the length they are
    measured in.
    """
    figure_width_pt = style_figure.PT_PER_CM * figure_shape.width_cm
    figure_height_pt = style_figure.PT_PER_CM * figure_shape.height_cm
    figure.subplots_adjust(
        wspace=_compute_mpl_panel_gap(
            param_name="panel_col_gap_pt",
            panel_gap_pt=panel_col_gap_pt,
            figure_length_pt=figure_width_pt,
            margin_lo_pt=figure_margins.left_pt,
            margin_hi_pt=figure_margins.right_pt,
            num_panels=num_panel_cols,
        ),
        hspace=_compute_mpl_panel_gap(
            param_name="panel_row_gap_pt",
            panel_gap_pt=panel_row_gap_pt,
            figure_length_pt=figure_height_pt,
            margin_lo_pt=figure_margins.bottom_pt,
            margin_hi_pt=figure_margins.top_pt,
            num_panels=num_panel_rows,
        ),
    )


##
## === FITTING A FIGURE TO WHAT IT DRAWS
##

_Side = box_positions.Positions.Side


@dataclasses.dataclass(kw_only=True)
class _PanelGroup:
    """A set of neighbouring panels, and how much of the grid they span."""

    panels: list[Panel]
    num_rows: int
    num_cols: int

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_flat_list(
            param=self.panels,
            param_name="panels",
            valid_elem_types=Panel,
        )


@dataclasses.dataclass(kw_only=True)
class _ColorbarPlacement:
    """Where one colorbar sits, in the terms its bounds were computed from."""

    colorbar_panel: Panel
    neighbouring_panels: _PanelGroup
    side: _Side
    length_fraction: float
    aspect_ratio: float
    gap_pt: float

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_type(
            param=self.colorbar_panel,
            param_name="colorbar_panel",
            valid_types=Panel,
        )
        validate_types.ensure_type(
            param=self.neighbouring_panels,
            param_name="neighbouring_panels",
            valid_types=_PanelGroup,
        )


@dataclasses.dataclass(kw_only=True)
class _SharedLabelPlacement:
    """
    A label naming an axis that several panels share.

    It sits outside the tick and axis labels each panel draws on its own, so where it
    goes is not known until those are measured; only its text is fixed.
    """

    text: mpl_text.Text
    panels: list[Panel]
    side: _Side
    gap_pt: float

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_type(
            param=self.text,
            param_name="text",
            valid_types=mpl_text.Text,
        )
        validate_types.ensure_flat_list(
            param=self.panels,
            param_name="panels",
            valid_elem_types=Panel,
        )


@dataclasses.dataclass(kw_only=True)
class ResolvedLayout:
    """What a figure needs to be fitted to its contents."""

    num_panel_rows: int
    num_panel_cols: int
    panel_width_pt: float | None = None
    panel_aspect_ratio: float
    panel_row_gap_pt: float
    panel_col_gap_pt: float
    figure_padding: style_figure.FigurePadding
    shared_labels: list[_SharedLabelPlacement] = dataclasses.field(default_factory=list)
    colorbars: list[_ColorbarPlacement] = dataclasses.field(default_factory=list)


## a figure is fitted from what it holds, so what it holds is recorded as it is built; kept
## beside the figure rather than on it, so nothing is attached to any Matplotlib object
RESOLVED_LAYOUTS: weakref.WeakKeyDictionary[mpl_figure.Figure, ResolvedLayout] = weakref.WeakKeyDictionary()


def as_panel_list(
    *,
    panels: PanelLike,
) -> list[Panel]:
    """Flatten however a caller names one panel or many into a plain list."""
    if isinstance(panels, mpl_axes.Axes):
        return [panels]
    if isinstance(panels, numpy.ndarray):
        return [panel for panel in panels.flatten().tolist()]
    return list(panels)


def get_figure(
    *,
    panels: list[Panel],
    param_name: str = "panels",
) -> mpl_figure.Figure:
    """The root figure `panels` belong to, raising if it is empty or unattached to one."""
    if not panels:
        raise ValueError(f"`{param_name}` must name at least one panel.")
    figure = panels[0].get_figure(root=True)
    if figure is None:
        raise ValueError(f"`{param_name}` do not belong to a figure.")
    return figure


def _get_renderer(
    *,
    figure: mpl_figure.Figure,
) -> mpl_backend_bases.RendererBase:
    """The renderer `figure` was last drawn with."""
    return figure.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]


def _get_resolved_layout(
    *,
    panels: list[Panel],
) -> ResolvedLayout | None:
    """The layout recorded for the figure `panels` belong to, if it was built to be fitted."""
    figure = get_figure(
        panels=panels,
        param_name="panels",
    )
    return RESOLVED_LAYOUTS.get(figure)


def _count_panel_rows_and_cols(
    *,
    panels: list[Panel],
) -> tuple[int, int]:
    """How many rows and columns of the grid `panels` covers between them."""
    rows: set[int] = set()
    cols: set[int] = set()
    for panel in panels:
        mpl_subplotspec = panel.get_subplotspec()
        if mpl_subplotspec is None:
            continue
        rows.update(
            range(
                mpl_subplotspec.rowspan.start,
                mpl_subplotspec.rowspan.stop,
            ),
        )
        cols.update(
            range(
                mpl_subplotspec.colspan.start,
                mpl_subplotspec.colspan.stop,
            ),
        )
    return (
        max(len(rows), 1),
        max(len(cols), 1),
    )


def register_colorbar(
    *,
    colorbar_panel: Panel,
    neighbouring_panels: list[Panel],
    side: _Side,
    length_fraction: float,
    aspect_ratio: float,
    gap_pt: float,
) -> None:
    """Record a colorbar against its figure, so it can be measured and replaced once resolved."""
    resolved_layout = _get_resolved_layout(panels=neighbouring_panels)
    if resolved_layout is None:
        return
    num_panel_rows, num_panel_cols = _count_panel_rows_and_cols(panels=neighbouring_panels)
    resolved_layout.colorbars.append(
        _ColorbarPlacement(
            colorbar_panel=colorbar_panel,
            neighbouring_panels=_PanelGroup(
                panels=neighbouring_panels,
                num_rows=num_panel_rows,
                num_cols=num_panel_cols,
            ),
            side=side,
            length_fraction=length_fraction,
            aspect_ratio=aspect_ratio,
            gap_pt=gap_pt,
        ),
    )


def _compute_panel_fractional_bounding_box(
    *,
    panels: list[Panel],
) -> mpl_transforms.Bbox:
    """The bounding box `panels` share (in figure coordinates)."""
    return mpl_transforms.Bbox.union([panel.get_position() for panel in panels])


def compute_colorbar_thickness_fraction(
    *,
    neighbouring_panels: PanelLike,
    side: _Side,
    length_fraction: float,
    aspect_ratio: float,
) -> float:
    """
    Convert the length-over-thickness of a bar into the share Matplotlib uses to measure
    the thickness of a neighbouring panel.

    Length runs along the bar and thickness runs across it, so which dimension is which
    swaps with the side the bar sits on.
    """
    neighbouring_panels = as_panel_list(panels=neighbouring_panels)
    figure = get_figure(
        panels=neighbouring_panels,
        param_name="neighbouring_panels",
    )
    figure_size = FigureSize(figure=figure)
    figure_width_pt = figure_size.width_pt
    figure_height_pt = figure_size.height_pt
    panel_fractional_bounding_box = _compute_panel_fractional_bounding_box(panels=neighbouring_panels)
    neighbouring_width_pt = panel_fractional_bounding_box.width * figure_width_pt
    neighbouring_height_pt = panel_fractional_bounding_box.height * figure_height_pt
    if side in (_Side.Left, _Side.Right):
        length_pt = length_fraction * neighbouring_height_pt
        neighbouring_thickness_pt = neighbouring_width_pt
    else:
        length_pt = length_fraction * neighbouring_width_pt
        neighbouring_thickness_pt = neighbouring_height_pt
    thickness_pt = length_pt / aspect_ratio
    return thickness_pt / neighbouring_thickness_pt


def register_shared_label(
    *,
    text: mpl_text.Text,
    panels: list[Panel],
    side: _Side,
    gap_pt: float,
) -> None:
    """Record a shared label against its figure, so it can be placed outside the panels once resolved."""
    resolved_layout = _get_resolved_layout(panels=panels)
    if resolved_layout is None:
        return
    resolved_layout.shared_labels.append(
        _SharedLabelPlacement(
            text=text,
            panels=panels,
            side=side,
            gap_pt=gap_pt,
        ),
    )


def _compute_panel_pixel_bounding_box(
    *,
    panels: list[Panel],
    renderer: mpl_backend_bases.RendererBase,
) -> mpl_transforms.Bbox:
    """The rendered extent `panels` share (in pixels)."""
    return mpl_transforms.Bbox.union([panel.get_window_extent(renderer) for panel in panels])


def _compute_content_pixel_bounding_box(
    *,
    figure: mpl_figure.Figure,
    extra_artists: list[mpl_artist.Artist],
    renderer: mpl_backend_bases.RendererBase,
) -> mpl_transforms.Bbox:
    """The rendered extent every panel and figure-level artist shares (in pixels)."""
    panel_content_bounding_boxes = [panel.get_tightbbox(renderer) for panel in figure.axes]
    extra_content_bounding_boxes = [artist.get_window_extent(renderer) for artist in extra_artists]
    return mpl_transforms.Bbox.union(panel_content_bounding_boxes + extra_content_bounding_boxes)


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class _SideOverflow:
    """How far something overflows past the panels (per side, in pt)."""

    left_pt: float
    right_pt: float
    bottom_pt: float
    top_pt: float

    def __post_init__(
        self,
    ) -> None:
        for param_name in (
                "left_pt",
                "right_pt",
                "bottom_pt",
                "top_pt",
        ):
            validate_types.ensure_finite_scalar(
                param=getattr(self, param_name),
                param_name=param_name,
                allow_none=False,
                require_positive=True,
                allow_zero=True,
            )

    def value_for_side(
        self,
        *,
        side: _Side,
    ) -> float:
        """The overflow on `side`."""
        if side is _Side.Left:
            return self.left_pt
        if side is _Side.Right:
            return self.right_pt
        if side is _Side.Bottom:
            return self.bottom_pt
        if side is _Side.Top:
            return self.top_pt
        raise ValueError(f"unexpected side: {side!r}.")  # pyright: ignore[reportUnreachable]


def _freeze_panel_tick_locations(
    *,
    figure: mpl_figure.Figure,
) -> None:
    """
    Pin the ticks of every panel to where they currently sit.

    An automatic locator picks its ticks from the size of the panel, so resizing a panel
    can relabel it and change what its labels take up. Pinning them here lets one
    measurement stand instead of measuring and resizing in turn.
    """
    for panel in figure.axes:
        for axis in (panel.xaxis, panel.yaxis):
            axis.set_major_locator(mpl_ticker.FixedLocator(list(axis.get_majorticklocs())))


def _measure_colorbar_overflow(
    *,
    resolved_layout: ResolvedLayout,
    renderer: mpl_backend_bases.RendererBase,
    pt_per_pixel: float,
) -> _SideOverflow:
    """Gap plus thickness of the colorbar that extends beyond the panels (per side); excludes colorbar ticks and labels."""
    left_pt = 0.0
    right_pt = 0.0
    bottom_pt = 0.0
    top_pt = 0.0
    for colorbar_placement in resolved_layout.colorbars:
        panel_pixel_bounding_box = _compute_panel_pixel_bounding_box(
            panels=colorbar_placement.neighbouring_panels.panels,
            renderer=renderer,
        )
        is_horizontal = colorbar_placement.side in (_Side.Left, _Side.Right)
        panel_pixel_length = (
            panel_pixel_bounding_box.height if is_horizontal else panel_pixel_bounding_box.width
        )
        colorbar_pixel_thickness = (
            colorbar_placement.length_fraction * panel_pixel_length
        ) / colorbar_placement.aspect_ratio
        overflow_pt = colorbar_placement.gap_pt + pt_per_pixel * colorbar_pixel_thickness
        if colorbar_placement.side is _Side.Left:
            left_pt = max(left_pt, overflow_pt)
        elif colorbar_placement.side is _Side.Right:
            right_pt = max(right_pt, overflow_pt)
        elif colorbar_placement.side is _Side.Bottom:
            bottom_pt = max(bottom_pt, overflow_pt)
        elif colorbar_placement.side is _Side.Top:
            top_pt = max(top_pt, overflow_pt)
        else:
            raise ValueError(
                f"unexpected side: {colorbar_placement.side!r}.",  # pyright: ignore[reportUnreachable]
            )
    return _SideOverflow(
        left_pt=left_pt,
        right_pt=right_pt,
        bottom_pt=bottom_pt,
        top_pt=top_pt,
    )


def _measure_panel_content_overflow(
    *,
    figure: mpl_figure.Figure,
    resolved_layout: ResolvedLayout,
) -> _SideOverflow:
    """
    Measure how far the labels overflow past the panels (per side, in pt).

    What is measured is text drawn at absolute pt sizes, so the result does not change
    when the panels are resized. The box of a colorbar is taken off first, since that
    part is proportional to the panel and is solved for rather than measured.
    """
    _freeze_panel_tick_locations(figure=figure)
    renderer = _get_renderer(figure=figure)
    pt_per_pixel = style_figure.PT_PER_INCH / figure.dpi
    panels = [
        panel for panel in figure.axes if all(
            colorbar_placement.colorbar_panel is not panel for colorbar_placement in resolved_layout.colorbars
        )
    ]
    panel_pixel_bounding_box = _compute_panel_pixel_bounding_box(
        panels=panels,
        renderer=renderer,
    )
    ## figure-level text is measured alongside the panels; shared labels are excluded, since
    ## their position is an output of this measurement, not an input to it
    shared_label_texts = [
        shared_label_placement.text for shared_label_placement in resolved_layout.shared_labels
    ]
    extra_artists = [
        artist for artist in (*figure.texts, *figure.legends) if artist not in shared_label_texts
    ]
    content_pixel_bounding_box = _compute_content_pixel_bounding_box(
        figure=figure,
        extra_artists=extra_artists,
        renderer=renderer,
    )
    content_overflow = _SideOverflow(
        left_pt=pt_per_pixel * (panel_pixel_bounding_box.x0 - content_pixel_bounding_box.x0),
        right_pt=pt_per_pixel * (content_pixel_bounding_box.x1 - panel_pixel_bounding_box.x1),
        bottom_pt=pt_per_pixel * (panel_pixel_bounding_box.y0 - content_pixel_bounding_box.y0),
        top_pt=pt_per_pixel * (content_pixel_bounding_box.y1 - panel_pixel_bounding_box.y1),
    )
    colorbar_overflow = _measure_colorbar_overflow(
        resolved_layout=resolved_layout,
        renderer=renderer,
        pt_per_pixel=pt_per_pixel,
    )
    return _SideOverflow(
        left_pt=content_overflow.left_pt - colorbar_overflow.left_pt,
        right_pt=content_overflow.right_pt - colorbar_overflow.right_pt,
        bottom_pt=content_overflow.bottom_pt - colorbar_overflow.bottom_pt,
        top_pt=content_overflow.top_pt - colorbar_overflow.top_pt,
    )


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class _ColorbarOverflowTerms:
    """
    How far colorbars overflow past the panels, as a linear function of `panel_width_pt`:
    `panel_coefficient * panel_width_pt + constant_pt`.

    Split this way because `panel_width_pt` is not always known yet when this is computed;
    it is what one caller solves for. Once it is known, `resolve` gives the plain value.
    """

    panel_coefficient: float
    constant_pt: float

    def resolve(
        self,
        *,
        panel_width_pt: float,
    ) -> float:
        """The overflow, now that `panel_width_pt` is known."""
        return self.constant_pt + panel_width_pt * self.panel_coefficient

    def __add__(
        self,
        other: "_ColorbarOverflowTerms",
    ) -> "_ColorbarOverflowTerms":
        """The combined overflow of two independent bands, as one still-deferred term."""
        return _ColorbarOverflowTerms(
            panel_coefficient=self.panel_coefficient + other.panel_coefficient,
            constant_pt=self.constant_pt + other.constant_pt,
        )


def _compute_colorbar_overflow(
    *,
    resolved_layout: ResolvedLayout,
    side: _Side,
) -> _ColorbarOverflowTerms:
    """
    Colorbar overflow past the panels on `side` (in pt), as `_ColorbarOverflowTerms`.

    A bar spanning several panels is a share of all of them together, so its thickness
    scales with that many panels and with the gaps between them. The deepest bar is taken
    rather than the total, since bars on the same side sit beside each other rather than
    stacking.
    """
    is_horizontal = side in (_Side.Left, _Side.Right)
    ## a bar beside the panels runs along their height, which is panel width over aspect;
    ## a bar above or below runs along their width
    span_gap_pt = resolved_layout.panel_row_gap_pt if is_horizontal else resolved_layout.panel_col_gap_pt
    panel_coefficient = 0.0
    constant_pt = 0.0
    for colorbar_placement in resolved_layout.colorbars:
        if colorbar_placement.side is not side:
            continue
        num_panels = (
            colorbar_placement.neighbouring_panels.num_rows
            if is_horizontal else colorbar_placement.neighbouring_panels.num_cols
        )
        panel_length_coefficient = 1.0 / resolved_layout.panel_aspect_ratio if is_horizontal else 1.0
        overflow_coefficient = (
            num_panels * panel_length_coefficient * colorbar_placement.length_fraction /
            colorbar_placement.aspect_ratio
        )
        overflow_constant_pt = (
            colorbar_placement.gap_pt + (num_panels - 1) * span_gap_pt * colorbar_placement.length_fraction /
            colorbar_placement.aspect_ratio
        )
        if overflow_coefficient + overflow_constant_pt > panel_coefficient + constant_pt:
            panel_coefficient = overflow_coefficient
            constant_pt = overflow_constant_pt
    return _ColorbarOverflowTerms(
        panel_coefficient=panel_coefficient,
        constant_pt=constant_pt,
    )


def _measure_shared_label_overflow(
    *,
    figure: mpl_figure.Figure,
    resolved_layout: ResolvedLayout,
) -> _SideOverflow:
    """How far the shared labels overflow past the labels each panel draws on its own (per side, in pt)."""
    renderer = _get_renderer(figure=figure)
    pt_per_pixel = style_figure.PT_PER_INCH / figure.dpi
    left_pt = 0.0
    right_pt = 0.0
    bottom_pt = 0.0
    top_pt = 0.0
    for shared_label_placement in resolved_layout.shared_labels:
        text_pixel_bounding_box = shared_label_placement.text.get_window_extent(renderer)
        is_horizontal = shared_label_placement.side in (_Side.Left, _Side.Right)
        text_pixel_thickness = text_pixel_bounding_box.width if is_horizontal else text_pixel_bounding_box.height
        text_thickness_pt = pt_per_pixel * text_pixel_thickness
        overflow_pt = shared_label_placement.gap_pt + text_thickness_pt
        if shared_label_placement.side is _Side.Left:
            left_pt = max(left_pt, overflow_pt)
        elif shared_label_placement.side is _Side.Right:
            right_pt = max(right_pt, overflow_pt)
        elif shared_label_placement.side is _Side.Bottom:
            bottom_pt = max(bottom_pt, overflow_pt)
        elif shared_label_placement.side is _Side.Top:
            top_pt = max(top_pt, overflow_pt)
        else:
            raise ValueError(
                f"unexpected side: {shared_label_placement.side!r}.",  # pyright: ignore[reportUnreachable]
            )
    return _SideOverflow(
        left_pt=left_pt,
        right_pt=right_pt,
        bottom_pt=bottom_pt,
        top_pt=top_pt,
    )


def _place_shared_labels(
    *,
    figure: mpl_figure.Figure,
    resolved_layout: ResolvedLayout,
    panel_overflow: _SideOverflow,
) -> None:
    """Put each shared label just outside the labels each panel draws on its own, centred on the panels."""
    renderer = _get_renderer(figure=figure)
    figure_size = FigureSize(figure=figure)
    figure_width_pt = figure_size.width_pt
    figure_height_pt = figure_size.height_pt
    pt_per_pixel = style_figure.PT_PER_INCH / figure.dpi
    for shared_label_placement in resolved_layout.shared_labels:
        is_horizontal = shared_label_placement.side in (_Side.Left, _Side.Right)
        panel_pixel_bounding_box = _compute_panel_pixel_bounding_box(
            panels=shared_label_placement.panels,
            renderer=renderer,
        )
        ## converted once, rather than coordinate by coordinate as each is used below
        panel_bounding_box_pt = mpl_transforms.Bbox(pt_per_pixel * panel_pixel_bounding_box.get_points())
        text_pixel_bounding_box = shared_label_placement.text.get_window_extent(renderer)
        text_pixel_thickness = text_pixel_bounding_box.width if is_horizontal else text_pixel_bounding_box.height
        text_thickness_pt = pt_per_pixel * text_pixel_thickness
        panel_overflow_pt = panel_overflow.value_for_side(side=shared_label_placement.side)
        offset_pt = panel_overflow_pt + shared_label_placement.gap_pt + 0.5 * text_thickness_pt
        offset_sign = -1.0 if shared_label_placement.side in (_Side.Left, _Side.Bottom) else 1.0
        if shared_label_placement.side is _Side.Left:
            edge_pt = panel_bounding_box_pt.x0
        elif shared_label_placement.side is _Side.Right:
            edge_pt = panel_bounding_box_pt.x1
        elif shared_label_placement.side is _Side.Bottom:
            edge_pt = panel_bounding_box_pt.y0
        elif shared_label_placement.side is _Side.Top:
            edge_pt = panel_bounding_box_pt.y1
        else:
            raise ValueError(
                f"unexpected side: {shared_label_placement.side!r}.",  # pyright: ignore[reportUnreachable]
            )
        if is_horizontal:
            x_position = (edge_pt + offset_sign * offset_pt) / figure_width_pt
            y_position = 0.5 * (panel_bounding_box_pt.y0 + panel_bounding_box_pt.y1) / figure_height_pt
        else:
            x_position = 0.5 * (panel_bounding_box_pt.x0 + panel_bounding_box_pt.x1) / figure_width_pt
            y_position = (edge_pt + offset_sign * offset_pt) / figure_height_pt
        shared_label_placement.text.set_position((x_position, y_position))


@dataclasses.dataclass(frozen=True)
class PanelBounds:
    """Bounding box for a panel in figure coordinates."""

    x_min_fraction: float
    y_min_fraction: float
    x_width_fraction: float
    y_width_fraction: float


def compute_panel_fractional_bounds(
    *,
    neighbouring_panels: PanelLike,
    side: _Side = box_positions.Positions.Side.Right,
    gap_fraction: float = 0.1,
    length_fraction: float = 1.0,
    thickness_fraction: float = 1.0,
) -> PanelBounds:
    """
    Compute figure bounds for a panel placed neighbouring `neighbouring_panels`.

    The new panel sits on the `side` of `neighbouring_panels`, offset by `gap_fraction`
    (in figure coordinates). `length_fraction` sets its extent parallel to `side`,
    centered on that edge. `thickness_fraction` sets its extent perpendicular to `side`,
    also as a fraction of the corresponding dimension of what it neighbours. Given
    several panels it neighbours all of them, which is how one colorbar comes to
    describe a whole grid.
    """
    validate_types.ensure_finite_float(
        param=length_fraction,
        param_name="length_fraction",
        allow_none=False,
        require_positive=True,
        allow_zero=False,
    )
    validate_types.ensure_finite_float(
        param=thickness_fraction,
        param_name="thickness_fraction",
        allow_none=False,
        require_positive=True,
        allow_zero=False,
    )
    ## a panel may sit flush against the one it neighbours, so no gap is a gap of zero
    validate_types.ensure_finite_float(
        param=gap_fraction,
        param_name="gap_fraction",
        allow_none=False,
        require_positive=True,
        allow_zero=True,
    )
    neighbouring_fractional_bounding_box = _compute_panel_fractional_bounding_box(
        panels=as_panel_list(panels=neighbouring_panels),
    )
    neighbouring_x0_fraction = neighbouring_fractional_bounding_box.x0
    neighbouring_x1_fraction = neighbouring_fractional_bounding_box.x1
    neighbouring_y0_fraction = neighbouring_fractional_bounding_box.y0
    neighbouring_y1_fraction = neighbouring_fractional_bounding_box.y1
    neighbouring_width_fraction = neighbouring_fractional_bounding_box.width
    neighbouring_height_fraction = neighbouring_fractional_bounding_box.height
    if side in (_Side.Left, _Side.Right):
        panel_x_width_fraction = thickness_fraction * neighbouring_width_fraction
        panel_y_width_fraction = length_fraction * neighbouring_height_fraction
        panel_y_min_fraction = (
            neighbouring_y0_fraction + 0.5 * (neighbouring_height_fraction - panel_y_width_fraction)
        )
        if side == _Side.Right:
            panel_x_min_fraction = neighbouring_x1_fraction + gap_fraction
        else:
            panel_x_min_fraction = neighbouring_x0_fraction - panel_x_width_fraction - gap_fraction
    elif side in (_Side.Top, _Side.Bottom):
        panel_x_width_fraction = length_fraction * neighbouring_width_fraction
        panel_y_width_fraction = thickness_fraction * neighbouring_height_fraction
        panel_x_min_fraction = (
            neighbouring_x0_fraction + 0.5 * (neighbouring_width_fraction - panel_x_width_fraction)
        )
        if side == _Side.Top:
            panel_y_min_fraction = neighbouring_y1_fraction + gap_fraction
        else:
            panel_y_min_fraction = neighbouring_y0_fraction - panel_y_width_fraction - gap_fraction
    else:
        raise ValueError(f"unexpected side: {side!r}.")  # pyright: ignore[reportUnreachable]
    return PanelBounds(
        x_min_fraction=panel_x_min_fraction,
        y_min_fraction=panel_y_min_fraction,
        x_width_fraction=panel_x_width_fraction,
        y_width_fraction=panel_y_width_fraction,
    )


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class _SolvedWidths:
    """The panel and figure width, once whichever one was not pinned is solved for."""

    panel_width_pt: float
    figure_width_pt: float


def _solve_widths(
    *,
    figure_width_pt: float,
    resolved_layout: ResolvedLayout,
    side_overflow: _SideOverflow,
    padding: style_figure.FigurePadding,
) -> _SolvedWidths:
    """
    Solve for whichever of the panel width and figure width was not pinned.

    The page fixes the width, so the panels are what give horizontally: `figure_width_pt`
    is what the figure currently measures, and stays fixed unless the panel width is pinned
    instead, in which case the figure grows around it.
    """
    colorbar_overflow_left = _compute_colorbar_overflow(
        resolved_layout=resolved_layout,
        side=_Side.Left,
    )
    colorbar_overflow_right = _compute_colorbar_overflow(
        resolved_layout=resolved_layout,
        side=_Side.Right,
    )
    ## a bar on the left and a bar on the right each eat into the figure independently, so
    ## their overflow adds rather than taking whichever is deeper
    colorbar_overflow_horizontal = colorbar_overflow_left + colorbar_overflow_right
    col_gaps_pt = (resolved_layout.num_panel_cols - 1) * resolved_layout.panel_col_gap_pt
    horizontal_constants_pt = (
        padding.left_pt + padding.right_pt + side_overflow.left_pt + side_overflow.right_pt + col_gaps_pt +
        colorbar_overflow_horizontal.constant_pt
    )
    if resolved_layout.panel_width_pt is not None:
        ## the panel is pinned, so the figure grows around it
        panel_width_pt = resolved_layout.panel_width_pt
        figure_width_pt = (
            horizontal_constants_pt + panel_width_pt *
            (resolved_layout.num_panel_cols + colorbar_overflow_horizontal.panel_coefficient)
        )
    else:
        ## the figure is pinned by the page, so the panels take what its labels leave
        panel_width_pt = (figure_width_pt - horizontal_constants_pt) / (
            resolved_layout.num_panel_cols + colorbar_overflow_horizontal.panel_coefficient
        )
        if panel_width_pt <= 0:
            raise ValueError(
                f"the labels and padding need {horizontal_constants_pt:.1f} pt of a figure only"
                f" {figure_width_pt:.1f} pt wide, so there is no room left for the panels.",
            )
    return _SolvedWidths(
        panel_width_pt=panel_width_pt,
        figure_width_pt=figure_width_pt,
    )


def _solve_figure_height(
    *,
    panel_width_pt: float,
    resolved_layout: ResolvedLayout,
    side_overflow: _SideOverflow,
    padding: style_figure.FigurePadding,
) -> float:
    """
    The figure height that fits the panels and everything beyond them (in pt).

    Nothing fixes the height, so it is free to follow from the panel width that was
    already solved for.
    """
    panel_height_pt = panel_width_pt / resolved_layout.panel_aspect_ratio
    colorbar_overflow_bottom = _compute_colorbar_overflow(
        resolved_layout=resolved_layout,
        side=_Side.Bottom,
    )
    colorbar_overflow_top = _compute_colorbar_overflow(
        resolved_layout=resolved_layout,
        side=_Side.Top,
    )
    ## a bar on the bottom and a bar on the top each eat into the figure independently, so
    ## their overflow adds rather than taking whichever is deeper
    colorbar_overflow_vertical = colorbar_overflow_bottom + colorbar_overflow_top
    row_gaps_pt = (resolved_layout.num_panel_rows - 1) * resolved_layout.panel_row_gap_pt
    panel_rows_height_pt = panel_height_pt * resolved_layout.num_panel_rows
    ## a bar above or below is as long as the panels are wide, so its thickness scales with
    ## the panel width that was just solved for, not with the panel height it eats into
    colorbar_overflow_vertical_pt = colorbar_overflow_vertical.resolve(panel_width_pt=panel_width_pt)
    return (
        padding.bottom_pt + padding.top_pt + side_overflow.bottom_pt + side_overflow.top_pt + row_gaps_pt +
        panel_rows_height_pt + colorbar_overflow_vertical_pt
    )


def _compute_figure_margins(
    *,
    resolved_layout: ResolvedLayout,
    side_overflow: _SideOverflow,
    padding: style_figure.FigurePadding,
    panel_width_pt: float,
) -> FigureMargins:
    """Margins that hold the labels, the padding, and any colorbar sitting outside the panels."""
    colorbar_overflow_left = _compute_colorbar_overflow(
        resolved_layout=resolved_layout,
        side=_Side.Left,
    )
    colorbar_overflow_right = _compute_colorbar_overflow(
        resolved_layout=resolved_layout,
        side=_Side.Right,
    )
    colorbar_overflow_bottom = _compute_colorbar_overflow(
        resolved_layout=resolved_layout,
        side=_Side.Bottom,
    )
    colorbar_overflow_top = _compute_colorbar_overflow(
        resolved_layout=resolved_layout,
        side=_Side.Top,
    )
    colorbar_overflow_left_pt = colorbar_overflow_left.resolve(panel_width_pt=panel_width_pt)
    colorbar_overflow_right_pt = colorbar_overflow_right.resolve(panel_width_pt=panel_width_pt)
    colorbar_overflow_bottom_pt = colorbar_overflow_bottom.resolve(panel_width_pt=panel_width_pt)
    colorbar_overflow_top_pt = colorbar_overflow_top.resolve(panel_width_pt=panel_width_pt)
    return FigureMargins(
        left_pt=padding.left_pt + side_overflow.left_pt + colorbar_overflow_left_pt,
        right_pt=padding.right_pt + side_overflow.right_pt + colorbar_overflow_right_pt,
        bottom_pt=padding.bottom_pt + side_overflow.bottom_pt + colorbar_overflow_bottom_pt,
        top_pt=padding.top_pt + side_overflow.top_pt + colorbar_overflow_top_pt,
    )


def _reposition_colorbars(
    *,
    figure: mpl_figure.Figure,
    resolved_layout: ResolvedLayout,
    figure_width_pt: float,
    figure_height_pt: float,
) -> None:
    """Place every colorbar against where its neighbouring panels now sit."""
    for colorbar_placement in resolved_layout.colorbars:
        is_horizontal = colorbar_placement.side in (_Side.Left, _Side.Right)
        figure_length_pt = figure_width_pt if is_horizontal else figure_height_pt
        panel_bounds = compute_panel_fractional_bounds(
            neighbouring_panels=colorbar_placement.neighbouring_panels.panels,
            side=colorbar_placement.side,
            gap_fraction=colorbar_placement.gap_pt / figure_length_pt,
            length_fraction=colorbar_placement.length_fraction,
            thickness_fraction=compute_colorbar_thickness_fraction(
                neighbouring_panels=colorbar_placement.neighbouring_panels.panels,
                side=colorbar_placement.side,
                length_fraction=colorbar_placement.length_fraction,
                aspect_ratio=colorbar_placement.aspect_ratio,
            ),
        )
        colorbar_placement.colorbar_panel.set_position(
            (
                panel_bounds.x_min_fraction,
                panel_bounds.y_min_fraction,
                panel_bounds.x_width_fraction,
                panel_bounds.y_width_fraction,
            ),
        )


def fit_figure_to_content(
    *,
    figure: mpl_figure.Figure,
) -> None:
    """
    Resize `figure` so its labels sit inside it, with the padding the style asks for.

    Does nothing to a figure that was not built to be fitted.
    """
    resolved_layout = RESOLVED_LAYOUTS.get(figure)
    if resolved_layout is None:
        return
    padding = resolved_layout.figure_padding
    panel_overflow = _measure_panel_content_overflow(
        figure=figure,
        resolved_layout=resolved_layout,
    )
    ## a shared label sits beyond the labels each panel draws on its own, so it adds to what
    ## each side holds
    shared_label_overflow = _measure_shared_label_overflow(
        figure=figure,
        resolved_layout=resolved_layout,
    )
    side_overflow = _SideOverflow(
        left_pt=panel_overflow.left_pt + shared_label_overflow.left_pt,
        right_pt=panel_overflow.right_pt + shared_label_overflow.right_pt,
        bottom_pt=panel_overflow.bottom_pt + shared_label_overflow.bottom_pt,
        top_pt=panel_overflow.top_pt + shared_label_overflow.top_pt,
    )
    solved_widths = _solve_widths(
        figure_width_pt=FigureSize(figure=figure).width_pt,
        resolved_layout=resolved_layout,
        side_overflow=side_overflow,
        padding=padding,
    )
    panel_width_pt = solved_widths.panel_width_pt
    figure_width_pt = solved_widths.figure_width_pt
    figure_height_pt = _solve_figure_height(
        panel_width_pt=panel_width_pt,
        resolved_layout=resolved_layout,
        side_overflow=side_overflow,
        padding=padding,
    )
    figure_margins = _compute_figure_margins(
        resolved_layout=resolved_layout,
        side_overflow=side_overflow,
        padding=padding,
        panel_width_pt=panel_width_pt,
    )
    figure_shape = BoxShape(
        width_cm=figure_width_pt / style_figure.PT_PER_CM,
        height_cm=figure_height_pt / style_figure.PT_PER_CM,
    )
    figure.set_size_inches(figure_shape.as_mpl_shape)
    set_figure_margins(
        figure=figure,
        figure_shape=figure_shape,
        figure_margins=figure_margins,
    )
    set_panel_gaps(
        figure=figure,
        figure_shape=figure_shape,
        figure_margins=figure_margins,
        num_panel_rows=resolved_layout.num_panel_rows,
        num_panel_cols=resolved_layout.num_panel_cols,
        panel_row_gap_pt=resolved_layout.panel_row_gap_pt,
        panel_col_gap_pt=resolved_layout.panel_col_gap_pt,
    )
    ## the panels have moved, so each shared label is placed again just outside their own
    _place_shared_labels(
        figure=figure,
        resolved_layout=resolved_layout,
        panel_overflow=panel_overflow,
    )
    ## the panels have moved, so every colorbar is placed again against where they are now
    _reposition_colorbars(
        figure=figure,
        resolved_layout=resolved_layout,
        figure_width_pt=figure_width_pt,
        figure_height_pt=figure_height_pt,
    )


## } MODULE
