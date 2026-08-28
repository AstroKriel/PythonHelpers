## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import dataclasses
import enum
import math
import shutil

from collections import abc as collections_abc

## third-party
import cycler
import matplotlib

## local
from jormi.ww_io import manage_log
from jormi.ww_types import box_positions
from jormi.ww_validation import validate_box_positions, validate_types

##
## === UNITS
##

## figure sizes are given in cm and text sizes in pt, the two units a page is specified
## in; inches appear only where Matplotlib insists on them
CM_PER_INCH: float = 2.54
PT_PER_INCH: float = 72.0
PT_PER_CM: float = PT_PER_INCH / CM_PER_INCH

## sampling density for a saved raster, per cm of figure size; 250 is roughly 635 dpi,
## which is what line art wants in print
DEFAULT_PIXELS_PER_CM: float = 250.0

##
## === COLOR THEMES
##


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class ThemeParams:
    """
    Theme colours, named for what they colour rather than their rcParams key; `color_sequence`
    is the list successive artists cycle through when given no colour of their own.
    """

    background_color: str
    foreground_color: str
    tick_color: str
    grid_color: str
    grid_alpha: float
    color_sequence: tuple[str, ...]

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_in_bounds(
            param=self.grid_alpha,
            param_name="grid_alpha",
            allow_none=False,
            min_value=0.0,
            max_value=1.0,
        )
        validate_types.ensure_tuple_of_strings(
            param=self.color_sequence,
            param_name="color_sequence",
            allow_none=False,
        )
        if not self.color_sequence:
            raise ValueError(
                "`color_sequence` must name at least one colour,"
                " since it is what an artist given no colour of its own falls back to.",
            )

    def as_rc_params(
        self,
    ) -> dict[str, object]:
        """Map each colour onto the Matplotlib rcParams that consume it."""
        return {
            "figure.facecolor": self.background_color,
            "axes.facecolor": self.background_color,
            "savefig.facecolor": self.background_color,
            "figure.edgecolor": self.background_color,
            "patch.edgecolor": self.foreground_color,
            "lines.color": self.foreground_color,
            "axes.edgecolor": self.foreground_color,
            "axes.labelcolor": self.foreground_color,
            "text.color": self.foreground_color,
            "axes.titlecolor": self.foreground_color,
            "xtick.color": self.tick_color,
            "ytick.color": self.tick_color,
            "grid.color": self.grid_color,
            "grid.alpha": self.grid_alpha,
            "axes.prop_cycle": cycler.cycler(color=list(self.color_sequence)),
        }


LIGHT_THEME_PARAMS = ThemeParams(
    background_color="white",
    foreground_color="#222222",
    tick_color="#333333",
    grid_color="#dddddd",
    grid_alpha=0.6,
    color_sequence=(
        "#1f77b4",
        "#2ca02c",
        "#d62728",
        "#ff7f0e",
        "#9467bd",
        "#17becf",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
    ),
)

DARK_THEME_PARAMS = ThemeParams(
    background_color="#0b0b0e",
    foreground_color="#e6e6e6",
    tick_color="#cfcfd2",
    grid_color="#2e2e35",
    grid_alpha=0.3,
    color_sequence=(
        "#7aa2f7",
        "#9ece6a",
        "#f7768e",
        "#e0af68",
        "#bb9af7",
        "#7dcfff",
        "#f6bd60",
        "#c0caf5",
        "#89ddff",
        "#ff9e64",
    ),
)


class Theme(enum.Enum):
    """Available Matplotlib color themes."""

    LIGHT = "light"
    DARK = "dark"


THEMES: collections_abc.Mapping[Theme, ThemeParams] = {
    Theme.LIGHT: LIGHT_THEME_PARAMS,
    Theme.DARK: DARK_THEME_PARAMS,
}

##
## === TYPESETTING
##


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class LatexParams:
    """
    How the text of a figure is typeset, when LaTeX sets it.

    `use_tex` routes every string through LaTeX rather than the mathtext Matplotlib
    provides on its own, so a figure is set in the same face as the document it is bound
    for. Plain strings then come out upright and `$...$` italic, which is why words read
    heavier than maths.

    `font_package` names the face the document loads; `font_family` picks which shape of
    that face text is set in. `math_packages` are bare package names the labels need, so
    a package that takes options does not belong here.
    """

    use_tex: bool = True
    font_package: str = "lmodern"
    font_family: str = "serif"
    math_packages: tuple[str, ...] = ("amsmath", )

    def _get_packages(
        self,
    ) -> str:
        """The `\\usepackage` lines, the face first so the maths is set to match it."""
        return "\n".join(
            f"\\usepackage{{{package_name}}}" for package_name in (self.font_package, *self.math_packages)
            if package_name
        )

    def as_rc_params(
        self,
    ) -> dict[str, object]:
        """
        Map the typesetting onto the Matplotlib rcParams that consume it.

        LaTeX is only requested when installed; otherwise the figure falls back to the
        mathtext Matplotlib provides, rather than not drawing text at all. The face is
        set regardless, since that shapes the text rather than typesets it.
        """
        available_and_requested = self.use_tex and (shutil.which("latex") is not None)
        return {
            "font.family": self.font_family,
            "text.usetex": available_and_requested,
            "text.latex.preamble": self._get_packages() if available_and_requested else "",
        }


##
## === FONT SIZES
##


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class TextSizeParams:
    """
    Point sizes for each kind of text that appears on a figure.

    A single scale sets every size: `largest_size_pt` is the biggest text on the figure, and
    each kind of text sits some number of steps down from it. Each kind is named by what
    it does rather than by how big it is, so moving one leaves the others where they are.

    `size_ratio` sets how far apart the steps are: raise it for a stronger hierarchy,
    lower it for a flatter one. Each `*_level` is how many steps down that kind of text
    sits, and levels need not be whole steps.
    """

    largest_size_pt: float = 12.0
    size_ratio: float = 1.16
    axis_label_level: float = 0.0
    tick_label_level: float = 2.0
    annotation_level: float = 2.0
    legend_level: float = 1.0

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_finite_scalar(
            param=self.largest_size_pt,
            param_name="largest_size_pt",
            allow_none=False,
            require_positive=True,
            allow_zero=False,
        )
        validate_types.ensure_finite_scalar(
            param=self.size_ratio,
            param_name="size_ratio",
            allow_none=False,
        )
        if not (self.size_ratio > 1):
            raise ValueError(
                f"`size_ratio` must be greater than one, but got {self.size_ratio}."
                " A ratio of one or less would step text up rather than down.",
            )
        for param_name in (
                "axis_label_level",
                "tick_label_level",
                "annotation_level",
                "legend_level",
        ):
            validate_types.ensure_finite_scalar(
                param=getattr(self, param_name),
                param_name=param_name,
                allow_none=False,
                require_positive=True,
                allow_zero=True,
            )

    @property
    def axis_label_size_pt(
        self,
    ) -> float:
        return self.compute_size_at_level(level=self.axis_label_level)

    @property
    def tick_label_size_pt(
        self,
    ) -> float:
        return self.compute_size_at_level(level=self.tick_label_level)

    @property
    def annotation_size_pt(
        self,
    ) -> float:
        return self.compute_size_at_level(level=self.annotation_level)

    @property
    def legend_size_pt(
        self,
    ) -> float:
        return self.compute_size_at_level(level=self.legend_level)

    def compute_size_at_level(
        self,
        *,
        level: float,
    ) -> float:
        """
        Point size of text sitting `level` steps below `largest_size`.

        Levels need not be whole steps, so a one-off label that wants to sit between two
        of the named kinds of text can ask for the size in between.
        """
        validate_types.ensure_finite_scalar(
            param=level,
            param_name="level",
            allow_none=False,
            require_positive=True,
            allow_zero=True,
        )
        return self.largest_size_pt * self.size_ratio**(-level)

    def compute_level_at_size(
        self,
        *,
        text_size_pt: float,
    ) -> float:
        """
        Level that `text_size_pt` sits at, inverting `compute_size_at_level`.

        Sizes are set by level rather than in pt, so asking for a size a little under one of
        the named kinds of text means finding the level it falls at.
        """
        validate_types.ensure_finite_scalar(
            param=text_size_pt,
            param_name="text_size_pt",
            allow_none=False,
            require_positive=True,
            allow_zero=False,
        )
        if text_size_pt > self.largest_size_pt:
            raise ValueError(
                f"`text_size_pt` ({text_size_pt}) is larger than `largest_size_pt` ({self.largest_size_pt}),"
                " and no text may be larger than that.",
            )
        return math.log(self.largest_size_pt / text_size_pt) / math.log(self.size_ratio)

    def as_rc_params(
        self,
    ) -> dict[str, object]:
        """Map each kind of text onto the Matplotlib rcParams that consume it."""
        return {
            "font.size": self.axis_label_size_pt,
            "axes.labelsize": self.axis_label_size_pt,
            ## figure-level x/y labels shared across a grid of axes
            "figure.labelsize": self.axis_label_size_pt,
            ## titles are not used, but a stray one should match the labels rather than
            ## fall back to the Matplotlib default of larger-than-everything
            "axes.titlesize": self.axis_label_size_pt,
            "figure.titlesize": self.axis_label_size_pt,
            "xtick.labelsize": self.tick_label_size_pt,
            "ytick.labelsize": self.tick_label_size_pt,
            "legend.fontsize": self.legend_size_pt,
        }


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class ArtistParams:
    """
    The marks that draw the data, in pt: lines, markers, and marker edges.

    Narrower than what Matplotlib calls an artist, which is anything drawable; text,
    the panel frame, and the legend are each styled by their own group.

    These are chosen for the medium a figure is bound for rather than derived from its
    size, so a wider figure keeps the same stroke weights: a pt stays a pt.
    """

    line_width_pt: float = 0.9
    marker_size_pt: float = 5.0
    marker_edge_width_pt: float = 0.6

    def __post_init__(
        self,
    ) -> None:
        ## a mark is turned off by giving it no width, so zero is allowed and only a negative is not
        for param_name in (
                "line_width_pt",
                "marker_size_pt",
                "marker_edge_width_pt",
        ):
            validate_types.ensure_finite_scalar(
                param=getattr(self, param_name),
                param_name=param_name,
                allow_none=False,
                require_positive=True,
                allow_zero=True,
            )

    def as_rc_params(
        self,
    ) -> dict[str, object]:
        """Map each mark onto the Matplotlib rcParams that consume it."""
        return {
            "lines.linewidth": self.line_width_pt,
            "lines.markersize": self.marker_size_pt,
            ## an edge holds a marker rather than being it, so it is drawn lighter than the
            ## data line; left unset Matplotlib would outweigh that line instead
            "lines.markeredgewidth": self.marker_edge_width_pt,
        }


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class FrameParams:
    """
    Panel furniture, in pt: frame, ticks, and the gaps around their labels.

    `line_width_pt` covers the frame, ticks, and any text box; none of them are the data.

    Gaps chain instead of sharing an edge: frame -> `tick_label_gap_pt` -> tick labels ->
    `axis_label_gap_pt` -> axis label. A longer tick label pushes the axis label out on
    its own. `ticks_point_inward` takes tick length from the panel, not the margin.
    """

    line_width_pt: float = 0.4
    major_tick_length_pt: float = 3.0
    minor_tick_length_pt: float = 1.6
    tick_label_gap_pt: float = 2.5
    axis_label_gap_pt: float = 3.0
    ticks_point_inward: bool = True
    ticks_on_all_sides: bool = True
    show_minor_ticks: bool = True

    def __post_init__(
        self,
    ) -> None:
        ## a frame with no ticks asks for a length of zero, so only a negative is refused
        for param_name in (
                "line_width_pt",
                "major_tick_length_pt",
                "minor_tick_length_pt",
                "tick_label_gap_pt",
                "axis_label_gap_pt",
        ):
            validate_types.ensure_finite_scalar(
                param=getattr(self, param_name),
                param_name=param_name,
                allow_none=False,
                require_positive=True,
                allow_zero=True,
            )

    def as_rc_params(
        self,
    ) -> dict[str, object]:
        """Map each part of the frame onto the Matplotlib rcParams that consume it."""
        tick_direction = "in" if self.ticks_point_inward else "out"
        return {
            "axes.linewidth": self.line_width_pt,
            ## a box drawn around text is furniture too, and the Matplotlib default of 1.0
            ## would outweigh both the frame and the data
            "patch.linewidth": self.line_width_pt,
            "xtick.major.width": self.line_width_pt,
            "ytick.major.width": self.line_width_pt,
            "xtick.minor.width": self.line_width_pt,
            "ytick.minor.width": self.line_width_pt,
            "xtick.major.size": self.major_tick_length_pt,
            "ytick.major.size": self.major_tick_length_pt,
            "xtick.minor.size": self.minor_tick_length_pt,
            "ytick.minor.size": self.minor_tick_length_pt,
            "xtick.major.pad": self.tick_label_gap_pt,
            "ytick.major.pad": self.tick_label_gap_pt,
            "xtick.minor.pad": self.tick_label_gap_pt,
            "ytick.minor.pad": self.tick_label_gap_pt,
            ## measured against the frame rather than the text, so it sits here
            "axes.labelpad": self.axis_label_gap_pt,
            "xtick.direction": tick_direction,
            "ytick.direction": tick_direction,
            "xtick.top": self.ticks_on_all_sides,
            "ytick.right": self.ticks_on_all_sides,
            "xtick.minor.visible": self.show_minor_ticks,
            "ytick.minor.visible": self.show_minor_ticks,
        }


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class LegendParams:
    """
    Where a legend sits and how tightly it is packed.

    The four spacings are each measured in em, a multiple of the text size the legend
    uses, as Matplotlib measures them, so they hold their proportions as the text size
    changes.
    """

    entry_row_gap_em: float = 0.2
    entry_col_gap_em: float = 2.0
    entry_text_gap_em: float = 0.8
    frame_margin_em: float = 0.4
    position: box_positions.Positions.PositionLike = box_positions.Positions.Corner.BottomRight
    show_frame: bool = False

    def __post_init__(
        self,
    ) -> None:
        ## a legend packed hard against its anchor asks for gaps of zero, so only a negative is refused
        for param_name in (
                "entry_row_gap_em",
                "entry_col_gap_em",
                "entry_text_gap_em",
                "frame_margin_em",
        ):
            validate_types.ensure_finite_scalar(
                param=getattr(self, param_name),
                param_name=param_name,
                allow_none=False,
                require_positive=True,
                allow_zero=True,
            )
        validate_box_positions.ensure_mpl_anchor(
            position=self.position,
            param_name="position",
        )

    def as_rc_params(
        self,
    ) -> dict[str, object]:
        """Map the legend style onto the Matplotlib rcParams that consume it."""
        return {
            "legend.labelspacing": self.entry_row_gap_em,
            "legend.columnspacing": self.entry_col_gap_em,
            "legend.handletextpad": self.entry_text_gap_em,
            "legend.borderpad": self.frame_margin_em,
            "legend.loc": validate_box_positions.as_mpl_anchor(position=self.position).value,
            "legend.frameon": self.show_frame,
        }


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class SaveParams:
    """
    How a figure is written to file.

    `crop_to_ink` is off by default, and deliberately: cropping makes the saved file a
    different size from the figure that was asked for, which is what page anchoring
    exists to prevent.
    """

    pixels_per_cm: float = DEFAULT_PIXELS_PER_CM
    crop_to_ink: bool = False
    ink_margin_cm: float = 0.0
    transparent_background: bool = False

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_finite_scalar(
            param=self.pixels_per_cm,
            param_name="pixels_per_cm",
            allow_none=False,
            require_positive=True,
            allow_zero=False,
        )
        validate_types.ensure_finite_scalar(
            param=self.ink_margin_cm,
            param_name="ink_margin_cm",
            allow_none=False,
            require_positive=True,
            allow_zero=True,
        )
        if self.crop_to_ink:
            manage_log.log_warning(
                text=(
                    "`crop_to_ink` makes the saved file the size of its ink, not the size the"
                    " figure was asked for, so its text will no longer sit at the pt size the"
                    " page expects. Widen the margins instead of cropping."
                ),
            )

    def as_rc_params(
        self,
    ) -> dict[str, object]:
        """Map the save style onto the Matplotlib rcParams that consume it."""
        return {
            "savefig.dpi": self.pixels_per_cm * CM_PER_INCH,
            "savefig.bbox": "tight" if self.crop_to_ink else None,
            "savefig.pad_inches": self.ink_margin_cm / CM_PER_INCH,
            "savefig.transparent": self.transparent_background,
        }


##
## === FIGURE LAYOUT
##


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class FigurePadding:
    """
    Space left clear outside everything a figure draws, in pt.

    The margins of a figure measure how much room the labels need; this says how much to
    leave beyond them. The extent of a label is not knowable before it is drawn, so a
    figure fitted to its contents measures the labels and is given only this one constant.
    """

    left_pt: float = 6.0
    right_pt: float = 6.0
    bottom_pt: float = 6.0
    top_pt: float = 6.0

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


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class FigureWidth:
    """
    How wide a figure is drawn, so that its text is sized for the page.

    Drawing a figure at the width it is printed at means a pt of text is a pt on the page,
    rather than being scaled by however much the document resizes the figure. How that
    width is shared between panels depends on the grid, so it is decided when the figure
    is built rather than here.
    """

    ## the widest a figure may be drawn: the full text width of the document the figures
    ## are bound for, which `width_fraction` then takes a share of
    max_width_cm: float = 16.0
    width_fraction: float = 1.0

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_finite_scalar(
            param=self.max_width_cm,
            param_name="max_width_cm",
            allow_none=False,
            require_positive=True,
            allow_zero=False,
        )
        validate_types.ensure_finite_scalar(
            param=self.width_fraction,
            param_name="width_fraction",
            allow_none=False,
            require_positive=True,
            allow_zero=False,
        )
        if self.width_fraction > 1.0:
            raise ValueError(
                f"`width_fraction` must lie in (0, 1], but got {self.width_fraction}."
                " A figure cannot be drawn wider than the page it sits on.",
            )

    @property
    def width_cm(
        self,
    ) -> float:
        """Width the figure is drawn at, being its share of the widest it may be."""
        return self.width_fraction * self.max_width_cm


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class PanelGaps:
    """
    Space between one pair of neighbouring panels, in pt (1 pt = 1/72 inch).

    In pt for the same reason the margins are: a gap holds the tick and axis labels of
    the panel beside it, and those are measured in pt.
    """

    row_pt: float = 10.0
    col_pt: float = 10.0

    def __post_init__(
        self,
    ) -> None:
        for param_name in (
                "row_pt",
                "col_pt",
        ):
            validate_types.ensure_finite_scalar(
                param=getattr(self, param_name),
                param_name=param_name,
                allow_none=False,
                require_positive=True,
                allow_zero=True,
            )


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class FigureLayout:
    """
    How much of the page a figure takes, and how much of that is left clear around it.

    All three are decisions about where a figure sits on a page, so they travel together.
    How tall a figure is, and how many axes it holds, are decided per figure instead.

    Room for the labels is not among them: how much they need is not knowable until they
    are drawn, so a figure is measured and its margins derived when it is saved.
    """

    figure_width: FigureWidth = FigureWidth()
    figure_padding: FigurePadding = FigurePadding()
    panel_gaps: PanelGaps = PanelGaps()


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class ColorbarLayout:
    """
    Where a colorbar sits relative to the panel it describes.

    A colorbar is placed as a panel neighbouring its own, so the space between the two is
    a panel gap: `gap.row_pt` above or below, `gap.col_pt` on the left or right. Left
    unset, it defaults to the gap the figure already spaces its panels by.

    `aspect_ratio` is bar length over thickness, so a bar keeps its proportions whatever
    it describes, rather than thickening as a fixed share of a larger panel group.
    """

    gap: PanelGaps | None = None
    aspect_ratio: float = 15.0

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_finite_scalar(
            param=self.aspect_ratio,
            param_name="aspect_ratio",
            allow_none=False,
            require_positive=True,
            allow_zero=False,
        )


## a figure spanning the full text width, and one spanning half of it, which is a single
## column of a two-column page
FULL_PAGE_FIGURE_LAYOUT = FigureLayout(figure_width=FigureWidth(width_fraction=1.0))
HALF_PAGE_FIGURE_LAYOUT = FigureLayout(figure_width=FigureWidth(width_fraction=0.5))

##
## === HELPERS
##


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class FigureParams:
    """
    Every choice that styles a figure, gathered so one value describes the whole style.

    The groups are listed in three tiers, and defined above in the same order:

        1. the theme and the typesetting, which colour and set everything else
        2. the five that each know the rcParams they produce
        3. the two layouts, which jormi reads itself rather than handing to Matplotlib
    """

    theme: Theme = Theme.LIGHT
    latex_params: LatexParams = LatexParams()
    text_size_params: TextSizeParams = TextSizeParams()
    artist_params: ArtistParams = ArtistParams()
    frame_params: FrameParams = FrameParams()
    legend_params: LegendParams = LegendParams()
    save_params: SaveParams = SaveParams()
    figure_layout: FigureLayout = FULL_PAGE_FIGURE_LAYOUT
    colorbar_layout: ColorbarLayout = ColorbarLayout()

    @property
    def theme_params(
        self,
    ) -> ThemeParams:
        """The colours the chosen theme sets, for what jormi draws itself."""
        return THEMES[self.theme]

    def as_rc_params(
        self,
    ) -> dict[str, object]:
        """
        Gather the rcParams every group produces.

        No two groups write the same key, so they are gathered in the order the fields are
        listed rather than in one that would decide who wins. A theme sets only colours,
        which is what keeps switching between them symmetric: applying a Matplotlib style
        sheet instead would change keys no theme sets back.
        """
        return {
            **self.theme_params.as_rc_params(),
            **self.latex_params.as_rc_params(),
            **self.text_size_params.as_rc_params(),
            **self.artist_params.as_rc_params(),
            **self.frame_params.as_rc_params(),
            **self.legend_params.as_rc_params(),
            **self.save_params.as_rc_params(),
        }


##
## === ACTIVE STYLE
##

_active_figure_params: FigureParams | None = None


def get_figure_params() -> FigureParams:
    """
    The parameters set by the most recent `set_figure_params` call.

    Read them here rather than from the store, which `set_figure_params` owns: it also
    pushes the rc-bearing groups into Matplotlib, so writing the store alone leaves the
    two disagreeing. Figures already built keep the layout they were built with.
    """
    if _active_figure_params is None:
        return FigureParams()
    return _active_figure_params


##
## === STYLE SELECTION
##


def set_figure_params(
    *,
    figure_params: FigureParams | None = None,
) -> None:
    """
    Apply `figure_params` to the global Matplotlib rcParams, and make it the active style.

    Figures made afterwards take their layout from it. To change one part of the style,
    build on what is already active: `dataclasses.replace(get_figure_params(), ...)`.
    """
    global _active_figure_params
    if figure_params is None:
        figure_params = FigureParams()
    _active_figure_params = figure_params
    matplotlib.rcParams.update(figure_params.as_rc_params())


## } MODULE
