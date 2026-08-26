## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import dataclasses
import math
import shutil

from collections.abc import Mapping
from enum import Enum

## third-party
import matplotlib

from cycler import cycler

## local
from jormi.ww_io import manage_log

##
## === UNITS
##

## figure sizes are given in cm and text sizes in pt, the two units a page is specified
## in; inches appear only where Matplotlib insists on them
CM_PER_INCH: float = 2.54
PT_PER_INCH: float = 72.0
PT_PER_CM: float = PT_PER_INCH / CM_PER_INCH

## how finely a saved raster is sampled, in the cm a figure is sized in; Matplotlib wants
## it per inch, so `CM_PER_INCH` converts at the point it is handed over. 250 is about 635
## dpi, which is what line art wants in print, and a round number in the unit used here
DEFAULT_PIXELS_PER_CM: float = 250.0

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

    A single scale sets every size: `largest_size` is the biggest text on the figure, and
    each kind of text sits some number of steps down from it. Each kind is named by what
    it does rather than by how big it is, so moving one leaves the others where they are.

    `size_ratio` sets how far apart the steps are: raise it for a stronger hierarchy,
    lower it for a flatter one. Each `*_level` is how many steps down that kind of text
    sits, and levels need not be whole steps.
    """

    largest_size: float = 12.0
    size_ratio: float = 1.16
    axis_label_level: float = 0.0
    tick_label_level: float = 2.0
    annotation_level: float = 2.0
    legend_level: float = 1.0

    def __post_init__(self) -> None:
        if not (self.largest_size > 0):
            raise ValueError(f"`largest_size` must be positive, but got {self.largest_size}.")
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
            level = getattr(self, param_name)
            if level < 0:
                raise ValueError(
                    f"`{param_name}` must not be negative, but got {level}."
                    " No text may be larger than `largest_size`.",
                )

    @property
    def axis_label_size(self) -> float:
        return self.compute_size_at_level(level=self.axis_label_level)

    @property
    def tick_label_size(self) -> float:
        return self.compute_size_at_level(level=self.tick_label_level)

    @property
    def annotation_size(self) -> float:
        return self.compute_size_at_level(level=self.annotation_level)

    @property
    def legend_size(self) -> float:
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
        if level < 0:
            raise ValueError(
                f"`level` must not be negative, but got {level}."
                " No text may be larger than `largest_size`.",
            )
        return self.largest_size * self.size_ratio**(-level)

    def compute_level_at_size(
        self,
        *,
        text_size: float,
    ) -> float:
        """
        Level that `text_size` sits at, inverting `compute_size_at_level`.

        Sizes are set by level rather than in pt, so asking for a size a little under one of
        the named kinds of text means finding the level it falls at.
        """
        if not (text_size > 0):
            raise ValueError(f"`text_size` must be positive, but got {text_size}.")
        if text_size > self.largest_size:
            raise ValueError(
                f"`text_size` ({text_size}) is larger than `largest_size` ({self.largest_size}),"
                " and no text may be larger than that.",
            )
        return math.log(self.largest_size / text_size) / math.log(self.size_ratio)

    def as_rc_params(self) -> dict[str, object]:
        """Map each kind of text onto the Matplotlib rcParams that consume it."""
        return {
            "font.size": self.axis_label_size,
            "axes.labelsize": self.axis_label_size,
            ## figure-level x/y labels shared across a grid of axes
            "figure.labelsize": self.axis_label_size,
            ## titles are not used, but a stray one should match the labels rather than
            ## fall back to Matplotlib's default of larger-than-everything
            "axes.titlesize": self.axis_label_size,
            "figure.titlesize": self.axis_label_size,
            "xtick.labelsize": self.tick_label_size,
            "ytick.labelsize": self.tick_label_size,
            "legend.fontsize": self.legend_size,
        }


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class ArtistParams:
    """
    The marks that draw the data, in pt: the lines, the markers, and the markers' edges.

    Narrower than what Matplotlib calls an artist, which is anything drawable at all;
    the text, the panel frame and the legend are each styled by their own group.

    Like the text, these are chosen for the medium a figure is bound for rather than
    derived from its size: a figure drawn wider keeps the same stroke weights, since a
    pt stays a pt.
    """

    line_width: float = 0.9
    marker_size: float = 5.0
    marker_edge_width: float = 0.6

    def __post_init__(self) -> None:
        ## a mark is turned off by giving it no width, so zero is allowed and only a negative is not
        for param_name in (
            "line_width",
            "marker_size",
            "marker_edge_width",
        ):
            param_value = getattr(self, param_name)
            if param_value < 0:
                raise ValueError(f"`{param_name}` must not be negative, but got {param_value}.")

    def as_rc_params(self) -> dict[str, object]:
        """Map each mark onto the Matplotlib rcParams that consume it."""
        return {
            "lines.linewidth": self.line_width,
            "lines.markersize": self.marker_size,
            ## an edge holds a marker rather than being it, so it is drawn lighter than the
            ## data line; left unset Matplotlib would outweigh that line instead
            "lines.markeredgewidth": self.marker_edge_width,
        }


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class FrameParams:
    """
    The panel's furniture, in pt: its frame, its ticks, and the gaps around their labels.

    One weight covers the frame, the ticks and any box drawn around text, since they are
    all furniture holding the data rather than the data itself.

    The two gaps are measured in a chain rather than from the same edge:

        frame --`tick_label_gap`--> tick labels --`axis_label_gap`--> axis label

    so they do not compound, and longer tick labels push the axis label outward on their
    own. Together with the labels themselves, that chain is what a margin has to hold.

    Tick lengths run inward while `ticks_point_inward` holds, so they take room from the
    panel rather than from the margin.
    """

    line_width: float = 0.4
    major_tick_length: float = 3.0
    minor_tick_length: float = 1.6
    tick_label_gap: float = 2.5
    axis_label_gap: float = 3.0
    ticks_point_inward: bool = True
    ticks_on_all_sides: bool = True
    show_minor_ticks: bool = True

    def __post_init__(self) -> None:
        ## a frame with no ticks asks for a length of zero, so only a negative is refused
        for param_name in (
            "line_width",
            "major_tick_length",
            "minor_tick_length",
            "tick_label_gap",
            "axis_label_gap",
        ):
            param_value = getattr(self, param_name)
            if param_value < 0:
                raise ValueError(f"`{param_name}` must not be negative, but got {param_value}.")

    def as_rc_params(self) -> dict[str, object]:
        """Map each part of the frame onto the Matplotlib rcParams that consume it."""
        tick_direction = "in" if self.ticks_point_inward else "out"
        return {
            "axes.linewidth": self.line_width,
            ## a box drawn around text is furniture too, and Matplotlib's default of 1.0
            ## would outweigh both the frame and the data
            "patch.linewidth": self.line_width,
            "xtick.major.width": self.line_width,
            "ytick.major.width": self.line_width,
            "xtick.minor.width": self.line_width,
            "ytick.minor.width": self.line_width,
            "xtick.major.size": self.major_tick_length,
            "ytick.major.size": self.major_tick_length,
            "xtick.minor.size": self.minor_tick_length,
            "ytick.minor.size": self.minor_tick_length,
            "xtick.major.pad": self.tick_label_gap,
            "ytick.major.pad": self.tick_label_gap,
            "xtick.minor.pad": self.tick_label_gap,
            "ytick.minor.pad": self.tick_label_gap,
            ## measured against the frame rather than the text, so it sits here
            "axes.labelpad": self.axis_label_gap,
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

    The three spacings are each measured as a fraction of the legend's own text size, as
    Matplotlib measures them, so they hold their proportions as the text size changes.
    """

    entry_gap: float = 0.2
    handle_gap: float = 0.8
    column_gap: float = 2.0
    frame_margin: float = 0.4
    location: str = "upper right"
    show_frame: bool = False

    def __post_init__(self) -> None:
        ## a legend packed hard against its anchor asks for gaps of zero, so only a negative is refused
        for param_name in (
            "entry_gap",
            "handle_gap",
            "column_gap",
            "frame_margin",
        ):
            param_value = getattr(self, param_name)
            if param_value < 0:
                raise ValueError(f"`{param_name}` must not be negative, but got {param_value}.")

    def as_rc_params(self) -> dict[str, object]:
        """Map the legend style onto the Matplotlib rcParams that consume it."""
        return {
            "legend.labelspacing": self.entry_gap,
            "legend.handletextpad": self.handle_gap,
            "legend.columnspacing": self.column_gap,
            "legend.borderpad": self.frame_margin,
            "legend.loc": self.location,
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
    crop_margin_cm: float = 0.0
    transparent_background: bool = False

    def __post_init__(self) -> None:
        if not (self.pixels_per_cm > 0):
            raise ValueError(f"`pixels_per_cm` must be positive, but got {self.pixels_per_cm}.")
        if self.crop_margin_cm < 0:
            raise ValueError(f"`crop_margin_cm` must not be negative, but got {self.crop_margin_cm}.")
        if self.crop_to_ink:
            manage_log.log_warning(
                text=(
                    "`crop_to_ink` makes the saved file the size of its ink, not the size the"
                    " figure was asked for, so its text will no longer sit at the pt size the"
                    " page expects. Widen the margins instead of cropping."
                ),
            )

    def as_rc_params(self) -> dict[str, object]:
        """Map the save style onto the Matplotlib rcParams that consume it."""
        return {
            "savefig.dpi": self.pixels_per_cm * CM_PER_INCH,
            "savefig.bbox": "tight" if self.crop_to_ink else None,
            "savefig.pad_inches": self.crop_margin_cm / CM_PER_INCH,
            "savefig.transparent": self.transparent_background,
        }


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class LatexParams:
    """
    How a figure's text is typeset, when LaTeX sets it.

    With `use_tex`, every string goes through LaTeX rather than Matplotlib's own mathtext,
    so a figure is set in the same face as the document it is bound for. Plain strings then
    come out upright and `$...$` italic, which is why words read heavier than maths.

    `font_package` is what makes the two match, so it should name the face the document
    loads. `math_packages` are the ones the labels need; `extra_preamble` is for anything
    else, and is placed last so it can override what comes before it.
    """

    use_tex: bool = True
    font_package: str = "lmodern"
    math_packages: tuple[str, ...] = ("amsmath",)
    extra_preamble: str = ""

    def as_rc_params(self) -> dict[str, object]:
        """
        Map the typesetting onto the Matplotlib rcParams that consume it.

        LaTeX is only asked for when it is installed, so a figure still draws without it,
        in Matplotlib's own mathtext rather than not at all.
        """
        if not (self.use_tex and (shutil.which("latex") is not None)):
            return {"text.usetex": False}
        packages = "\n".join(
            f"\\usepackage{{{package_name}}}"
            for package_name in (self.font_package, *self.math_packages)
            if package_name
        )
        return {
            "text.usetex": True,
            "text.latex.preamble": f"{packages}\n{self.extra_preamble}",
        }


##
## === FIGURE LAYOUT
##

@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class FigureMargins:
    """
    Space between the figure edge and the axes, in pt (1 pt = 1/72 inch).

    Absolute, not fractional, because what the margins hold is measured in pt: tick
    labels, axis labels and the tick marks themselves. A fraction would give a wide
    figure more room than its labels need and a narrow one too little.
    """

    left: float = 34.0
    right: float = 6.0
    bottom: float = 28.0
    top: float = 6.0

    def __post_init__(self) -> None:
        for param_name in (
            "left",
            "right",
            "bottom",
            "top",
        ):
            param_value = getattr(self, param_name)
            if param_value < 0:
                raise ValueError(f"`{param_name}` must not be negative, but got {param_value}.")


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class FigurePadding:
    """
    Space left clear outside everything a figure draws, in pt.

    Where `FigureMargins` says how much room to leave for the labels, this says how much to
    leave beyond them. A label's extent is not knowable before it is drawn, so a figure that
    is fitted to its contents measures the labels and is given only this one constant.
    """

    left: float = 6.0
    right: float = 6.0
    bottom: float = 6.0
    top: float = 6.0

    def __post_init__(self) -> None:
        for param_name in (
            "left",
            "right",
            "bottom",
            "top",
        ):
            param_value = getattr(self, param_name)
            if param_value < 0:
                raise ValueError(f"`{param_name}` must not be negative, but got {param_value}.")


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

    def __post_init__(self) -> None:
        if not (self.max_width_cm > 0):
            raise ValueError(f"`max_width_cm` must be positive, but got {self.max_width_cm}.")
        if not (0.0 < self.width_fraction <= 1.0):
            raise ValueError(
                f"`width_fraction` must lie in (0, 1], but got {self.width_fraction}."
                " A figure cannot be drawn wider than the page it sits on.",
            )

    @property
    def width_cm(self) -> float:
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

    column: float = 10.0
    row: float = 10.0

    def __post_init__(self) -> None:
        for param_name in (
            "column",
            "row",
        ):
            param_value = getattr(self, param_name)
            if param_value < 0:
                raise ValueError(f"`{param_name}` must not be negative, but got {param_value}.")


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
    a panel gap: `gap.column` for a bar on the left or right, `gap.row` for one above or
    below. Left unset it is the gap the figure already spaces its panels by, so one value
    covers the whole figure; set it to space a bar differently from the panels.

    `aspect_ratio` is the bar's length over its thickness, so a bar keeps its proportions
    whatever it describes: one beside a single panel and one spanning a whole grid are the
    same shape, where a share of the panel would have made the second twice as thick.
    """

    gap: PanelGaps | None = None
    aspect_ratio: float = 15.0

    def __post_init__(self) -> None:
        if not (self.aspect_ratio > 0):
            raise ValueError(f"`aspect_ratio` must be positive, but got {self.aspect_ratio}.")


## a figure spanning the full text width, and one spanning half of it, which is a single
## column of a two-column page
FULL_PAGE_FIGURE_LAYOUT = FigureLayout(figure_width=FigureWidth(width_fraction=1.0))
HALF_PAGE_FIGURE_LAYOUT = FigureLayout(figure_width=FigureWidth(width_fraction=0.5))


##
## === COLOR THEMES
##

@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class ThemeParams:
    """
    The colours a theme sets, each named for what it colours rather than for the keys
    it lands in.

    A theme is only this overlay, so the two are structurally identical and switching
    between them puts every key back.
    """

    background_color: str
    foreground_color: str
    tick_color: str
    grid_color: str
    grid_alpha: float
    cycled_colors: tuple[str, ...]

    def as_rc_params(self) -> dict[str, object]:
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
            "axes.prop_cycle": cycler(color=list(self.cycled_colors)),
        }


LIGHT_THEME_PARAMS = ThemeParams(
    background_color="white",
    foreground_color="#222222",
    tick_color="#333333",
    grid_color="#dddddd",
    grid_alpha=0.6,
    cycled_colors=(
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
    cycled_colors=(
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


class Theme(Enum):
    """Available Matplotlib color themes."""

    LIGHT = "light"
    DARK = "dark"


THEMES: Mapping[Theme, ThemeParams] = {
    Theme.LIGHT: LIGHT_THEME_PARAMS,
    Theme.DARK: DARK_THEME_PARAMS,
}

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

    Each group knows the rcParams it produces; the two layouts are the exception, since
    jormi places panels and colorbars itself rather than handing that to Matplotlib.
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
    def theme_params(self) -> ThemeParams:
        """The colours the chosen theme sets, for what jormi draws itself."""
        return THEMES[self.theme]

    def as_rc_params(self) -> dict[str, object]:
        """Gather every group's rcParams, with the theme's colours overlaid last."""
        rc_params: dict[str, object] = {
            ## the typeface, which pairs with the LaTeX settings below
            "font.family": "serif",
            **self.text_size_params.as_rc_params(),
            **self.artist_params.as_rc_params(),
            **self.frame_params.as_rc_params(),
            **self.legend_params.as_rc_params(),
            **self.save_params.as_rc_params(),
            **self.latex_params.as_rc_params(),
        }
        ## a theme is only a colour overlay, so switching between them is symmetric;
        ## applying one of Matplotlib's style sheets would change keys no theme sets back
        rc_params.update(THEMES[self.theme].as_rc_params())
        return rc_params


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
    Apply `figure_params` to Matplotlib's global rcParams, and make it the active style.

    Figures made afterwards take their layout from it. To change one part of the style,
    build on what is already active: `dataclasses.replace(get_figure_params(), ...)`.
    """
    global _active_figure_params
    if figure_params is None:
        figure_params = FigureParams()
    _active_figure_params = figure_params
    matplotlib.rcParams.update(figure_params.as_rc_params())


## } MODULE
