## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import dataclasses
import shutil

from collections.abc import Mapping
from enum import Enum

## third-party
import matplotlib

from cycler import cycler

##
## === FONT SIZES
##


@dataclasses.dataclass(
    frozen=True,
    kw_only=True,
)
class TextSizes:
    """
    Point sizes for each kind of text that appears on a figure.

    A single scale sets every size: `largest_size` is the biggest text on the figure, and
    each kind of text sits some number of steps down from it. Each kind is named by what
    it does rather than by how big it is, so moving one leaves the others where they are.

    `size_ratio` sets how far apart the steps are: raise it for a stronger hierarchy,
    lower it for a flatter one. Each `*_level` is how many steps down that kind of text
    sits, and levels need not be whole steps.
    """

    largest_size: float = 10.0
    size_ratio: float = 1.2
    axis_label_level: float = 0.0
    tick_label_level: float = 2.0
    annotation_level: float = 1.0
    legend_level: float = 1.0

    def __post_init__(self) -> None:
        if not (self.largest_size > 0):
            raise ValueError(f"`largest_size` must be positive, but got {self.largest_size}.")
        if not (self.size_ratio > 1):
            raise ValueError(
                f"`size_ratio` must be greater than one, but got {self.size_ratio}."
                " A ratio of one or less would step text up rather than down.",
            )
        for name in (
            "axis_label_level",
            "tick_label_level",
            "annotation_level",
            "legend_level",
        ):
            level = getattr(self, name)
            if level < 0:
                raise ValueError(
                    f"`{name}` must not be negative, but got {level}."
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


##
## === FIGURE LAYOUT
##

## figure sizes are given in cm and text sizes in pt, the two units a page is specified
## in; inches appear only where Matplotlib insists on them
CM_PER_INCH: float = 2.54
PT_PER_INCH: float = 72.0
PT_PER_CM: float = PT_PER_INCH / CM_PER_INCH


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
        for name in (
            "left",
            "right",
            "bottom",
            "top",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"`{name}` must not be negative, but got {value}.")


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
class FigureLayout:
    """
    How much of the page a figure takes, and how much of that is left clear for labels.

    Both are decisions about where a figure sits on a page, so they travel together.
    How tall a figure is, and how many axes it holds, are decided per figure instead.
    """

    figure_width: FigureWidth = FigureWidth()
    figure_margins: FigureMargins = FigureMargins()


## a figure spanning the full text width, and one spanning half of it, which is a single
## column of a two-column page
FULL_PAGE_FIGURE_LAYOUT = FigureLayout(figure_width=FigureWidth(width_fraction=1.0))
HALF_PAGE_FIGURE_LAYOUT = FigureLayout(figure_width=FigureWidth(width_fraction=0.5))


##
## === ACTIVE STYLE
##

_active_text_sizes: TextSizes = TextSizes()
_active_figure_layout: FigureLayout = FULL_PAGE_FIGURE_LAYOUT


def get_text_sizes() -> TextSizes:
    """
    Text sizes set by the most recent `set_theme` call.

    Read them here rather than from the store, which `set_theme` owns: it also pushes the
    sizes into rcParams, so writing the store alone leaves the two disagreeing.
    """
    return _active_text_sizes


def get_figure_layout() -> FigureLayout:
    """
    Figure layout set by the most recent `set_theme` call.

    Read it here rather than from the store, which `set_theme` owns. Figures already built
    keep the layout they were built with.
    """
    return _active_figure_layout


##
## === COLOR THEMES
##

LIGHT_RC_PARAMS: dict[str, object] = {
    ## backgrounds
    "figure.facecolor":
    "white",
    "axes.facecolor":
    "white",
    "savefig.facecolor":
    "white",
    ## foreground
    "axes.edgecolor":
    "#222222",
    "axes.labelcolor":
    "#222222",
    "text.color":
    "#222222",
    "axes.titlecolor":
    "#222222",
    "xtick.color":
    "#333333",
    "ytick.color":
    "#333333",
    ## grid
    "grid.color":
    "#dddddd",
    "grid.alpha":
    0.6,
    ## default colors
    "axes.prop_cycle":
    cycler(
        color=[
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
        ],
    ),
}

DARK_RC_PARAMS: dict[str, object] = {
    ## backgrounds
    "figure.facecolor":
    "#0b0b0e",
    "axes.facecolor":
    "#0b0b0e",
    "savefig.facecolor":
    "#0b0b0e",
    ## foreground
    "axes.edgecolor":
    "#e6e6e6",
    "axes.labelcolor":
    "#e6e6e6",
    "text.color":
    "#e6e6e6",
    "axes.titlecolor":
    "#e6e6e6",
    "xtick.color":
    "#cfcfd2",
    "ytick.color":
    "#cfcfd2",
    ## grid
    "grid.color":
    "#2e2e35",
    "grid.alpha":
    0.3,
    ## default colors
    "axes.prop_cycle":
    cycler(
        color=[
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
        ],
    ),
}


class Theme(Enum):
    """Available Matplotlib color themes."""

    LIGHT = "light"
    DARK = "dark"


THEMES: Mapping[Theme, dict[str, object]] = {
    Theme.LIGHT: LIGHT_RC_PARAMS,
    Theme.DARK: DARK_RC_PARAMS,
}

##
## === HELPERS
##


def _get_base_rc_params(
    *,
    use_tex: bool = True,
    text_sizes: TextSizes | None = None,
) -> dict[str, object]:
    if text_sizes is None:
        text_sizes = TextSizes()
    rc_params: dict[str, object] = {
        ## font
        "font.family": "serif",
        **text_sizes.as_rc_params(),
        ## Lines, ticks and pads are in pt, like the text, and are chosen for the
        ## medium a figure is bound for rather than derived from the text size. A figure
        ## drawn wider keeps the same stroke weights, since a point stays a point.
        "lines.linewidth": 0.9,
        "lines.markersize": 5.0,
        "axes.linewidth": 0.6,
        ## ticks
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "axes.labelpad": 3.0,
        "xtick.major.pad": 2.5,
        "ytick.major.pad": 2.5,
        "xtick.minor.pad": 2.5,
        "ytick.minor.pad": 2.5,
        ## legend
        "legend.labelspacing": 0.2,
        "legend.loc": "upper right",
        "legend.frameon": False,
        ## figure + saving
        "figure.figsize": (
            8.0,
            6.0,
        ),
        "savefig.dpi": 200,
        "savefig.bbox": None,
        "savefig.transparent": False,
        "savefig.pad_inches": 0.0,
    }
    if use_tex and (shutil.which("latex") is not None):
        rc_params.update(
            {
                "text.usetex":
                True,
                "text.latex.preamble":
                r"""
                    \usepackage{bm,amsmath,mathrsfs,amssymb,url,xfrac}
                    \providecommand{\mathdefault}[1]{#1}
                """,
            },
        )
    else:
        rc_params.update({"text.usetex": False})
    return rc_params


def _compose_rc_params(
    *,
    theme: Theme = Theme.LIGHT,
    use_tex: bool = True,
    text_sizes: TextSizes | None = None,
) -> dict[str, object]:
    rc_params = _get_base_rc_params(
        use_tex=use_tex,
        text_sizes=text_sizes,
    ).copy()
    rc_params.update(THEMES[theme])
    return rc_params


##
## === THEME SELECTION
##


def set_theme(
    *,
    theme: Theme | str = Theme.LIGHT,
    use_tex: bool = True,
    text_sizes: TextSizes | None = None,
    figure_layout: FigureLayout | None = None,
) -> None:
    """
    Apply a theme to Matplotlib's global rcParams.

    `text_sizes` sets the point size of each kind of text. `figure_layout` becomes the
    default for figures made after it, setting how much page they take and how much is
    left clear for their labels.
    """
    global _active_text_sizes, _active_figure_layout
    if text_sizes is None:
        text_sizes = TextSizes()
    if figure_layout is None:
        figure_layout = FULL_PAGE_FIGURE_LAYOUT
    _active_text_sizes = text_sizes
    _active_figure_layout = figure_layout
    if isinstance(theme, str):
        theme = Theme(theme)
    if theme == Theme.DARK:
        try:
            import matplotlib.pyplot as _mpl_plot
            _mpl_plot.style.use("dark_background")
        except Exception:
            pass
    matplotlib.rcParams.update(
        _compose_rc_params(
            theme=theme,
            use_tex=use_tex,
            text_sizes=text_sizes,
        ),
    )


## } MODULE
