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
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TypeAlias,
    overload,
)

## third-party
import numpy
from matplotlib import pyplot as mpl_plot
from matplotlib import rcParams
from matplotlib.axes import Axes as mpl_Axes
from matplotlib.figure import Figure as mpl_Figure
from numpy.typing import NDArray

## local
from jormi.ww_io import (
    manage_io,
    manage_log,
    manage_shell,
)
from jormi.ww_plots import style_plots
from jormi.ww_validation import validate_box_positions, validate_types
from jormi.ww_types import box_positions

##
## === TYPE ALIASES
##

PlotPanel: TypeAlias = mpl_Axes
PlotPanelGrid: TypeAlias = NDArray[numpy.object_]

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
        for name in (
            "width_cm",
            "height_cm",
        ):
            value = getattr(self, name)
            if not (value > 0):
                raise ValueError(f"`{name}` must be positive, but got {value}.")

    @property
    def as_mpl_shape(self) -> tuple[float, float]:
        """The pair, width first and in inches, as Matplotlib's `figsize` reads it."""
        return (
            self.width_cm / style_plots.CM_PER_INCH,
            self.height_cm / style_plots.CM_PER_INCH,
        )

    @property
    def aspect_ratio(self) -> float:
        """Width over height."""
        return self.width_cm / self.height_cm


##
## === INTERNAL HELPERS
##

DEFAULT_PANEL_SHAPE: BoxShape = BoxShape(
    width_cm=15.0,
    height_cm=10.0,
)


def _compute_figure_shape(
    *,
    num_panel_rows: int = 1,
    num_panel_columns: int = 1,
    figure_scale: float = 1.0,
    panel_shape: BoxShape = DEFAULT_PANEL_SHAPE,
) -> BoxShape:
    """Compute figure size (cm) from the share each panel gets."""
    if (num_panel_rows < 1) or (num_panel_columns < 1):
        raise ValueError("`num_panel_rows` and `num_panel_columns` must both be >= 1.")
    return BoxShape(
        width_cm=figure_scale * panel_shape.width_cm * num_panel_columns,
        height_cm=figure_scale * panel_shape.height_cm * num_panel_rows,
    )


def _place_panels_in_figure(
    *,
    figure: mpl_Figure,
    figure_shape: BoxShape,
    figure_margins: style_plots.FigureMargins,
    x_spacing: float | None = None,
    y_spacing: float | None = None,
) -> None:
    """
    Place the panels within `figure`, leaving `figure_margins` clear around them.

    Margins are in pt, while Matplotlib places panels as fractions of the figure, so
    `figure_shape` is what converts between the two.
    """
    width_pt = figure_shape.width_cm * style_plots.PT_PER_CM
    height_pt = figure_shape.height_cm * style_plots.PT_PER_CM
    if (figure_margins.left + figure_margins.right) >= width_pt:
        raise ValueError(
            f"margins `left` + `right` ({figure_margins.left + figure_margins.right} pt)"
            f" leave no room for the panels in a figure {width_pt:.1f} pt wide.",
        )
    if (figure_margins.bottom + figure_margins.top) >= height_pt:
        raise ValueError(
            f"margins `bottom` + `top` ({figure_margins.bottom + figure_margins.top} pt)"
            f" leave no room for the panels in a figure {height_pt:.1f} pt tall.",
        )
    figure.subplots_adjust(
        left=figure_margins.left / width_pt,
        right=1.0 - (figure_margins.right / width_pt),
        bottom=figure_margins.bottom / height_pt,
        top=1.0 - (figure_margins.top / height_pt),
    )
    if (x_spacing is not None) and (y_spacing is not None):
        figure.subplots_adjust(
            wspace=x_spacing,
            hspace=y_spacing,
        )


def _split_width_across_panels(
    *,
    width_cm: float,
    num_panel_columns: int,
    aspect_ratio: float,
) -> BoxShape:
    """
    Share a figure's width between the panels in a row, giving the share each one gets.

    `aspect_ratio` is the width / height of that share; a panel is drawn smaller than
    its share, by whatever the margins hold.
    """
    if num_panel_columns < 1:
        raise ValueError(f"`num_panel_columns` must be >= 1, but got {num_panel_columns}.")
    if not (aspect_ratio > 0):
        raise ValueError(f"`aspect_ratio` must be positive, but got {aspect_ratio}.")
    panel_width_cm = width_cm / num_panel_columns
    return BoxShape(
        height_cm=panel_width_cm / aspect_ratio,
        width_cm=panel_width_cm,
    )


def _get_figure_shape(
    *,
    num_panel_rows: int,
    num_panel_columns: int,
    figure_scale: float,
    panel_shape: BoxShape | None,
    figure_layout: style_plots.FigureLayout | None,
    aspect_ratio: float | None,
) -> tuple[BoxShape, style_plots.FigureLayout]:
    """
    Size a figure from a page layout, or in its own terms, but not both.

    A layout pins the figure to a share of the page, so adding panels makes each one
    smaller. `panel_shape` instead gives each panel a fixed size, so the figure grows as
    panels are added, and the figure is no longer tied to a page.
    """
    if figure_layout is None:
        active_figure_layout = style_plots.get_figure_layout()
    else:
        active_figure_layout = figure_layout
    if panel_shape is None:
        if aspect_ratio is None:
            aspect_ratio = DEFAULT_PANEL_SHAPE.aspect_ratio
        page_panel_shape = _split_width_across_panels(
            width_cm=active_figure_layout.figure_width.width_cm,
            num_panel_columns=num_panel_columns,
            aspect_ratio=aspect_ratio,
        )
        figure_shape = _compute_figure_shape(
            num_panel_rows=num_panel_rows,
            num_panel_columns=num_panel_columns,
            panel_shape=page_panel_shape,
        )
        return figure_shape, active_figure_layout
    if figure_layout is not None:
        raise ValueError(
            "`figure_layout` and `panel_shape` are mutually exclusive: a layout sizes the"
            " figure to a share of the page, while `panel_shape` sizes each panel outright.",
        )
    if aspect_ratio is not None:
        raise ValueError(
            "`aspect_ratio` only applies when a figure is sized to a page;"
            " with `panel_shape` the shape of each panel is already set by it.",
        )
    figure_shape = _compute_figure_shape(
        num_panel_rows=num_panel_rows,
        num_panel_columns=num_panel_columns,
        figure_scale=figure_scale,
        panel_shape=panel_shape,
    )
    return figure_shape, active_figure_layout


##
## === FIGURE FACTORY
##


@overload
def create_figure(
    *,
    num_panel_rows: None = None,
    num_panel_columns: None = None,
    figure_scale: float = 1.0,
    panel_shape: BoxShape | None = None,
    figure_layout: style_plots.FigureLayout | None = None,
    aspect_ratio: float | None = None,
    x_spacing: float = 0.05,
    y_spacing: float = 0.05,
    share_x: bool = False,
    share_y: bool = False,
    auto_style: bool = True,
    theme: style_plots.Theme | str | None = None,
) -> tuple[mpl_Figure, PlotPanel]:
    ...


@overload
def create_figure(
    *,
    num_panel_rows: int,
    num_panel_columns: int,
    figure_scale: float = 1.0,
    panel_shape: BoxShape | None = None,
    figure_layout: style_plots.FigureLayout | None = None,
    aspect_ratio: float | None = None,
    x_spacing: float = 0.05,
    y_spacing: float = 0.05,
    share_x: bool = False,
    share_y: bool = False,
    auto_style: bool = True,
    theme: style_plots.Theme | str | None = None,
) -> tuple[mpl_Figure, PlotPanelGrid]:
    ...


def create_figure(
    *,
    num_panel_rows: int | None = None,
    num_panel_columns: int | None = None,
    figure_scale: float = 1.0,
    panel_shape: BoxShape | None = None,
    figure_layout: style_plots.FigureLayout | None = None,
    aspect_ratio: float | None = None,
    x_spacing: float = 0.05,
    y_spacing: float = 0.05,
    share_x: bool = False,
    share_y: bool = False,
    auto_style: bool = True,
    theme: style_plots.Theme | str | None = None,
) -> tuple[mpl_Figure, PlotPanel | PlotPanelGrid]:
    """
    Create a Matplotlib figure and Axis / Axes grid.

    Overloads:
        - create_figure() -> (figure, panel)
        - create_figure(num_panel_rows=N, num_panel_columns=M) -> (figure, panels) with shape (N, M)

    Sizing
    ------
    `panel_shape` is the share of the figure given to one panel, in cm, so the figure
    comes out `panel_shape` times the grid. It is not the size the panel is drawn at: tick
    labels and axis labels are held in margins taken out of that share, so the panel itself
    is drawn smaller. A colorbar is placed beyond the panel rather than within those
    margins, and so can reach past the figure edge.

    Notes
    -----
    - If `num_panel_rows` and `num_panel_columns` are both None (or omitted), a single-panel
      figure is created and a single Axis is returned.
    - If `num_panel_rows` and `num_panel_columns` are both provided as integers, a grid of
      panels is created and a 2D object-dtype array of Axes is returned.
    - Mixed None/int specifications are not allowed.
    """
    if auto_style and (theme is not None):
        style_plots.set_theme(theme=theme)
    if (num_panel_rows is None) and (num_panel_columns is None):
        figure_shape, active_figure_layout = _get_figure_shape(
            num_panel_rows=1,
            num_panel_columns=1,
            figure_scale=figure_scale,
            panel_shape=panel_shape,
            figure_layout=figure_layout,
            aspect_ratio=aspect_ratio,
        )
        figure, panel = mpl_plot.subplots(
            nrows=1,
            ncols=1,
            figsize=figure_shape.as_mpl_shape,
            sharex=share_x,
            sharey=share_y,
            squeeze=True,
        )
        _place_panels_in_figure(
            figure=figure,
            figure_shape=figure_shape,
            figure_margins=active_figure_layout.figure_margins,
        )
        return figure, panel
    if (num_panel_rows is None) or (num_panel_columns is None):
        raise ValueError(
            "Either specify both `num_panel_rows` and `num_panel_columns`, or neither."
            " Mixed None/int combinations are not supported.",
        )
    validate_types.ensure_finite_int(
        param=num_panel_rows,
        param_name="num_panel_rows",
        require_positive=True,
    )
    validate_types.ensure_finite_int(
        param=num_panel_columns,
        param_name="num_panel_columns",
        require_positive=True,
    )
    if (num_panel_rows == 1) and (num_panel_columns == 1):
        raise ValueError(
            "For a single-panel figure, omit `num_panel_rows` and `num_panel_columns` so that"
            " a single Axis is returned instead of a 1x1 Axes grid.",
        )
    figure_shape, active_figure_layout = _get_figure_shape(
        num_panel_rows=num_panel_rows,
        num_panel_columns=num_panel_columns,
        figure_scale=figure_scale,
        panel_shape=panel_shape,
        figure_layout=figure_layout,
        aspect_ratio=aspect_ratio,
    )
    figure, panels = mpl_plot.subplots(
        nrows=num_panel_rows,
        ncols=num_panel_columns,
        figsize=figure_shape.as_mpl_shape,
        sharex=share_x,
        sharey=share_y,
        squeeze=False,
    )
    _place_panels_in_figure(
        figure=figure,
        figure_shape=figure_shape,
        figure_margins=active_figure_layout.figure_margins,
        x_spacing=x_spacing,
        y_spacing=y_spacing,
    )
    panels_grid: PlotPanelGrid = numpy.asarray(panels, dtype=object)
    return figure, panels_grid


def create_figure_grid(
    *,
    num_panel_rows: int = 1,
    num_panel_columns: int = 1,
    figure_scale: float = 1.0,
    panel_shape: BoxShape | None = None,
    figure_layout: style_plots.FigureLayout | None = None,
    aspect_ratio: float | None = None,
    x_spacing: float = 0.05,
    y_spacing: float = 0.05,
    share_x: bool = False,
    share_y: bool = False,
    auto_style: bool = True,
    theme: style_plots.Theme | str | None = None,
) -> tuple[mpl_Figure, PlotPanelGrid]:
    """
    Like `create_figure`, but always returns a 2D panel grid of shape (num_panel_rows, num_panel_columns), so
    callers can always index panels as panels_grid[row, col].
    """
    if (num_panel_rows == 1) and (num_panel_columns == 1):
        figure, panel = create_figure(
            figure_scale=figure_scale,
            panel_shape=panel_shape,
            figure_layout=figure_layout,
            aspect_ratio=aspect_ratio,
            auto_style=auto_style,
            theme=theme,
        )
        panels_grid: PlotPanelGrid = numpy.asarray([[panel]], dtype=object)
        return figure, panels_grid
    figure, panels_grid = create_figure(
        num_panel_rows=num_panel_rows,
        num_panel_columns=num_panel_columns,
        figure_scale=figure_scale,
        panel_shape=panel_shape,
        figure_layout=figure_layout,
        aspect_ratio=aspect_ratio,
        x_spacing=x_spacing,
        y_spacing=y_spacing,
        share_x=share_x,
        share_y=share_y,
        auto_style=auto_style,
        theme=theme,
    )
    return figure, panels_grid


##
## === AXIS HELPERS
##

_Side = box_positions.Positions.Side


@dataclass(frozen=True)
class PanelBounds:
    """Bounding box for a panel in figure coordinates."""

    x_min: float
    y_min: float
    x_width: float
    y_width: float


def compute_adjacent_panel_bounds(
    *,
    panel: PlotPanel,
    side: _Side = box_positions.Positions.Side.Right,
    gap: float = 0.1,
    thickness: float = 1.0,
    length: float = 1.0,
) -> PanelBounds:
    """Compute figure bounds for a panel placed adjacent to `panel`.

    The new panel sits on the `side` of `panel`, offset by `gap` (in figure coordinates).
    `thickness` sets its extent perpendicular to `side`, as a fraction of `panel`'s
    corresponding dimension. `length` sets its span parallel to `side`, also as a
    fraction, centered on `panel`'s edge.
    """
    box = panel.get_position()
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


def add_inset_panel(
    *,
    panel: PlotPanel,
    bounds: tuple[float, float, float, float] = (0.0, 1.0, 1.0, 0.5),
    x_label: str | None = None,
    y_label: str | None = None,
    fontsize: float | None = None,
    x_label_alignment: box_positions.Positions.PositionLike = box_positions.Positions.Side.Top,
    y_label_alignment: box_positions.Positions.PositionLike = box_positions.Positions.Side.Right,
) -> PlotPanel:
    """Add an inset Axis to `panel`."""
    x_label_side = validate_box_positions.as_box_side(x_label_alignment)
    y_label_side = validate_box_positions.as_box_side(y_label_alignment)
    inset_panel = panel.inset_axes(bounds)
    if fontsize is None:
        fontsize = rcParams["axes.labelsize"]
    if x_label is not None:
        inset_panel.set_xlabel(
            xlabel=x_label,
            fontsize=fontsize,
        )
        inset_panel.xaxis.set_label_position(x_label_side.value)  # pyright: ignore[reportArgumentType]
    if y_label is not None:
        inset_panel.set_ylabel(
            ylabel=y_label,
            fontsize=fontsize,
        )
        inset_panel.yaxis.set_label_position(y_label_side.value)  # pyright: ignore[reportArgumentType]
    inset_panel.tick_params(
        axis="x",
        labeltop=(x_label_side is box_positions.Positions.Side.Top),
        labelbottom=(x_label_side is box_positions.Positions.Side.Bottom),
        top=True,
        bottom=True,
    )
    if x_label_side is box_positions.Positions.Side.Top:
        inset_panel.xaxis.tick_top()
    inset_panel.tick_params(
        axis="y",
        labelleft=(y_label_side is box_positions.Positions.Side.Left),
        labelright=(y_label_side is box_positions.Positions.Side.Right),
        left=True,
        right=True,
    )
    if y_label_side is box_positions.Positions.Side.Right:
        inset_panel.yaxis.tick_right()
    return inset_panel


##
## === IO HELPERS
##


def save_figure(
    *,
    figure: mpl_Figure,
    figure_path: str | Path,
    dpi: int = 200,
    verbose: bool = True,
) -> None:
    """
    Save `figure` to `figure_path`; close it.

    Accepts `.png` or `.pdf` paths; errors are logged rather than raised.
    """
    if not str(figure_path).endswith(".png") and not str(figure_path).endswith(".pdf"):
        raise ValueError("figures must end with `.png` or `.pdf`.")
    try:
        figure.savefig(figure_path, dpi=dpi)
        if verbose:
            manage_log.log_action(
                title="Save figure",
                outcome=manage_log.ActionOutcome.SUCCESS,
                message="Saved figure.",
                notes={"file": str(figure_path)},
            )
    except FileNotFoundError as exception:
        manage_log.log_error(text=f"FileNotFoundError: {exception}")
    except PermissionError as exception:
        manage_log.log_error(
            text=f"PermissionError: You do not have permission to save to: {figure_path}",
            notes={"details": str(exception)},
        )
    except IOError as exception:
        manage_log.log_error(
            text=f"IOError: An error occurred while trying to save the figure to: {figure_path}",
            notes={"details": str(exception)},
        )
    except Exception as exception:
        manage_log.log_error(text=f"Unexpected error while saving the figure to {figure_path}: {exception}")
    finally:
        mpl_plot.close(figure)


def animate_pngs_to_mp4(
    *,
    frames_dir: str | Path,
    mp4_path: str | Path,
    pattern: str = "frame_*.png",
    fps: int = 30,
    timeout_seconds: int = 60,
) -> None:
    """
    Combine PNG frames in `frames_dir` matching `pattern` into an MP4 at `mp4_path`.

    Requires `ffmpeg` on the system path. Creates the parent directory of `mp4_path` if needed.
    """
    frames_dir = Path(frames_dir)
    mp4_path = Path(mp4_path)
    manage_io.create_directory(
        directory=mp4_path.parent,
        verbose=False,
    )
    args = " ".join(
        [
            "-hide_banner",  # less stdout
            "-loglevel error",  # only errors
            "-y",  # overwrite output
            f"-framerate {fps}",  # input fps (put before -i)
            "-pattern_type glob",  # enable glob input (put before -i)
            f'-i "{pattern}"',  # input pattern (e.g., frame_*.png)
            '-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2"',  # enforce even dims
            "-c:v mpeg4 -q:v 3",  # codec + quality
            "-pix_fmt yuv420p",  # broad compatibility
            f"-r {fps}",  # output fps
        ],
    )
    cmd = f'ffmpeg {args} "{mp4_path}"'
    manage_shell.execute_shell_command(
        command=cmd,
        working_directory=frames_dir,
        timeout_seconds=timeout_seconds,
    )
    manage_log.log_action(
        title="Save animation",
        outcome=manage_log.ActionOutcome.SUCCESS,
        message="Saved mp4.",
        notes={"file": str(mp4_path)},
    )


## } MODULE
