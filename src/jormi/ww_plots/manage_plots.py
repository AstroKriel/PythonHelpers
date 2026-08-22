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

PlotAxis: TypeAlias = mpl_Axes
PlotAxesGrid: TypeAlias = NDArray[numpy.object_]

##
## === BOX SHAPE
##


@dataclass(
    frozen=True,
    kw_only=True,
)
class BoxShape:
    """
    Width and height of a rectangle, in inches: a whole figure, or the share it gives
    one axis.

    Named rather than a bare pair, so which of the two is the height never has to be
    remembered; `as_mpl_shape` produces the pair for handing straight to Matplotlib.
    """

    width: float
    height: float

    def __post_init__(self) -> None:
        for name in (
            "width",
            "height",
        ):
            value = getattr(self, name)
            if not (value > 0):
                raise ValueError(f"`{name}` must be positive, but got {value}.")

    @property
    def as_mpl_shape(self) -> tuple[float, float]:
        """The pair, width first, as Matplotlib's `figsize` reads it."""
        return self.width, self.height

    @property
    def aspect_ratio(self) -> float:
        """Width over height."""
        return self.width / self.height


##
## === INTERNAL HELPERS
##

DEFAULT_AXIS_SHAPE: BoxShape = BoxShape(
    width=6.0,
    height=4.0,
)


def _compute_figure_shape(
    *,
    num_rows: int = 1,
    num_cols: int = 1,
    figure_scale: float = 1.0,
    axis_shape: BoxShape = DEFAULT_AXIS_SHAPE,
) -> BoxShape:
    """Compute figure size (inches) from the share each axis gets."""
    if (num_rows < 1) or (num_cols < 1):
        raise ValueError("`num_rows` and `num_cols` must both be >= 1.")
    return BoxShape(
        width=figure_scale * axis_shape.width * num_cols,
        height=figure_scale * axis_shape.height * num_rows,
    )


def _place_axes_in_figure(
    *,
    fig: mpl_Figure,
    margins: style_plots.FigureMargins,
    x_spacing: float | None = None,
    y_spacing: float | None = None,
) -> None:
    """
    Place the axes within `fig`, leaving `margins` clear around them.

    Margins are in points, while Matplotlib places axes as fractions of the figure, so
    the figure's own size is what converts between the two.
    """
    width_points, height_points = (
        float(length_inches) * style_plots.POINTS_PER_INCH for length_inches in fig.get_size_inches()
    )
    if (margins.left_margin + margins.right_margin) >= width_points:
        raise ValueError(
            f"`left_margin` + `right_margin` ({margins.left_margin + margins.right_margin} pt)"
            f" leave no room for the axes in a figure {width_points:.1f} pt wide.",
        )
    if (margins.bottom_margin + margins.top_margin) >= height_points:
        raise ValueError(
            f"`bottom_margin` + `top_margin` ({margins.bottom_margin + margins.top_margin} pt)"
            f" leave no room for the axes in a figure {height_points:.1f} pt tall.",
        )
    fig.subplots_adjust(
        left=margins.left_margin / width_points,
        right=1.0 - (margins.right_margin / width_points),
        bottom=margins.bottom_margin / height_points,
        top=1.0 - (margins.top_margin / height_points),
    )
    if (x_spacing is not None) and (y_spacing is not None):
        fig.subplots_adjust(
            wspace=x_spacing,
            hspace=y_spacing,
        )


def _split_width_across_axes(
    *,
    width_inches: float,
    num_cols: int,
    aspect_ratio: float,
) -> BoxShape:
    """
    Share a figure's width between the axes in a row, giving the share each one gets.

    `aspect_ratio` is the width / height of that share; an axis is drawn smaller than
    its share, by whatever the margins hold.
    """
    if num_cols < 1:
        raise ValueError(f"`num_cols` must be >= 1, but got {num_cols}.")
    if not (aspect_ratio > 0):
        raise ValueError(f"`aspect_ratio` must be positive, but got {aspect_ratio}.")
    axis_width_inches = width_inches / num_cols
    return BoxShape(
        height=axis_width_inches / aspect_ratio,
        width=axis_width_inches,
    )


def _get_figure_shape(
    *,
    num_rows: int,
    num_cols: int,
    figure_scale: float,
    axis_shape: BoxShape | None,
    figure_layout: style_plots.FigureLayout | None,
    aspect_ratio: float | None,
) -> tuple[BoxShape, style_plots.FigureLayout]:
    """
    Size a figure from a page layout, or in its own terms, but not both.

    A layout pins the figure to a share of the page, so adding axes makes each one
    smaller. `axis_shape` instead gives each axis a fixed size, so the figure grows as
    axes are added, and the figure is no longer tied to a page.
    """
    active_layout = style_plots.get_figure_layout() if (figure_layout is None) else figure_layout
    if axis_shape is None:
        return (
            _compute_figure_shape(
                num_rows=num_rows,
                num_cols=num_cols,
                axis_shape=_split_width_across_axes(
                    width_inches=active_layout.width.width_inches,
                    num_cols=num_cols,
                    aspect_ratio=DEFAULT_AXIS_SHAPE.aspect_ratio if (aspect_ratio is None) else aspect_ratio,
                ),
            ),
            active_layout,
        )
    if figure_layout is not None:
        raise ValueError(
            "`figure_layout` and `axis_shape` are mutually exclusive: a layout sizes the"
            " figure to a share of the page, while `axis_shape` sizes each axis outright.",
        )
    if aspect_ratio is not None:
        raise ValueError(
            "`aspect_ratio` only applies when a figure is sized to a page;"
            " with `axis_shape` the shape of each axis is already set by it.",
        )
    return (
        _compute_figure_shape(
            num_rows=num_rows,
            num_cols=num_cols,
            figure_scale=figure_scale,
            axis_shape=axis_shape,
        ),
        active_layout,
    )


##
## === FIGURE FACTORY
##


@overload
def create_figure(
    *,
    num_rows: None = None,
    num_cols: None = None,
    figure_scale: float = 1.0,
    axis_shape: BoxShape | None = None,
    figure_layout: style_plots.FigureLayout | None = None,
    aspect_ratio: float | None = None,
    x_spacing: float = 0.05,
    y_spacing: float = 0.05,
    share_x: bool = False,
    share_y: bool = False,
    auto_style: bool = True,
    theme: style_plots.Theme | str | None = None,
) -> tuple[mpl_Figure, PlotAxis]:
    ...


@overload
def create_figure(
    *,
    num_rows: int,
    num_cols: int,
    figure_scale: float = 1.0,
    axis_shape: BoxShape | None = None,
    figure_layout: style_plots.FigureLayout | None = None,
    aspect_ratio: float | None = None,
    x_spacing: float = 0.05,
    y_spacing: float = 0.05,
    share_x: bool = False,
    share_y: bool = False,
    auto_style: bool = True,
    theme: style_plots.Theme | str | None = None,
) -> tuple[mpl_Figure, PlotAxesGrid]:
    ...


def create_figure(
    *,
    num_rows: int | None = None,
    num_cols: int | None = None,
    figure_scale: float = 1.0,
    axis_shape: BoxShape | None = None,
    figure_layout: style_plots.FigureLayout | None = None,
    aspect_ratio: float | None = None,
    x_spacing: float = 0.05,
    y_spacing: float = 0.05,
    share_x: bool = False,
    share_y: bool = False,
    auto_style: bool = True,
    theme: style_plots.Theme | str | None = None,
) -> tuple[mpl_Figure, PlotAxis | PlotAxesGrid]:
    """
    Create a Matplotlib figure and Axis / Axes grid.

    Overloads:
        - create_figure() -> (fig, axis)
        - create_figure(num_rows=N, num_cols=M) -> (fig, axs) with shape (N, M)

    Sizing
    ------
    `axis_shape` is the share of the figure given to one axis, in inches, so the figure
    comes out `axis_shape` times the grid. It is not the size the axis is drawn at: tick
    labels and axis labels are held in margins taken out of that share, so the axis itself
    is drawn smaller. A colorbar is placed beyond the axis rather than within those
    margins, and so can reach past the figure edge.

    Notes
    -----
    - If `num_rows` and `num_cols` are both None (or omitted), a single-panel
      figure is created and a single Axis is returned.
    - If `num_rows` and `num_cols` are both provided as integers, a grid of
      axes is created and a 2D object-dtype array of Axes is returned.
    - Mixed None/int specifications are not allowed.
    """
    if auto_style and (theme is not None):
        style_plots.set_theme(theme=theme)
    if (num_rows is None) and (num_cols is None):
        figure_shape, active_layout = _get_figure_shape(
            num_rows=1,
            num_cols=1,
            figure_scale=figure_scale,
            axis_shape=axis_shape,
            figure_layout=figure_layout,
            aspect_ratio=aspect_ratio,
        )
        fig, ax = mpl_plot.subplots(
            nrows=1,
            ncols=1,
            figsize=figure_shape.as_mpl_shape,
            sharex=share_x,
            sharey=share_y,
            squeeze=True,
        )
        _place_axes_in_figure(
            fig=fig,
            margins=active_layout.margins,
        )
        return fig, ax
    if (num_rows is None) or (num_cols is None):
        raise ValueError(
            "Either specify both `num_rows` and `num_cols`, or neither."
            " Mixed None/int combinations are not supported.",
        )
    validate_types.ensure_finite_int(
        param=num_rows,
        param_name="num_rows",
        require_positive=True,
    )
    validate_types.ensure_finite_int(
        param=num_cols,
        param_name="num_cols",
        require_positive=True,
    )
    if (num_rows == 1) and (num_cols == 1):
        raise ValueError(
            "For a single-panel figure, omit `num_rows` and `num_cols` so that"
            " a single Axis is returned instead of a 1x1 Axes grid.",
        )
    figure_shape, active_layout = _get_figure_shape(
        num_rows=num_rows,
        num_cols=num_cols,
        figure_scale=figure_scale,
        axis_shape=axis_shape,
        figure_layout=figure_layout,
        aspect_ratio=aspect_ratio,
    )
    fig, axs = mpl_plot.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=figure_shape.as_mpl_shape,
        sharex=share_x,
        sharey=share_y,
        squeeze=False,
    )
    _place_axes_in_figure(
        fig=fig,
        margins=active_layout.margins,
        x_spacing=x_spacing,
        y_spacing=y_spacing,
    )
    axs_grid: PlotAxesGrid = numpy.asarray(axs, dtype=object)
    return fig, axs_grid


def create_figure_grid(
    *,
    num_rows: int = 1,
    num_cols: int = 1,
    figure_scale: float = 1.0,
    axis_shape: BoxShape | None = None,
    figure_layout: style_plots.FigureLayout | None = None,
    aspect_ratio: float | None = None,
    x_spacing: float = 0.05,
    y_spacing: float = 0.05,
    share_x: bool = False,
    share_y: bool = False,
    auto_style: bool = True,
    theme: style_plots.Theme | str | None = None,
) -> tuple[mpl_Figure, PlotAxesGrid]:
    """
    Like `create_figure`, but always returns a 2D axes grid of shape (num_rows, num_cols), so
    callers can always index axes as axs_grid[row, col].
    """
    if (num_rows == 1) and (num_cols == 1):
        fig, ax = create_figure(
            figure_scale=figure_scale,
            axis_shape=axis_shape,
            figure_layout=figure_layout,
            aspect_ratio=aspect_ratio,
            auto_style=auto_style,
            theme=theme,
        )
        axs_grid: PlotAxesGrid = numpy.asarray([[ax]], dtype=object)
        return fig, axs_grid
    fig, axs_grid = create_figure(
        num_rows=num_rows,
        num_cols=num_cols,
        figure_scale=figure_scale,
        axis_shape=axis_shape,
        figure_layout=figure_layout,
        aspect_ratio=aspect_ratio,
        x_spacing=x_spacing,
        y_spacing=y_spacing,
        share_x=share_x,
        share_y=share_y,
        auto_style=auto_style,
        theme=theme,
    )
    return fig, axs_grid


##
## === AXIS HELPERS
##

_Side = box_positions.Positions.Side


@dataclass(frozen=True)
class AxisBounds:
    """Bounding box for an axis in figure coordinates."""

    x_min: float
    y_min: float
    x_width: float
    y_width: float


def compute_adjacent_ax_bounds(
    *,
    ax: PlotAxis,
    side: _Side = box_positions.Positions.Side.Right,
    gap: float = 0.1,
    thickness: float = 1.0,
    length: float = 1.0,
) -> AxisBounds:
    """Compute figure bounds for an axis placed adjacent to `ax`.

    The new axis sits on the `side` of `ax`, offset by `gap` (in figure coordinates).
    `thickness` sets its extent perpendicular to `side`, as a fraction of `ax`'s
    corresponding dimension. `length` sets its span parallel to `side`, also as a
    fraction, centered on `ax`'s edge.
    """
    box = ax.get_position()
    if side in (_Side.Left, _Side.Right):
        x_width = box.width * thickness
        y_width = box.height * length
        if side == _Side.Right:
            return AxisBounds(
                x_min=box.x1 + gap,
                y_min=box.y0 + (box.height - y_width) / 2,
                x_width=x_width,
                y_width=y_width,
            )
        else:
            return AxisBounds(
                x_min=box.x0 - x_width - gap,
                y_min=box.y0 + (box.height - y_width) / 2,
                x_width=x_width,
                y_width=y_width,
            )
    elif side in (_Side.Top, _Side.Bottom):
        x_width = box.width * length
        y_width = box.height * thickness
        if side == _Side.Top:
            return AxisBounds(
                x_min=box.x0 + (box.width - x_width) / 2,
                y_min=box.y1 + gap,
                x_width=x_width,
                y_width=y_width,
            )
        else:
            return AxisBounds(
                x_min=box.x0 + (box.width - x_width) / 2,
                y_min=box.y0 - y_width - gap,
                x_width=x_width,
                y_width=y_width,
            )
    else:
        raise ValueError(f"unexpected side: {side!r}.")  # pyright: ignore[reportUnreachable]


def add_inset_axis(
    *,
    ax: PlotAxis,
    bounds: tuple[float, float, float, float] = (0.0, 1.0, 1.0, 0.5),
    x_label: str | None = None,
    y_label: str | None = None,
    fontsize: float | None = None,
    x_label_alignment: box_positions.Positions.PositionLike = box_positions.Positions.Side.Top,
    y_label_alignment: box_positions.Positions.PositionLike = box_positions.Positions.Side.Right,
) -> PlotAxis:
    """Add an inset Axis to `ax`."""
    x_label_side = validate_box_positions.as_box_side(x_label_alignment)
    y_label_side = validate_box_positions.as_box_side(y_label_alignment)
    ax_inset = ax.inset_axes(bounds)
    if fontsize is None:
        fontsize = rcParams["axes.labelsize"]
    if x_label is not None:
        ax_inset.set_xlabel(
            xlabel=x_label,
            fontsize=fontsize,
        )
        ax_inset.xaxis.set_label_position(x_label_side.value)  # pyright: ignore[reportArgumentType]
    if y_label is not None:
        ax_inset.set_ylabel(
            ylabel=y_label,
            fontsize=fontsize,
        )
        ax_inset.yaxis.set_label_position(y_label_side.value)  # pyright: ignore[reportArgumentType]
    ax_inset.tick_params(
        axis="x",
        labeltop=(x_label_side is box_positions.Positions.Side.Top),
        labelbottom=(x_label_side is box_positions.Positions.Side.Bottom),
        top=True,
        bottom=True,
    )
    if x_label_side is box_positions.Positions.Side.Top:
        ax_inset.xaxis.tick_top()
    ax_inset.tick_params(
        axis="y",
        labelleft=(y_label_side is box_positions.Positions.Side.Left),
        labelright=(y_label_side is box_positions.Positions.Side.Right),
        left=True,
        right=True,
    )
    if y_label_side is box_positions.Positions.Side.Right:
        ax_inset.yaxis.tick_right()
    return ax_inset


##
## === IO HELPERS
##


def save_figure(
    *,
    fig: mpl_Figure,
    figure_path: str | Path,
    dpi: int = 200,
    verbose: bool = True,
) -> None:
    """
    Save `fig` to `figure_path`; close it.

    Accepts `.png` or `.pdf` paths; errors are logged rather than raised.
    """
    if not str(figure_path).endswith(".png") and not str(figure_path).endswith(".pdf"):
        raise ValueError("figures must end with `.png` or `.pdf`.")
    try:
        fig.savefig(figure_path, dpi=dpi)
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
        mpl_plot.close(fig)


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
