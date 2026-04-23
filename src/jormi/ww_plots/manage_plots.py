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
from jormi.ww_types import (
    box_positions,
    check_types,
)

##
## === TYPE ALIASES
##

PlotAxis: TypeAlias = mpl_Axes
PlotAxesGrid: TypeAlias = NDArray[numpy.object_]

##
## === INTERNAL HELPERS
##


def _get_fig_shape(
    num_rows: int = 1,
    num_cols: int = 1,
    fig_scale: float = 1.0,
    axis_shape: tuple[float, float] = (4, 6),
) -> tuple[float, float]:
    """Compute figure size (inches) from per-panel axis_shape = (height, width)."""
    if (num_rows < 1) or (num_cols < 1):
        raise ValueError("`num_rows` and `num_cols` must both be >= 1.")
    fig_width = fig_scale * axis_shape[1] * num_cols
    fig_height = fig_scale * axis_shape[0] * num_rows
    return fig_width, fig_height


##
## === FIGURE FACTORY
##


@overload
def create_figure(
    *,
    num_rows: None = None,
    num_cols: None = None,
    fig_scale: float = 1.0,
    axis_shape: tuple[float, float] = (4, 6),
    x_spacing: float = 0.05,
    y_spacing: float = 0.05,
    share_x: bool = False,
    share_y: bool = False,
    auto_style: bool = True,
    theme: style_plots.Theme | str = style_plots.Theme.LIGHT,
) -> tuple[mpl_Figure, PlotAxis]:
    ...


@overload
def create_figure(
    *,
    num_rows: int,
    num_cols: int,
    fig_scale: float = 1.0,
    axis_shape: tuple[float, float] = (4, 6),
    x_spacing: float = 0.05,
    y_spacing: float = 0.05,
    share_x: bool = False,
    share_y: bool = False,
    auto_style: bool = True,
    theme: style_plots.Theme | str = style_plots.Theme.LIGHT,
) -> tuple[mpl_Figure, PlotAxesGrid]:
    ...


def create_figure(
    *,
    num_rows: int | None = None,
    num_cols: int | None = None,
    fig_scale: float = 1.0,
    axis_shape: tuple[float, float] = (4, 6),
    x_spacing: float = 0.05,
    y_spacing: float = 0.05,
    share_x: bool = False,
    share_y: bool = False,
    auto_style: bool = True,
    theme: style_plots.Theme | str = style_plots.Theme.LIGHT,
) -> tuple[mpl_Figure, PlotAxis | PlotAxesGrid]:
    """
    Create a Matplotlib figure and Axis / Axes grid.

    Overloads:
        - create_figure() -> (fig, axis)
        - create_figure(num_rows=N, num_cols=M) -> (fig, axs) with shape (N, M)

    Notes
    -----
    - If `num_rows` and `num_cols` are both None (or omitted), a single-panel
      figure is created and a single Axis is returned.
    - If `num_rows` and `num_cols` are both provided as integers, a grid of
      axes is created and a 2D object-dtype array of Axes is returned.
    - Mixed None/int specifications are not allowed.
    """
    if auto_style:
        style_plots.set_theme(theme=theme)
    if (num_rows is None) and (num_cols is None):
        fig_width, fig_height = _get_fig_shape(
            num_rows=1,
            num_cols=1,
            fig_scale=fig_scale,
            axis_shape=axis_shape,
        )
        fig, ax = mpl_plot.subplots(
            nrows=1,
            ncols=1,
            figsize=(fig_width, fig_height),
            sharex=share_x,
            sharey=share_y,
            squeeze=True,
        )
        return fig, ax
    if (num_rows is None) or (num_cols is None):
        raise ValueError(
            "Either specify both `num_rows` and `num_cols`, or neither."
            " Mixed None/int combinations are not supported.",
        )
    check_types.ensure_finite_int(
        param=num_rows,
        param_name="num_rows",
        require_positive=True,
    )
    check_types.ensure_finite_int(
        param=num_cols,
        param_name="num_cols",
        require_positive=True,
    )
    if (num_rows == 1) and (num_cols == 1):
        raise ValueError(
            "For a single-panel figure, omit `num_rows` and `num_cols` so that"
            " a single Axis is returned instead of a 1x1 Axes grid.",
        )
    fig_width, fig_height = _get_fig_shape(
        num_rows=num_rows,
        num_cols=num_cols,
        fig_scale=fig_scale,
        axis_shape=axis_shape,
    )
    fig, axs = mpl_plot.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(fig_width, fig_height),
        sharex=share_x,
        sharey=share_y,
        squeeze=False,
    )
    fig.subplots_adjust(
        wspace=x_spacing,
        hspace=y_spacing,
    )
    axs_grid: PlotAxesGrid = numpy.asarray(axs, dtype=object)
    return fig, axs_grid


def create_figure_grid(
    *,
    num_rows: int = 1,
    num_cols: int = 1,
    fig_scale: float = 1.0,
    axis_shape: tuple[float, float] = (4, 6),
    x_spacing: float = 0.05,
    y_spacing: float = 0.05,
    share_x: bool = False,
    share_y: bool = False,
    auto_style: bool = True,
    theme: style_plots.Theme | str = style_plots.Theme.LIGHT,
) -> tuple[mpl_Figure, PlotAxesGrid]:
    """
    Like `create_figure`, but always returns a 2D axes grid of shape (num_rows, num_cols), so
    callers can always index axes as axs_grid[row, col].
    """
    if (num_rows == 1) and (num_cols == 1):
        fig, ax = create_figure(
            fig_scale=fig_scale,
            axis_shape=axis_shape,
            auto_style=auto_style,
            theme=theme,
        )
        axs_grid: PlotAxesGrid = numpy.asarray([[ax]], dtype=object)
        return fig, axs_grid
    fig, axs_grid = create_figure(
        num_rows=num_rows,
        num_cols=num_cols,
        fig_scale=fig_scale,
        axis_shape=axis_shape,
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

_Side = box_positions.TypeHints.Box.Side


@dataclass(frozen=True)
class AxisBounds:
    x_min: float
    y_min: float
    x_width: float
    y_width: float


def compute_adjacent_ax_bounds(
    ax: PlotAxis,
    side: _Side = box_positions.TypeHints.Box.Side.Right,
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
        raise ValueError(f"Unexpected side: {side!r}")  # pyright: ignore[reportUnreachable]


def add_inset_axis(
    ax: PlotAxis,
    bounds: tuple[float, float, float, float] = (0.0, 1.0, 1.0, 0.5),
    x_label: str | None = None,
    y_label: str | None = None,
    fontsize: float | None = None,
    x_label_alignment: box_positions.TypeHints.PositionLike = box_positions.TypeHints.Box.Side.Top,
    y_label_alignment: box_positions.TypeHints.PositionLike = box_positions.TypeHints.Box.Side.Right,
) -> PlotAxis:
    """Add an inset Axis to `ax`."""
    x_label_side = box_positions.as_box_side(x_label_alignment)
    y_label_side = box_positions.as_box_side(y_label_alignment)
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
        labeltop=(x_label_side is box_positions.TypeHints.Box.Side.Top),
        labelbottom=(x_label_side is box_positions.TypeHints.Box.Side.Bottom),
        top=True,
        bottom=True,
    )
    if x_label_side is box_positions.TypeHints.Box.Side.Top:
        ax_inset.xaxis.tick_top()
    ax_inset.tick_params(
        axis="y",
        labelleft=(y_label_side is box_positions.TypeHints.Box.Side.Left),
        labelright=(y_label_side is box_positions.TypeHints.Box.Side.Right),
        left=True,
        right=True,
    )
    if y_label_side is box_positions.TypeHints.Box.Side.Right:
        ax_inset.yaxis.tick_right()
    return ax_inset


##
## === IO HELPERS
##


def save_figure(
    fig: mpl_Figure,
    fig_path: str | Path,
    dpi: int = 200,
    verbose: bool = True,
) -> None:
    if not str(fig_path).endswith(".png") and not str(fig_path).endswith(".pdf"):
        raise ValueError("Figures should end with .png or .pdf")
    try:
        fig.savefig(fig_path, dpi=dpi)
        if verbose:
            manage_log.log_action(
                title="Save figure",
                outcome=manage_log.ActionOutcome.SUCCESS,
                message="Saved figure.",
                notes={"file": str(fig_path)},
            )
    except FileNotFoundError as exception:
        manage_log.log_error(f"FileNotFoundError: {exception}")
    except PermissionError as exception:
        manage_log.log_error(
            f"PermissionError: You do not have permission to save to: {fig_path}",
            notes={"details": str(exception)},
        )
    except IOError as exception:
        manage_log.log_error(
            f"IOError: An error occurred while trying to save the figure to: {fig_path}",
            notes={"details": str(exception)},
        )
    except Exception as exception:
        manage_log.log_error(f"Unexpected error while saving the figure to {fig_path}: {exception}")
    finally:
        mpl_plot.close(fig)


def animate_pngs_to_mp4(
    frames_dir: str | Path,
    mp4_path: str | Path,
    pattern: str = "frame_*.png",
    fps: int = 30,
    timeout_seconds: int = 60,
) -> None:
    frames_dir = Path(frames_dir)
    mp4_path = Path(mp4_path)
    manage_io.create_directory(mp4_path.parent, verbose=False)
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
