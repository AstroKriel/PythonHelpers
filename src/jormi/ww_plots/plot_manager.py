## { MODULE

##
## === WORKSPACE SETUP
##

import matplotlib

matplotlib.use("Agg", force=True)

##
## === DEPENDENCIES
##

import numpy

from typing import TypeAlias, overload
from pathlib import Path
from numpy.typing import NDArray

from matplotlib import rcParams
from matplotlib import pyplot as mpl_plot
from matplotlib.axes import Axes as mpl_Axes
from matplotlib.figure import Figure as mpl_Figure

from jormi.ww_types import type_manager, cardinal_anchors
from jormi.ww_io import io_manager, shell_manager
from jormi.ww_plots import plot_styler

##
## === TYPE ALIASES
##

PlotAxis: TypeAlias = mpl_Axes
PlotAxesArray: TypeAlias = NDArray[numpy.object_]

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
    theme: plot_styler.Theme | str = plot_styler.Theme.LIGHT,
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
    theme: plot_styler.Theme | str = plot_styler.Theme.LIGHT,
) -> tuple[mpl_Figure, PlotAxesArray]:
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
    theme: plot_styler.Theme | str = plot_styler.Theme.LIGHT,
) -> tuple[mpl_Figure, PlotAxis | PlotAxesArray]:
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
        plot_styler.set_theme(theme=theme)
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
    type_manager.ensure_finite_int(
        param=num_rows,
        param_name="num_rows",
        require_positive=True,
    )
    type_manager.ensure_finite_int(
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
    axs_array: PlotAxesArray = numpy.asarray(axs, dtype=object)
    return fig, axs_array


##
## === AXIS HELPERS
##


def add_inset_axis(
    ax: PlotAxis,
    bounds: tuple[float, float, float, float] = (0.0, 1.0, 1.0, 0.5),
    x_label: str | None = None,
    y_label: str | None = None,
    fontsize: float | None = None,
    x_label_alignment: cardinal_anchors.VerticalAnchorLike = cardinal_anchors.VerticalAnchor.Top,
    y_label_alignment: cardinal_anchors.HorizontalAnchorLike = cardinal_anchors.HorizontalAnchor.Right,
) -> PlotAxis:
    """Add an inset Axis to `ax`."""
    x_label_anchor = cardinal_anchors.as_vertical_anchor(x_label_alignment)
    y_label_anchor = cardinal_anchors.as_horizontal_anchor(y_label_alignment)
    cardinal_anchors.ensure_vertical_edge_anchor(
        anchor=x_label_anchor,
        param_name="x_label_anchor",
    )
    cardinal_anchors.ensure_horizontal_edge_anchor(
        anchor=y_label_anchor,
        param_name="y_label_anchor",
    )
    ax_inset = ax.inset_axes(bounds)
    if fontsize is None:
        fontsize = rcParams["axes.labelsize"]
    x_label_anchor_value = x_label_anchor.value
    y_label_anchor_value = y_label_anchor.value
    ## asserts are necessary to keep static analysis happy (formally unreachable)
    assert x_label_anchor_value != "center"
    assert y_label_anchor_value != "center"
    if x_label is not None:
        ax_inset.set_xlabel(
            xlabel=x_label,
            fontsize=fontsize,
        )
        ax_inset.xaxis.set_label_position(x_label_anchor_value)
    if y_label is not None:
        ax_inset.set_ylabel(
            ylabel=y_label,
            fontsize=fontsize,
        )
        ax_inset.yaxis.set_label_position(y_label_anchor_value)
    ax_inset.tick_params(
        axis="x",
        labeltop=cardinal_anchors.is_top_edge(x_label_anchor),
        labelbottom=cardinal_anchors.is_bottom_edge(x_label_anchor),
        top=True,
        bottom=True,
    )
    ax_inset.tick_params(
        axis="y",
        labelleft=cardinal_anchors.is_left_edge(y_label_anchor),
        labelright=cardinal_anchors.is_right_edge(y_label_anchor),
        left=True,
        right=True,
    )
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
            print("Saved figure:", fig_path)
    except FileNotFoundError as exception:
        print(f"FileNotFoundError: {exception}")
    except PermissionError as exception:
        print(f"PermissionError: You do not have permission to save to: {fig_path}")
        print(f"Details: {exception}")
    except IOError as exception:
        print(f"IOError: An error occurred while trying to save the figure to: {fig_path}")
        print(f"Details: {exception}")
    except Exception as exception:
        print(f"Unexpected error while saving the figure to {fig_path}: {exception}")
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
    io_manager.init_directory(mp4_path.parent)
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
    shell_manager.execute_shell_command(
        command=cmd,
        working_directory=frames_dir,
        timeout_seconds=timeout_seconds,
    )
    print("Saved:", mp4_path)


## } MODULE
