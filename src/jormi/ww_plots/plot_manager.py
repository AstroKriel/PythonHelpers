##
## === WORKSPACE SETUP
##

import matplotlib

matplotlib.use("Agg", force=True)

##
## === DEPENDENCIES
##

import numpy
from typing import TypeAlias, Iterator, Literal
from pathlib import Path
from dataclasses import dataclass
from numpy.typing import NDArray
from matplotlib import rcParams
from matplotlib import pyplot as mpl_plot
from matplotlib.axes import Axes as mpl_Axes
from matplotlib.figure import Figure as mpl_Figure
from jormi.utils import list_utils
from jormi.ww_io import io_manager, shell_manager
from jormi.ww_plots import plot_styler

##
## === AXES TYPES
##

Axis: TypeAlias = mpl_Axes


@dataclass(frozen=True)
class _AxesBase:
    """
    Base wrapper around an object-dtype numpy array holding mpl Axes.
    """
    ndarray: NDArray[object]

    def __len__(
        self,
    ) -> int:
        return len(self.ndarray)

    def __iter__(
        self,
    ) -> Iterator:
        return iter(self.ndarray)

    def _ensure_mpl_axes(
        self,
    ) -> None:
        if self.ndarray.dtype != object:
            raise TypeError(f"Axes arrays must have dtype=object; got {self.ndarray.dtype}")
        for elem in self.ndarray.flat:
            if not isinstance(elem, Axis):
                raise TypeError("All elements must be matplotlib Axes objects.")

    @staticmethod
    def _validate_index(
        elem_index: int,
        num_elems: int,
        axis_name: Literal["row", "col"],
    ) -> int:
        if not isinstance(elem_index, (int, numpy.integer)):
            raise TypeError(f"{axis_name} must be an int, got {type(elem_index).__name__}.")
        elem_index = int(elem_index)
        if elem_index < 0:
            elem_index += num_elems
        if (elem_index < 0) or (num_elems <= elem_index):
            raise IndexError(f"{axis_name} {elem_index} out of bounds for size={num_elems}.")
        return elem_index


@dataclass(frozen=True)
class AxesRow(_AxesBase):
    """
    1D array of matplotlib-Axes with shape (num_cols,).
    """

    def __post_init__(
        self,
    ) -> None:
        if self.ndarray.ndim != 1:
            raise ValueError(
                f"AxesRow expects 1D, got {self.ndarray.ndim}D, shape={self.ndarray.shape}.",
            )
        self._ensure_mpl_axes()

    def get_ax(
        self,
        col_index: int,
    ) -> Axis:
        valid_index = self._validate_index(
            elem_index=col_index,
            num_elems=len(self),
            axis_name="col",
        )
        return self.ndarray[valid_index]

    def __getitem__(
        self,
        col_index: int,
    ) -> Axis:
        return self.get_ax(col_index)


@dataclass(frozen=True)
class AxesCol(_AxesBase):
    """
    1D array of matplotlib-Axes with shape (num_rows,).
    """

    def __post_init__(
        self,
    ) -> None:
        if self.ndarray.ndim != 1:
            raise ValueError(
                f"AxesCol expects 1D, got {self.ndarray.ndim}D, shape={self.ndarray.shape}.",
            )
        self._ensure_mpl_axes()

    def get_ax(
        self,
        row_index: int,
    ) -> Axis:
        valid_index = self._validate_index(
            elem_index=row_index,
            num_elems=len(self),
            axis_name="row",
        )
        return self.ndarray[valid_index]

    def __getitem__(
        self,
        row_index: int,
    ) -> Axis:
        return self.get_ax(row_index)


@dataclass(frozen=True)
class AxesGrid(_AxesBase):
    """
    2D array of matplotlib-Axes with shape (num_rows, num_cols).
    """

    def __post_init__(
        self,
    ) -> None:
        if self.ndarray.ndim != 2:
            raise ValueError(
                f"AxesGrid expects 2D, got {self.ndarray.ndim}D, shape={self.ndarray.shape}.",
            )
        self._ensure_mpl_axes()

    @property
    def shape(
        self,
    ) -> tuple[int, int]:
        return self.ndarray.shape

    @property
    def num_rows(
        self,
    ) -> int:
        return self.ndarray.shape[0]

    @property
    def num_cols(
        self,
    ) -> int:
        return self.ndarray.shape[1]

    def get_row(
        self,
        row_index: int = 0,
    ) -> AxesRow:
        valid_row_index = self._validate_index(
            elem_index=row_index,
            num_elems=self.num_rows,
            axis_name="row",
        )
        return AxesRow(self.ndarray[valid_row_index, :])

    def get_col(
        self,
        col_index: int = 0,
    ) -> AxesCol:
        valid_col_index = self._validate_index(
            elem_index=col_index,
            num_elems=self.num_cols,
            axis_name="col",
        )
        return AxesCol(self.ndarray[:, valid_col_index])

    def get_ax(
        self,
        row_index: int,
        col_index: int,
    ) -> Axis:
        valid_row_index = self._validate_index(
            elem_index=row_index,
            num_elems=self.num_rows,
            axis_name="row",
        )
        valid_col_index = self._validate_index(
            elem_index=col_index,
            num_elems=self.num_cols,
            axis_name="col",
        )
        elem = self.ndarray[valid_row_index, valid_col_index]
        if not isinstance(elem, Axis):
            raise TypeError("Element is not a matplotlib Axes.")
        return elem

    def __getitem__(
        self,
        key,
    ) -> Axis | AxesRow | AxesCol:
        """
        Supports:
            grid[i, j] -> Axis
            grid[i, :] -> AxesRow
            grid[:, j] -> AxesCol
        """
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("AxesGrid expects 2D indexing: grid[row_index, col_index].")
        valid_types = (int, numpy.integer)
        ## grid[i, j] -> Axis
        if isinstance(key[0], valid_types) and isinstance(key[1], valid_types):
            return self.get_ax(key[0], key[1])
        ## grid[i, :] -> AxesRow
        if isinstance(key[0], valid_types) and (key[1] is slice(None)):
            return self.get_row(key[0])
        ## grid[:, j] -> AxesCol
        if (key[0] is slice(None)) and isinstance(key[1], valid_types):
            return self.get_col(key[1])
        raise TypeError("Only scalar indices or full slices are supported: [i, j], [i, :], [:, j].")


##
## === FUNCTIONS
##


def _get_fig_shape(
    num_rows: int = 1,
    num_cols: int = 1,
    fig_scale: float = 1.0,
    axis_shape: tuple[float, float] = (4, 6),
) -> tuple[float, float]:
    """
    Compute figure size (inches) from per-panel axis_shape = (height, width).
    """
    if num_rows < 1 or num_cols < 1:
        raise ValueError("num_rows and num_cols must both be >= 1.")
    fig_width = fig_scale * axis_shape[1] * num_cols
    fig_height = fig_scale * axis_shape[0] * num_rows
    return fig_width, fig_height


def create_ax(
    fig_scale: float = 1.0,
    axis_shape: tuple[float, float] = (4, 6),
    auto_style: bool = True,
    theme: plot_styler.Theme | str = plot_styler.Theme.LIGHT,
) -> tuple[mpl_Figure, Axis]:
    """
    Create a single-panel figure and return (fig, ax).
    """
    if auto_style:
        plot_styler.set_theme(theme=theme)
    fig_width, fig_height = _get_fig_shape(fig_scale=fig_scale, axis_shape=axis_shape)
    fig, ax = mpl_plot.subplots(figsize=(fig_width, fig_height))
    return fig, ax


def create_axs_grid(
    num_rows: int = 1,
    num_cols: int = 1,
    fig_scale: float = 1.0,
    axis_shape: tuple[float, float] = (4, 6),
    x_spacing: float = 0.05,
    y_spacing: float = 0.05,
    share_x: bool = False,
    share_y: bool = False,
    auto_style: bool = True,
    theme: plot_styler.Theme | str = plot_styler.Theme.LIGHT,
) -> tuple[mpl_Figure, AxesGrid]:
    """
    Create a figure with a flexible grid layout; always returns a 2D AxesGrid.
    """
    if (num_rows == 1) and (num_cols == 1):
        raise ValueError("You requested a single-panel figure (1x1). Use `create_ax(...)` instead.")
    if auto_style:
        plot_styler.set_theme(theme=theme)
    fig_width, fig_height = _get_fig_shape(
        num_rows=num_rows,
        num_cols=num_cols,
        fig_scale=fig_scale,
        axis_shape=axis_shape,
    )
    fig, numpy_grid = mpl_plot.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(fig_width, fig_height),
        sharex=share_x,
        sharey=share_y,
        squeeze=False,
    )
    fig.subplots_adjust(wspace=x_spacing, hspace=y_spacing)
    return fig, AxesGrid(numpy_grid)


def add_inset_axis(
    ax: Axis,
    bounds: tuple[float, float, float, float] = (0.0, 1.0, 1.0, 0.5),
    x_label: str | None = None,
    y_label: str | None = None,
    fontsize: float | None = None,
    x_label_side: Literal["bottom", "top"] = "top",
    y_label_side: Literal["left", "right"] = "right",
):
    valid_x_sides = ["top", "bottom"]
    valid_y_sides = ["left", "right"]
    if x_label_side not in valid_x_sides:
        raise ValueError(
            f"`x_label_side` = `{x_label_side}` is invalid. Choose from: {list_utils.cast_to_string(valid_x_sides)}",
        )
    if y_label_side not in valid_y_sides:
        raise ValueError(
            f"`y_label_side` = `{y_label_side}` is invalid. Choose from: {list_utils.cast_to_string(valid_y_sides)}",
        )
    ax_inset = ax.inset_axes(bounds)
    if fontsize is None: fontsize = rcParams["axes.labelsize"]
    if x_label is not None: ax_inset.set_xlabel(x_label, fontsize=fontsize)
    if y_label is not None: ax_inset.set_ylabel(y_label, fontsize=fontsize)
    ax_inset.xaxis.set_label_position(x_label_side)
    ax_inset.yaxis.set_label_position(y_label_side)
    ax_inset.tick_params(
        axis="x",
        labeltop=(x_label_side == "top"),
        labelbottom=(x_label_side == "bottom"),
        top=True,
        bottom=True,
    )
    ax_inset.tick_params(
        axis="y",
        labelleft=(y_label_side == "left"),
        labelright=(y_label_side == "right"),
        left=True,
        right=True,
    )
    return ax_inset


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
        if verbose: print("Saved figure:", fig_path)
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
