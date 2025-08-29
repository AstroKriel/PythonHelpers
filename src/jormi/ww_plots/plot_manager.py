## START OF MODULE


## ###############################################################
## WORKSPACE SETUP
## ###############################################################

import matplotlib
matplotlib.use("Agg", force=True)


## ###############################################################
## DEPENDENCIES
## ###############################################################

from typing import Literal, Any
from pathlib import Path
from matplotlib import rcParams
from matplotlib import pyplot as mpl_plot
from matplotlib.axes import Axes as mpl_axes
from matplotlib.figure import Figure
from jormi.utils import list_utils
from jormi.ww_io import io_manager, shell_manager


## ###############################################################
## FUNCTIONS
## ###############################################################

def create_figure(
    num_rows   : int   = 1,
    num_cols   : int   = 1,
    fig_scale  : float = 1.0,
    axis_shape : tuple = (4, 6),
    x_spacing  : float = 0.05,
    y_spacing  : float = 0.05,
    share_x    : bool = False,
    share_y    : bool = False,
  ) -> tuple[Figure, Any]:
  """Initialize a figure with a flexible grid layout."""
  fig_width  = fig_scale * axis_shape[1] * num_cols
  fig_height = fig_scale * axis_shape[0] * num_rows
  fig, axs = mpl_plot.subplots(
    nrows   = num_rows,
    ncols   = num_cols,
    figsize = (fig_width, fig_height),
    sharex  = share_x,
    sharey  = share_y,
    squeeze = (num_rows == 1 or num_cols == 1),
  )
  fig.subplots_adjust(wspace=x_spacing, hspace=y_spacing)
  return fig, axs

def add_inset_axis(
    ax           : mpl_axes,
    bounds       : tuple[float, float, float, float] = (0.0, 1.0, 1.0, 0.5),
    x_label      : str | None = None,
    y_label      : str | None = None,
    fontsize     : float | None = None,
    x_label_side : Literal["bottom", "top"] = "top",
    y_label_side : Literal["left", "right"] = "right",
  ):
  valid_x_sides = [ "top", "bottom" ]
  valid_y_sides = [ "left", "right" ]
  if x_label_side not in valid_x_sides: raise ValueError(f"`x_label_side` = `{x_label_side}` is invalid. Choose from: {list_utils.cast_to_string(valid_x_sides)}")
  if y_label_side not in valid_y_sides: raise ValueError(f"`y_label_side` = `{y_label_side}` is invalid. Choose from: {list_utils.cast_to_string(valid_y_sides)}")
  ax_inset = ax.inset_axes(bounds)
  if fontsize is None: fontsize = rcParams["axes.labelsize"]
  if x_label is not None: ax_inset.set_xlabel(x_label, fontsize=fontsize)
  if y_label is not None: ax_inset.set_ylabel(y_label, fontsize=fontsize)
  ax_inset.xaxis.set_label_position(x_label_side)
  ax_inset.yaxis.set_label_position(y_label_side)
  ax_inset.tick_params(
    axis        = "x",
    labeltop    = (x_label_side == "top"),
    labelbottom = (x_label_side == "bottom"),
    top         = True,
    bottom      = True,
  )
  ax_inset.tick_params(
    axis        = "y",
    labelleft   = (y_label_side == "left"),
    labelright  = (y_label_side == "right"),
    left        = True,
    right       = True,
  )
  return ax_inset

def save_figure(
    fig,
    file_path : str | Path,
    dpi       : int = 200,
    verbose   : bool = True,
  ) -> None:
  if not str(file_path).endswith(".png") and not str(file_path).endswith(".pdf"):
    raise ValueError("Figures should end with .png or .pdf")
  try:
    fig.savefig(file_path, dpi=dpi)
    if verbose: print("Saved figure:", file_path)
  except FileNotFoundError as exception:
    print(f"FileNotFoundError: {exception}")
  except PermissionError as exception:
    print(f"PermissionError: You do not have permission to save to: {file_path}")
    print(f"Details: {exception}")
  except IOError as exception:
    print(f"IOError: An error occurred while trying to save the figure to: {file_path}")
    print(f"Details: {exception}")
  except Exception as exception:
    print(f"Unexpected error while saving the figure to {file_path}: {exception}")
  finally:
    mpl_plot.close(fig)

def animate_pngs_to_mp4(
    frames_dir      : str | Path,
    mp4_path        : str | Path,
    pattern         : str = "frame_%05d.png",
    fps             : int = 30,
    timeout_seconds : int = 60,
  ) -> None:
  frames_dir = Path(frames_dir)
  mp4_path   = Path(mp4_path)
  io_manager.init_directory(mp4_path.parent)
  cmd = (
    f'ffmpeg -hide_banner -loglevel error -y '
    f'-framerate {fps} -i {pattern} '
    f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
    f'-c:v mpeg4 -q:v 3 -pix_fmt yuv420p '
    f'-r {fps} "{mp4_path}"'
  )
  shell_manager.execute_shell_command(
    command           = cmd,
    working_directory = frames_dir,
    timeout_seconds   = timeout_seconds,
  )
  print("Saved:", mp4_path)


## END OF MODULE