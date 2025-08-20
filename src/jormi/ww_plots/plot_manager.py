## START OF MODULE


## ###############################################################
## WORKSPACE SETUP
## ###############################################################
import matplotlib
matplotlib.use("Agg", force=True)


## ###############################################################
## DEPENDENCIES
## ###############################################################

from typing import Any
from pathlib import Path
from matplotlib import pyplot as mpl_plot
from matplotlib.figure import Figure
from jormi.ww_io import io_manager, shell_manager
from jormi.ww_plots.plot_styler import *


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

def save_figure(
    fig,
    file_path : str | Path,
    draft     : bool = False,
    verbose   : bool = True,
  ) -> None:
  if not str(file_path).endswith(".png") and not str(file_path).endswith(".pdf"):
    raise ValueError("Figures should end with .png or .pdf")
  dpi = 100 if draft else 200
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