## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
from pathlib import Path
from typing import cast, Union
from numpy.typing import NDArray
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import pyplot as mpl_plot
from jormi.ww_plots.plot_styler import *


## ###############################################################
## WORKSPACE SETUP
## ###############################################################
mpl_plot.switch_backend("agg")


## ###############################################################
## TYPES
## ###############################################################

AxesGrid = NDArray[numpy.object_]
AxesLike = Union[Axes, AxesGrid]


## ###############################################################
## FUNCTIONS
## ###############################################################

def cast_to_axis(ax: AxesLike) -> Axes:
  if isinstance(ax, Axes):
    return ax
  elif isinstance(ax, numpy.ndarray):
    if ax.size == 1: return ax.item()
    raise TypeError(f"Expected a single Axes, but got an array with shape {ax.shape}")
  else: raise TypeError(f"Unsupported type for AxesLike: {type(ax)!r}")

def create_figure(
    num_rows   : int   = 1,
    num_cols   : int   = 1,
    fig_scale  : float = 1.0,
    axis_shape : tuple = (4, 6),
    x_spacing  : float = 0.05,
    y_spacing  : float = 0.05,
    share_x    : bool = False,
    share_y    : bool = False,
  ) -> tuple[Figure, Union[Axes, numpy.ndarray]]:
  """Initialize a figure with a flexible grid layout."""
  fig_width  = fig_scale * axis_shape[1] * num_cols
  fig_height = fig_scale * axis_shape[0] * num_rows
  fig, axs = mpl_plot.subplots(
    nrows   = num_rows,
    ncols   = num_cols,
    figsize = (fig_width, fig_height),
    sharex  = share_x,
    sharey  = share_y,
    squeeze = (num_rows == 1 and num_cols == 1),
  )
  fig.subplots_adjust(wspace=x_spacing, hspace=y_spacing)
  return fig, axs

def save_figure(
    fig,
    file_path : str | Path,
    draft     : bool = False,
    verbose   : bool = True
  ) -> None:
  if not str(file_path).endswith(".png") and not str(file_path).endswith(".pdf"):
    raise ValueError("Figures should end with .png or .pdf")
  dpi = 100 if draft else 200
  try:
    fig.savefig(file_path, dpi=dpi)
    mpl_plot.close(fig)
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


## END OF MODULE