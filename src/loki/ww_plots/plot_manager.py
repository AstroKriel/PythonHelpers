## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import matplotlib.pyplot as mpl_plot
from loki.ww_plots.plot_styler import *


## ###############################################################
## FUNCTIONS
## ###############################################################
def create_figure(
    num_rows         : int   = 1,
    num_cols         : int   = 1,
    fig_scale        : float = 1.0,
    fig_aspect_ratio : tuple = (4, 6),
    wspace           : float = 0.05,
    hspace           : float = 0.05,
  ) -> tuple[mpl_plot.Figure, numpy.ndarray]:
  """Initialize a figure with a flexible grid layout."""
  fig_width  = fig_scale * fig_aspect_ratio[1] * num_cols
  fig_height = fig_scale * fig_aspect_ratio[0] * num_rows
  fig, axs = mpl_plot.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
  fig.subplots_adjust(wspace=wspace, hspace=hspace)
  if (num_rows > 1) or (num_cols > 1): axs = numpy.squeeze(axs)
  return fig, axs

def save_figure(fig, file_path, bool_draft=False, verbose=True):
  try:
    dpi = 100 if bool_draft else 200
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