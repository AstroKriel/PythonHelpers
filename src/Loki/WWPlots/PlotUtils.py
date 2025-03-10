## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import matplotlib.pyplot as mplplot
from matplotlib.gridspec import GridSpec
from typing import Union
from Loki.WWPlots.PlotStyler import *


## ###############################################################
## FUNCTIONS
## ###############################################################
def initFigure(
    num_rows         : int   = 1,
    num_cols         : int   = 1,
    fig_scale        : float = 1.25,
    fig_aspect_ratio : tuple = (4, 6),
    wspace           : float = -1,
    hspace           : float = -1,
    bool_return_axis : bool  = True
  ) -> Union[
    tuple[mplplot.Figure, numpy.ndarray],
    tuple[mplplot.Figure, GridSpec]
  ]:
  """Initialize a figure with a flexible grid layout."""
  fig = mplplot.figure(
    constrained_layout = True,
    figsize            = (
      fig_scale * fig_aspect_ratio[1] * num_cols,
      fig_scale * fig_aspect_ratio[0] * num_rows
  ))
  fig_grid = GridSpec(
    num_rows, num_cols,
    figure = fig,
    wspace = wspace,
    hspace = hspace
  )
  if bool_return_axis:
    if num_rows + num_cols == 2: return fig, fig.add_subplot(fig_grid[0,0])
    axs = numpy.empty((num_rows, num_cols), dtype=object)
    for row in range(num_rows):
      for col in range(num_cols):
        axs[row, col] = fig.add_subplot(fig_grid[row, col])
    return fig, numpy.squeeze(axs)
  return fig, fig_grid

def saveFigure(fig, filepath_fig, bool_draft=False, bool_verbose=True):
  try:
    fig.set_constrained_layout(True)
    dpi = 100 if bool_draft else 200
    fig.savefig(filepath_fig, dpi=dpi)
    mplplot.close(fig)
    if bool_verbose: print("Saved figure:", filepath_fig)
  except FileNotFoundError as exception:
    print(f"FileNotFoundError: `{filepath_fig}` does not exist.")
    print(f"Details: {exception}")
  except IOError as exception:
    print(f"IOError: An error occurred while trying to save the figure to '{filepath_fig}'. This might be due to a permission issue or invalid file format.")
    print(f"Details: {exception}")
  except Exception as exception:
    print(f"Unexpected error: {exception}")


## END OF MODULE