## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import matplotlib.pyplot as mplplot
from matplotlib.gridspec import GridSpec


## ###############################################################
## FUNCTIONS
## ###############################################################
def initFigure(
    num_rows         = 1,
    num_cols         = 1,
    fig_scale        = 1.0,
    fig_aspect_ratio = (4, 6),
    wspace           = -1,
    hspace           = -1
  ):
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
  return fig, fig_grid

def saveFigure(fig, filepath_fig, bool_tight=True, bool_draft=False, bool_verbose=True):
  if bool_tight and not(fig.get_constrained_layout()): fig.set_tight_layout(True)
  if bool_draft: dpi = 100
  else: dpi = 200
  fig.savefig(filepath_fig, dpi=dpi)
  mplplot.close(fig)
  if bool_verbose: print("Saved figure:", filepath_fig)


## END OF MODULE