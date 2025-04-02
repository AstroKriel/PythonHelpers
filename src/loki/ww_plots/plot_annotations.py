## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from matplotlib.axes import Axes as mpl_axes
from matplotlib.lines import Line2D as mpl_line2d


## ###############################################################
## FUNCTIONS
## ###############################################################
def tick_log_formater(tick_value, _):
   # Handle special cases for ticks close to 0 or below 0
  if tick_value <= 0: return "N/A"
  # when the value is a power of 10, display it as "10^n"
  elif numpy.isclose(tick_value, 1): return "1"
  elif tick_value == 10:   return "$10^1$"
  elif tick_value == 100:  return "$10^2$"
  elif tick_value == 1000: return "$10^3$"
  # use scientific notation
  else: return f"${tick_value:.1e}$"

def add_inset_axis(
    ax, 
    bounds   = [ 0.0, 1.0, 1.0, 0.5 ],
    x_label  = None,
    y_label  = None,
    fontsize = 20
  ):
  ax_inset = ax.inset_axes(bounds)
  ax_inset.tick_params(top=True, bottom=True, labeltop=True, labelbottom=False)
  ax_inset.set_xlabel(x_label, fontsize=fontsize)
  ax_inset.set_ylabel(y_label, fontsize=fontsize)
  ax_inset.xaxis.set_label_position("top")
  return ax_inset

def add_custom_legend(
    ax             : mpl_axes,
    artists        : list[str],
    labels         : list[str],
    colors         : list[str],
    marker_size    : float = 8,
    line_width     : float = 1.5,
    fontsize       : float = 16,
    text_color     : str = "black",
    position       : str = "upper right",
    anchor         : tuple[float, float] = (1.0, 1.0),
    enable_frame   : bool = False,
    frame_alpha    : float = 0.5,
    num_cols       : float = 1,
    text_padding   : float = 0.5,
    label_spacing  : float = 0.5,
    column_spacing : float = 0.5,
  ):
  artists_to_draw = []
  valid_markers   = [ ".", "o", "s", "D", "^", "v" ]
  valid_lines     = [ "-", "--", "-.", ":" ]
  for artist, color in zip(artists, colors):
    if artist in valid_markers: artist_to_draw = mpl_line2d([0], [0], marker=artist, color=color, linewidth=0, markeredgecolor="black", markersize=marker_size)
    elif artist in valid_lines: artist_to_draw = mpl_line2d([0], [0], linestyle=artist, color=color, linewidth=line_width)
    else: raise ValueError(f"Invalid artist = `{artist}`. Must be a valid marker ({valid_markers}) or line style ({valid_lines}).")
    artists_to_draw.append(artist_to_draw)
  legend = ax.legend(
    artists_to_draw,
    labels,
    bbox_to_anchor = anchor,
    loc            = position,
    fontsize       = fontsize,
    labelcolor     = text_color,
    frameon        = enable_frame,
    framealpha     = frame_alpha,
    facecolor      = "white",
    edgecolor      = "grey",
    ncol           = num_cols,
    borderpad      = 0.45,
    handletextpad  = text_padding,
    labelspacing   = label_spacing,
    columnspacing  = column_spacing,
  )
  ax.add_artist(legend)


## END OF MODULE