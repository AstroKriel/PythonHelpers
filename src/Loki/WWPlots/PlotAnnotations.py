## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
from matplotlib.lines import Line2D


## ###############################################################
## FUNCTIONS
## ###############################################################
def addCustomLegend(
    ax, list_artists, list_labels, list_colors,
    label_color   = "black",
    loc           = "upper right",
    bbox          = (1.0, 1.0),
    bool_frame    = False,
    title         = None,
    ms            = 8,
    lw            = 1.5,
    ncol          = 1,
    handletextpad = 0.5,
    rspacing      = 0.5,
    cspacing      = 0.5,
    fontsize      = 16,
    alpha         = 0.6
  ):
  if len(list_artists) + len(list_labels) == 0: return
  list_artists_to_draw = []
  list_valid_markers   = [ ".", "o", "s", "D", "^", "v" ]
  list_valid_lines     = [ "-", "--", "-.", ":" ]
  for artist, color in zip(list_artists, list_colors):
    if artist in list_valid_markers:
      obj_to_draw = Line2D([0], [0], marker=artist, color=color, linewidth=0, markeredgecolor="black", markersize=ms)
    elif artist in list_valid_lines:
      obj_to_draw = Line2D([0], [0], linestyle=artist, color=color, linewidth=lw)
    else: raise ValueError(f"Error: `{artist}` is not a valid marker or line style.")
    list_artists_to_draw.append(obj_to_draw)
  legend = ax.legend(
    list_artists_to_draw,
    list_labels,
    frameon        = bool_frame,
    title          = title,
    loc            = loc,
    bbox_to_anchor = bbox,
    ncol           = ncol,
    borderpad      = 0.45,
    handletextpad  = handletextpad,
    labelspacing   = rspacing,
    columnspacing  = cspacing,
    fontsize       = fontsize,
    labelcolor     = label_color,
    framealpha     = alpha,
    facecolor      = "white",
    edgecolor      = "grey"
  )
  ax.add_artist(legend)

def addInsetAxis(
    ax, 
    ax_inset_bounds = [ 0.0, 1.0, 1.0, 0.5 ],
    label_x         = None,
    label_y         = None,
    fontsize        = 20
  ):
  ax_inset = ax.inset_axes(ax_inset_bounds)
  ax_inset.tick_params(top=True, bottom=True, labeltop=True, labelbottom=False)
  ax_inset.set_xlabel(label_x, fontsize=fontsize)
  ax_inset.set_ylabel(label_y, fontsize=fontsize)
  ax_inset.xaxis.set_label_position("top")
  return ax_inset


## END OF MODULE