## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
from matplotlib.lines import Line2D
from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox

from Loki.WWCollections import ListUtils


## ###############################################################
## FUNCTIONS
## ###############################################################
def addLegend(
    ax,
    loc      = "upper right",
    bbox     = (0.99, 0.99),
    ncol     = 1,
    lw       = 2.0,
    fontsize = 20,
    alpha    = 0.85,
    zorder   = 10
  ):
  obj_legend = ax.legend(
    ncol=ncol, loc=loc, bbox_to_anchor=bbox, fontsize=fontsize, framealpha=alpha,
    frameon=True, facecolor="white", edgecolor="grey"
  )
  obj_legend.set_zorder(zorder)
  for line in obj_legend.get_lines():
    line.set_linewidth(lw)
  return obj_legend

def addLegend_fromArtists(
    ax, list_artists, list_legend_labels,
    list_marker_colors = [ "k" ],
    bool_frame         = False,
    label_color        = "black",
    loc                = "upper right",
    bbox               = (1.0, 1.0),
    ms                 = 8,
    lw                 = 2,
    title              = None,
    ncol               = 1,
    handletextpad      = 0.5,
    rspacing           = 0.5,
    cspacing           = 0.5,
    fontsize           = 16,
    alpha              = 0.6
  ):
  if len(list_artists) + len(list_legend_labels) == 0: return
  ## check that the config are the correct length
  ListUtils.extendListToMatchLength(list_artists,       list_legend_labels)
  ListUtils.extendListToMatchLength(list_marker_colors, list_legend_labels)
  ## iniialise list of artists to draw
  list_legend_artists = []
  ## lists of artists the user can choose from
  list_markers = [ ".", "o", "s", "D", "^", "v" ]
  list_lines   = [ "-", "--", "-.", ":" ]
  for artist, marker_color in zip(list_artists, list_marker_colors):
    ## if the artist is a marker
    if artist in list_markers:
      list_legend_artists.append( 
        Line2D([0], [0], marker=artist, color=marker_color, linewidth=0, markeredgecolor="black", markersize=ms)
      )
    ## if the artist is a line
    elif artist in list_lines:
      list_legend_artists.append(
        Line2D([0], [0], linestyle=artist, color=marker_color, linewidth=lw)
      )
    ## otherwise throw an error
    else: raise Exception(f"Error: `{artist}` is not a valid valid.")
  ## create legend
  legend = ax.legend(
    list_legend_artists,
    list_legend_labels,
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
  ## draw legend
  ax.add_artist(legend)

def labelLogFormatter(x):
  if x % 1 == 0:
    return r"$10^{" + f"{x:.0f}" + "}$"
  else: return r"$10^{" + f"{x:.1f}" + "}$"

def addBoxOfLabels(
    fig, ax, list_labels,
    list_colors = [ "k" ],
    xpos        = 0.05,
    ypos        = 0.95,
    bbox        = (0.0, 1.0),
    alpha       = 0.5,
    fontsize    = 16,
  ):
  if len(list_labels) == 0: return
  ListUtils.extendListToMatchLength(list_colors, list_labels)
  list_text_areas = [
    TextArea(label, textprops={
      "fontsize" : fontsize,
      "color"    : list_colors[index_label]
    })
    for index_label, label in enumerate(list_labels)
    if (label is not None) and len(label) > 0
  ]
  texts_vbox = VPacker(
    children = list_text_areas,
    pad      = 2.5,
    sep      = 5.0,
  )
  abox = AnnotationBbox(
    texts_vbox,
    fontsize      = fontsize,
    xy            = (xpos, ypos),
    xycoords      = ax.transAxes,
    box_alignment = bbox,
    bboxprops     = dict(color="grey", facecolor="white", boxstyle="round", alpha=alpha, zorder=10)
  )
  abox.set_figure(fig)
  fig.artists.append(abox)

def addInsetAxis(
    ax, 
    ax_inset_bounds = [ 0.0, 1.0, 1.0, 0.5 ],
    label_x         = None,
    label_y         = None,
    fontsize        = 20
  ):
  ## create inset axis
  ax_inset = ax.inset_axes(ax_inset_bounds)
  ax_inset.tick_params(top=True, bottom=True, labeltop=True, labelbottom=False)
  ## add axis label
  ax_inset.set_xlabel(label_x, fontsize=fontsize)
  ax_inset.set_ylabel(label_y, fontsize=fontsize)
  ax_inset.xaxis.set_label_position("top")
  ## return inset axis
  return ax_inset

def labelDualAxis_sharedX(
    axs,
    label_left  = r"",
    label_right = r"",
    color_left  = "black",
    color_right = "black"
  ):
  axs[0].set_ylabel(label_left,  color=color_left)
  axs[1].set_ylabel(label_right, color=color_right, rotation=-90, labelpad=40)
  ## colour left/right axis-splines
  axs[0].tick_params(axis="y", colors=color_left)
  axs[1].tick_params(axis="y", colors=color_right)
  axs[1].spines["left" ].set_color(color_left)
  axs[1].spines["right"].set_color(color_right)

def labelDualAxis_sharedY(
    axs,
    label_bottom = r"",
    label_top    = r"",
    color_bottom = "black",
    color_top    = "black"
  ):
  axs[0].set_xlabel(label_bottom, color=color_bottom)
  axs[1].set_xlabel(label_top,    color=color_top, labelpad=20)
  ## colour bottom/top axis-splines
  axs[0].tick_params(axis="x", colors=color_bottom)
  axs[1].tick_params(axis="x", colors=color_top)
  axs[1].spines["bottom"].set_color(color_bottom)
  axs[1].spines["top"   ].set_color(color_top)


## END OF MODULE