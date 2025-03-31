## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import functools
import cmasher

import matplotlib.pyplot as mplplot
import matplotlib.colors as mplcolors
import matplotlib.ticker as mplticker

from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Loki.WWPlots import PlotAnnotations


## ###############################################################
## FUNCTIONS
## ###############################################################
def create_norm(vmin=0.0, vmax=1.0, NormType=mplcolors.Normalize):
  return NormType(vmin=vmin, vmax=vmax)

def create_cmap(
    cmap_name,
    cmin=0.0, cmax=1.0,
    vmin=0.0, vmid=None, vmax=1.0,
    NormType = mplcolors.Normalize
  ):
  if vmid is not None: NormType = functools.partial(mplcolors.TwoSlopeNorm, vcenter=vmid)
  cmap = cmasher.get_sub_cmap(cmap_name, cmin, cmax)
  norm = create_norm(vmin, vmax, NormType)
  return cmap, norm

def add_cbar_from_cmap(
    fig, ax, cmap,
    norm=None, vmin=0.0, vmax=1.0,
    orientation    = "horizontal",
    bool_log_ticks = False,
    cbar_title     = None,
    cbar_title_pad = 10,
    fontsize       = 16,
    size           = 10
  ):
  if norm is None: norm = create_norm(vmin, vmax)
  mappable = ScalarMappable(cmap=cmap, norm=norm)
  ax_div = make_axes_locatable(ax)
  if   "h" == orientation[0].lower():
    orientation = "horizontal"
    ax_cbar = ax_div.append_axes(position="top",   size=f"{size:.1f}%", pad="2%")
  elif "v" == orientation[0].lower():
    orientation = "vertical"
    ax_cbar = ax_div.append_axes(position="right", size=f"{size:.1f}%", pad="2%")
  else: raise Exception(f"Error: `{orientation}` is not a supported orientation!")
  # fig.add_axes(ax_cbar)
  cbar = fig.colorbar(mappable=mappable, cax=ax_cbar, orientation=orientation)
  if "h" in orientation:
    ax_cbar.set_title(cbar_title, fontsize=fontsize, pad=cbar_title_pad)
    ax_cbar.xaxis.set_ticks_position("top")
    ax_cbar.set_xscale("linear")
    if bool_log_ticks: ax_cbar.xaxis.set_major_formatter(mplticker.FuncFormatter(PlotAnnotations.labelLogFormatter))
  else:
    ax_cbar.set_ylabel(cbar_title, fontsize=fontsize, rotation=-90, va="bottom")
    ax_cbar.set_yscale("linear")
    if bool_log_ticks: ax_cbar.yaxis.set_major_formatter(mplticker.FuncFormatter(PlotAnnotations.labelLogFormatter))
  return cbar

def add_cbar_from_mappable(
    mappable,
    fig         = None,
    ax          = None,
    orientation = "vertical",
    cbar_title  = None,
    size        = 7.5,
    title_pad   = 12.5,
    fontsize    = 20
  ):
  if (fig is None) or (ax is None):
    ax  = mappable.axes
    fig = ax.figure
  ax_div = make_axes_locatable(ax)
  if   "h" in orientation: ax_cbar = ax_div.append_axes(position="top",   size=f"{size:.1f}%", pad="2%")
  elif "v" in orientation: ax_cbar = ax_div.append_axes(position="right", size=f"{size:.1f}%", pad="2%")
  cbar = fig.colorbar(mappable=mappable, cax=ax_cbar, orientation=orientation)
  if "h" in orientation:
    ax_cbar.set_title(cbar_title, fontsize=fontsize, pad=title_pad)
    ax_cbar.xaxis.set_ticks_position("top")
  else: cbar.ax.set_ylabel(cbar_title, fontsize=fontsize, rotation=-90, va="bottom")
  mplplot.sca(ax)
  return cbar


## END OF MODULE