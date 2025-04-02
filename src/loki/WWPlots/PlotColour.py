## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import cmasher

import matplotlib.pyplot as mpl_plot
import matplotlib.colors as mpl_colors
import matplotlib.ticker as mpl_ticker

from loki.WWPlots import PlotAnnotations


## ###############################################################
## FUNCTIONS
## ###############################################################
def create_norm(
      vmin : float = 0.0,
      vmax : float = 1.0,
      vmid : float = None,
      norm_type : str = "linear",
    ):
    if norm_type not in ["linear", "log"]: raise ValueError(f"Unsupported norm_type: {norm_type}. Choose either `linear` or `log`.")
    if norm_type == "linear":
      if vmid is not None:
        return mpl_colors.TwoSlopeNorm(vmin=vmin, vcenter=vmid, vmax=vmax)
      else: return mpl_colors.Normalize(vmin=vmin, vmax=vmax)
    elif norm_type == "log":
      if vmid is not None:
        threshold_for_linear_region = 1e-2 * max(abs(vmin), abs(vmax)) # 1 percent of absolute largest value
        threshold_for_linear_region = max(1e-5, threshold_for_linear_region) # prevent threshold from being too small
        return mpl_colors.SymLogNorm(linthresh=threshold_for_linear_region, vmin=vmin, vmax=vmax)
      else:
        if vmin <= 0 or vmax <= 0: raise ValueError("`LogNorm` requires positive `vmin` and `vmax` values.")
        return mpl_colors.LogNorm(vmin=vmin, vmax=vmax)

def create_colormap(
    cmap_name: str,
    cmin: float = 0.0,
    cmax: float = 1.0,
    vmin: float = 0.0,
    vmid: float = None,
    vmax: float = 1.0,
    norm_type: str = "linear",
  ):
  cmap = cmasher.get_sub_cmap(cmap_name, cmin, cmax)
  norm = create_norm(vmin, vmax, vmid, norm_type)
  return cmap, norm

def add_cbar_from_cmap(
    fig, ax, cmap,
    norm=None, vmin=0.0, vmax=1.0,
    orientation    = "horizontal",
    bool_log_ticks = False,
    cbar_title     = None,
    cbar_title_pad = 10,
    fontsize       = 16,
  ):
  if norm is None: norm = create_norm(vmin, vmax)
  box = ax.get_position()
  if   "h" in orientation: ax_cbar = fig.add_axes([box.x0, box.y1 + 0.02, box.width, 0.05])
  elif "v" in orientation: ax_cbar = fig.add_axes([box.x1 + 0.02, box.y0, 0.05, box.height])
  else: raise Exception(f"Error: `{orientation}` is not a supported orientation!")
  # fig.add_axes(ax_cbar)
  cbar = fig.colorbar(mappable=None, cmap=cmap, norm=norm, cax=ax_cbar, orientation=orientation)
  if "h" in orientation:
    ax_cbar.set_title(cbar_title, fontsize=fontsize, pad=cbar_title_pad)
    ax_cbar.xaxis.set_ticks_position("top")
    ax_cbar.set_xscale("linear")
    if bool_log_ticks: ax_cbar.xaxis.set_major_formatter(mpl_ticker.FuncFormatter(PlotAnnotations.labelLogFormatter))
  else:
    ax_cbar.set_ylabel(cbar_title, fontsize=fontsize, rotation=-90, va="bottom")
    ax_cbar.set_yscale("linear")
    if bool_log_ticks: ax_cbar.yaxis.set_major_formatter(mpl_ticker.FuncFormatter(PlotAnnotations.labelLogFormatter))
  return cbar


## END OF MODULE