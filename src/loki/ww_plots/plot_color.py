## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import cmasher
import matplotlib.colors as mpl_colors
import matplotlib.ticker as mpl_ticker
from loki.ww_plots import plot_annotations


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


## END OF MODULE