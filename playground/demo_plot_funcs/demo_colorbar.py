## ###############################################################
## DEPENDANCIES
## ###############################################################
import numpy
import matplotlib.ticker as mpl_ticker
from loki.ww_plots import plot_manager, plot_color


def add_cbar_from_cmap(
    ax, cmap, norm,
    label         : str = "",
    side          : str = "right",
    percentage    : float = 0.1,
    cbar_padding  : float = 0.02,
    label_padding : float = 10,
    fontsize      : float = 20,
  ):
  fig = ax.figure
  box = ax.get_position()
  if side in [ "left", "right" ]:
    orientation = "vertical"
    cbar_size = box.width * percentage
    if side == "right": cbar_bounds = [ box.x1 + cbar_padding, box.y0, cbar_size, box.height ]
    else:               cbar_bounds = [ box.x0 - cbar_size - cbar_padding, box.y0, cbar_size, box.height ]
  elif side in [ "top", "bottom" ]:
    orientation = "horizontal"
    cbar_size = box.height * percentage
    if side == "top": cbar_bounds = [ box.x0, box.y1 + cbar_padding, box.width, cbar_size ]
    else:             cbar_bounds = [ box.x0, box.y0 - cbar_size - cbar_padding, box.width, cbar_size ]
  else: raise ValueError(f"Unsupported side: {side}")
  ax_cbar = fig.add_axes(cbar_bounds)
  cbar = fig.colorbar(mappable=None, cmap=cmap, norm=norm, cax=ax_cbar, orientation=orientation)
  if orientation == "horizontal":
    cbar.ax.set_title(label, fontsize=fontsize, pad=label_padding)
    cbar.ax.xaxis.set_ticks_position(side)
  else:
    cbar.set_label(label, fontsize=fontsize, rotation=-90, va="bottom")
    cbar.ax.yaxis.set_ticks_position(side)
  return cbar



## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def generate_gaussian_data(size=100, sigma=20):
    x_mu = y_mu = size // 2
    domain = numpy.linspace(0, size-1, size)
    x_grid, y_grid = numpy.meshgrid(domain, domain)
    data = numpy.exp(-((x_grid - x_mu)**2 + (y_grid - y_mu)**2) / (2 * sigma**2))
    return data


## ###############################################################
## DEMO SCRIPT
## ###############################################################
def main():
  data = generate_gaussian_data()
  cmap, norm = plot_color.create_colormap(
    cmap_name = "cmr.arctic",
    norm_type = "linear",
    vmin      = numpy.min(data),
    vmax      = numpy.max(data),
  )
  fig, ax = plot_manager.create_figure()
  axis_bounds = [-1, 1, -1, 1]
  ax.imshow(
    data,
    extent = axis_bounds,
    cmap   = cmap,
    norm   = norm
  )
  ax.set_aspect("equal")
  ax.set_xlim([axis_bounds[0], axis_bounds[1]])
  ax.set_ylim([axis_bounds[2], axis_bounds[3]])
  add_cbar_from_cmap(
    ax    = ax,
    cmap  = cmap,
    norm  = norm,
    side  = "right",
    label = "this is my title",
  )
  ax.set_xticks([])
  ax.set_yticks([])
  plot_manager.save_figure(fig, "colorbar.png")


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT