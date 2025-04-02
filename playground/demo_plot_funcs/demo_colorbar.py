## ###############################################################
## DEPENDANCIES
## ###############################################################
import numpy
from loki.ww_plots import plot_manager, plot_color


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
  plot_color.add_cbar_from_cmap(
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