## ###############################################################
## DEPENDANCIES
## ###############################################################
import numpy
from loki.ww_plots import plot_manager, plot_data


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
  fig, ax = plot_manager.create_figure()
  axis_bounds = [-1, 1, -1, 1]
  plot_data.plot_sfield_slice(
    ax,
    field_slice  = data,
    axis_bounds  = axis_bounds,
    cmap_name    = "cmr.arctic",
    add_colorbar = True,
    cbar_label   = "this is my label",
    cbar_side    = "right",
  )
  ax.set_aspect("equal")
  ax.set_xlim([axis_bounds[0], axis_bounds[1]])
  ax.set_ylim([axis_bounds[2], axis_bounds[3]])
  ax.set_xticks([])
  ax.set_yticks([])
  plot_manager.save_figure(fig, "colorbar.png")


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT