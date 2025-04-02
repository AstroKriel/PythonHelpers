## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import functools

import matplotlib.pyplot as mplplot
import matplotlib.colors as mplcolors


from loki.WWPlots import PlotColour
from loki.WWFields import FieldOperators


## ###############################################################
## FUNCTIONS
## ###############################################################
def plot_sfield_slice(
    ax, field_slice,
    axis_bounds       = [-1, 1, -1, 1],
    cbar_bounds       = None,
    cbar_title        = None,
    cbar_orientation  = "horizontal",
    cmap_name         = "cmr.arctic",
  ):
  fig = ax.figure
  im_obj = ax.imshow(
    field_slice,
    extent = axis_bounds,
    cmap   = mplplot.get_cmap(cmap_name),
    norm   = NormType(
      vmin = 0.9 * numpy.min(field_slice) if cbar_bounds is None else cbar_bounds[0],
      vmax = 1.1 * numpy.max(field_slice) if cbar_bounds is None else cbar_bounds[1]
    )
  )
  if add_colorbar:
    PlotColour.add_cbar_from_mappable(
      fig         = fig,
      ax          = ax,
      mappable    = im_obj,
      cbar_title  = cbar_title,
      orientation = cbar_orientation
    )

def plot_vfield_slice(
    ax,
    field_slice_xrows,
    field_slice_xcols,
    bool_plot_magnitude   = True,
    bool_add_colorbar     = False,
    bool_norm_sfield      = False,
    bool_log10_sfield     = False,
    bool_center_cbar      = False,
    cmap_name             = "cmr.iceburn",
    cbar_orientation      = "horizontal",
    cbar_bounds           = None,
    cbar_title            = None,
    bool_plot_quiver      = False,
    num_quivers           = 25,
    quiver_width          = 5e-3,
    bool_plot_streamlines = True,
    streamline_weights    = None,
    streamline_width      = None,
    streamline_scale      = 1.5,
    streamline_linestyle  = "-",
    field_color           = "white",
    bool_label_axis       = False
  ):
  fig = ax.figure
  ## plot magnitude of vector field
  if bool_plot_magnitude:
    field_magnitude = FieldOperators.compute_vfield_magnitude([field_slice_xrows, field_slice_xcols])
    if bool_norm_sfield:  field_magnitude = field_magnitude**2 / FieldOperators.compute_sfield_rms(field_magnitude)**2
    if bool_log10_sfield: field_magnitude = numpy.log10(field_magnitude)
    if bool_center_cbar:  NormType = functools.partial(mplcolors.TwoSlopeNorm, vcenter=0)
    else:                 NormType = mplcolors.Normalize
    im_obj = ax.imshow(
      field_magnitude,
      origin = "lower",
      extent = [-1.0, 1.0, -1.0, 1.0],
      cmap   = mplplot.get_cmap(cmap_name),
      norm   = NormType(
        vmin = 0.9 * numpy.min(field_magnitude) if cbar_bounds is None else cbar_bounds[0],
        vmax = 1.1 * numpy.max(field_magnitude) if cbar_bounds is None else cbar_bounds[1]
      )
    )
    ## add colorbar
    if bool_add_colorbar:
      PlotColour.add_cbar_from_mappable(
        fig         = fig,
        ax          = ax,
        mappable    = im_obj,
        cbar_title  = cbar_title,
        orientation = cbar_orientation
      )
  ## overlay vector field
  if bool_plot_quiver:
    quiver_step_rows = field_slice_xrows.shape[0] // num_quivers
    quiver_step_cols = field_slice_xcols.shape[1] // num_quivers
    field_slice_xrows_subset = field_slice_xrows[::quiver_step_rows, ::quiver_step_cols]
    field_slice_xcols_subset = field_slice_xcols[::quiver_step_rows, ::quiver_step_cols]
    coords_row = numpy.linspace(-1.0, 1.0, field_slice_xrows_subset.shape[0])
    coords_col = numpy.linspace(-1.0, 1.0, field_slice_xcols_subset.shape[1])
    grid_x, grid_y = numpy.meshgrid(coords_col, coords_row, indexing="xy")
    ax.quiver(
      grid_x,
      grid_y,
      field_slice_xcols_subset,
      field_slice_xrows_subset,
      width = quiver_width,
      color = field_color
    )
  if bool_plot_streamlines:
    coords_row = numpy.linspace(-1.0, 1.0, field_slice_xrows.shape[0])
    coords_col = numpy.linspace(-1.0, 1.0, field_slice_xcols.shape[1])
    grid_x, grid_y = numpy.meshgrid(coords_col, coords_row, indexing="xy")
    if streamline_width is None:
      if streamline_weights is not None: streamline_width = streamline_scale * (1 + streamline_weights / numpy.max(streamline_weights))
      else: streamline_width = 1
    ax.streamplot(
      grid_x,
      grid_y,
      field_slice_xcols,
      field_slice_xrows,
      color      = field_color,
      arrowstyle = streamline_linestyle,
      linewidth  = streamline_width,
      density    = 2.0,
      arrowsize  = 1.0,
    )
  ## add axis labels
  ax.set_xlim([-1, 1])
  ax.set_ylim([-1, 1])
  if bool_label_axis:
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels([r"$-L/2$", r"$-L/4$", r"$0$", r"$L/4$", r"$L/2$"])
    ax.set_yticklabels([r"$-L/2$", r"$-L/4$", r"$0$", r"$L/4$", r"$L/2$"])
  else:
    ax.set_xticks([])
    ax.set_yticks([])


## END OF MODULE