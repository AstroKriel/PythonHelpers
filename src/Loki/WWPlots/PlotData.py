## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import functools

import matplotlib.pyplot as mplplot
import matplotlib.colors as mplcolors

from matplotlib.collections import LineCollection

from Loki.WWPlots import PlotColour
from Loki.WWFields import FieldOperators


## ###############################################################
## FUNCTIONS
## ###############################################################
def plot_data_wo_scaling_axis(
    ax, x, y,
    color="k", ls=":", lw=1, label=None, alpha=1.0, zorder=1
  ):
  col = LineCollection(
    [ numpy.column_stack((x, y)) ],
    colors = color,
    ls     = ls,
    lw     = lw,
    label  = label,
    alpha  = alpha,
    zorder = zorder
  )
  ax.add_collection(col, autolim=False)

def plot_error_bar(
    ax, x, array_y,
    label   = None,
    color   = "k",
    marker  = "o",
    capsize = 7.5,
    alpha   = 1.0,
    zorder  = 5
  ):
  array_y = [
    y
    for y in array_y
    if y is not None
  ]
  if len(array_y) < 5: return
  y_p16  = numpy.nanpercentile(array_y, 16)
  y_p50  = numpy.nanpercentile(array_y, 50)
  y_p84  = numpy.nanpercentile(array_y, 84)
  y_1sig = numpy.vstack([
    y_p50 - y_p16,
    y_p84 - y_p50
  ])
  ax.errorbar(
    x, y_p50,
    yerr    = y_1sig,
    color   = color,
    fmt     = marker,
    label   = label,
    alpha   = alpha,
    capsize = capsize,
    zorder  = zorder,
    markersize=7, markeredgecolor="black",
    elinewidth=1.5, linestyle="None"
  )

def plot_pdf(
    ax, list_data,
    num_bins     = 10,
    weights      = None,
    color        = "black",
    bool_flip_ax = False
  ):
  list_dens, list_bin_edges = numpy.histogram(list_data, bins=num_bins, weights=weights)
  list_dens_norm = numpy.append(0, list_dens / list_dens.sum())
  if bool_flip_ax:
    ax.plot(list_dens_norm[::-1], list_bin_edges[::-1], drawstyle="steps", color=color)
    ax.fill_between(list_dens_norm[::-1], list_bin_edges[::-1], step="pre", alpha=0.2, color=color)
  else:
    ax.plot(list_bin_edges, list_dens_norm, drawstyle="steps", color=color)
    ax.fill_between(list_bin_edges, list_dens_norm, step="pre", alpha=0.2, color=color)
  return list_bin_edges, list_dens_norm

def plot_sfield_slice(
    field_slice,
    fig               = None,
    ax                = None,
    bool_add_colorbar = False,
    bool_center_cbar  = False,
    cbar_bounds       = None,
    cbar_title        = None,
    cbar_orientation  = "horizontal",
    cmap_name         = "cmr.arctic",
    NormType          = mplcolors.LogNorm,
    bool_label_axis   = False,
  ):
  ## check that a figure object has been passed
  if (fig is None) and (ax is None): fig, ax = mplplot.subplots(constrained_layout=True)
  if (fig is None) and not(ax is None): fig = ax.figure
  ## plot scalar field
  if bool_center_cbar: NormType = functools.partial(mplcolors.TwoSlopeNorm, vcenter=0)
  im_obj = ax.imshow(
    field_slice,
    extent = [-1, 1, -1, 1],
    cmap   = mplplot.get_cmap(cmap_name),
    norm   = NormType(
      vmin = 0.9 * numpy.min(field_slice) if cbar_bounds is None else cbar_bounds[0],
      vmax = 1.1 * numpy.max(field_slice) if cbar_bounds is None else cbar_bounds[1]
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
  ## add axis labels
  if bool_label_axis:
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels([r"$-L/2$", r"$-L/4$", r"$0$", r"$L/4$", r"$L/2$"])
    ax.set_yticklabels([r"$-L/2$", r"$-L/4$", r"$0$", r"$L/4$", r"$L/2$"])
  else:
    # ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
  return fig, ax

def plot_vfield_slice(
    field_slice_xrows,
    field_slice_xcols,
    fig                   = None,
    ax                    = None,
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
  ## check that a figure object has been passed
  if (fig is None) and (ax is None): fig, ax = mplplot.subplots(constrained_layout=True)
  if (fig is None) and not(ax is None): fig = ax.figure
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
    # ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])


## END OF MODULE