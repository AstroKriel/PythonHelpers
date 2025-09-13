## { MODULE

##
## === DEPENDENCIES ===
##

import numpy
from jormi.ww_plots import add_color

##
## === FUNCTIONS ===
##


def plot_sfield_slice(
    ax,
    field_slice: numpy.ndarray,
    axis_bounds: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
    cbar_bounds: tuple[float, float] | None = None,
    cmap_name: str = "cmr.arctic",
    add_colorbar: bool = True,
    cbar_label: str | None = None,
    cbar_side: str = "right",
):
    if field_slice.ndim != 2: raise ValueError("`field_slice` must be a 2D array.")
    vmin = 0.95 * numpy.min(field_slice) if (cbar_bounds is None) else cbar_bounds[0]
    vmax = 1.05 * numpy.max(field_slice) if (cbar_bounds is None) else cbar_bounds[1]
    cmap, norm = add_color.create_cmap(
        cmap_name=cmap_name,
        vmin=vmin,
        vmax=vmax,
    )
    im_obj = ax.imshow(field_slice, extent=axis_bounds, cmap=cmap, norm=norm)
    ax.set_xlim([axis_bounds[0], axis_bounds[1]])
    ax.set_ylim([axis_bounds[2], axis_bounds[3]])
    if add_colorbar:
        add_color.add_cbar_from_cmap(
            ax=ax,
            cmap=cmap,
            norm=norm,
            label=cbar_label,
            side=cbar_side,
        )
    return im_obj


def _generate_grid(
    field_shape: tuple[int, int],
    axis_bounds: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
):
    is_tuple = isinstance(axis_bounds, tuple)
    all_elems_defn = len(axis_bounds) == 4
    valid_elem_type = all(isinstance(value, (int, float)) for value in axis_bounds)
    if not all((is_tuple, all_elems_defn, valid_elem_type)):
        raise ValueError("`axis_bounds` must be a tuple with four floats.")
    coords_row = numpy.linspace(axis_bounds[0], axis_bounds[1], field_shape[0])
    coords_col = numpy.linspace(axis_bounds[2], axis_bounds[3], field_shape[1])
    grid_x, grid_y = numpy.meshgrid(coords_col, coords_row, indexing="xy")
    return grid_x, grid_y


def plot_vfield_slice_quiver(
    ax,
    field_slice_rows: numpy.ndarray,
    field_slice_cols: numpy.ndarray,
    axis_bounds: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
    num_quivers: int = 25,
    quiver_width: float = 5e-3,
    field_color: str = "white",
):
    if field_slice_rows.shape != field_slice_cols.shape:
        raise ValueError("`field_slice_rows` and `field_slice_cols` must be the same shape.")
    grid_x, grid_y = _generate_grid(field_slice_rows.shape, axis_bounds)
    quiver_step_rows = max(1, field_slice_rows.shape[0] // num_quivers)
    quiver_step_cols = max(1, field_slice_cols.shape[1] // num_quivers)
    field_slice_xrows_subset = field_slice_rows[::quiver_step_rows, ::quiver_step_cols]
    field_slice_xcols_subset = field_slice_cols[::quiver_step_rows, ::quiver_step_cols]
    ax.quiver(
        grid_x[::quiver_step_rows, ::quiver_step_cols],
        grid_y[::quiver_step_rows, ::quiver_step_cols],
        field_slice_xcols_subset,
        field_slice_xrows_subset,
        width=quiver_width,
        color=field_color,
    )
    ax.set_xlim([axis_bounds[0], axis_bounds[1]])
    ax.set_ylim([axis_bounds[2], axis_bounds[3]])


def plot_vfield_slice_streamplot(
    ax,
    field_slice_rows: numpy.ndarray,
    field_slice_cols: numpy.ndarray,
    axis_bounds: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
    streamline_weights: numpy.ndarray | None = None,
    streamline_width: float | None = None,
    streamline_scale: float = 1.5,
    streamline_linestyle: str = "-",
    field_color: str = "white",
):
    if field_slice_rows.shape != field_slice_cols.shape:
        raise ValueError("`field_slice_rows` and `field_slice_cols` must have the same shape.")
    grid_x, grid_y = _generate_grid(field_slice_rows.shape, axis_bounds)
    if streamline_width is None:
        if streamline_weights is None: streamline_width = 1
        elif streamline_weights.shape != field_slice_cols.shape:
            raise ValueError("`streamline_weights` must have the same shape as field slices.")
        else:
            streamline_width = streamline_scale * (1 + streamline_weights / numpy.max(streamline_weights))
    ax.streamplot(
        grid_x,
        grid_y,
        field_slice_cols,
        field_slice_rows,
        color=field_color,
        linewidth=streamline_width,
        density=2.0,
        arrowsize=1.0,
        linestyle=streamline_linestyle,
    )
    ax.set_xlim([axis_bounds[0], axis_bounds[1]])
    ax.set_ylim([axis_bounds[2], axis_bounds[3]])


## } MODULE
