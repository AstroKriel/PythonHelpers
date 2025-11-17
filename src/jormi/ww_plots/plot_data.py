## { MODULE

##
## === DEPENDENCIES
##

import numpy
from typing import Literal

from jormi.ww_types import array_types
from jormi.ww_plots import plot_manager, add_color

##
## === DATA TYPES
##

DataFormat = Literal["xy", "ij"]

##
## === FUNCTIONS
##


def as_plot_view(
    data_array: numpy.ndarray,
    data_format: DataFormat,
) -> numpy.ndarray:
    """
    Convert a 2D array to be plot-ready: [rows, cols].
        - layout="xy": is indexed [x, y]
        - layout="ij": is indexed [i:rows, j:cols]
    """
    match data_format:
        case "xy":
            return data_array.T
        case "ij":
            return data_array
        case _:
            raise ValueError(f"Data format `{data_format}` is not supported. Use 'xy' or 'ij'.")


def plot_2d_sarray(
    ax: plot_manager.Axis,
    sarray_in: numpy.ndarray,
    data_format: DataFormat,
    axis_aspect_ratio: Literal["equal", "auto"] = "equal",
    axis_bounds: tuple[float, float, float, float] | None = None,
    cbar_bounds: tuple[float, float] | None = None,
    cmap_name: str = "cmr.arctic",
    add_cbar: bool = True,
    cbar_label: str | None = None,
    cbar_side: str = "right",
):
    array_types.ensure_dim(
        array=sarray_in,
        dim=2,
    )
    sarray_plot = as_plot_view(
        data_array=sarray_in,
        data_format=data_format,
    )
    if cbar_bounds is None:
        min_value = 0.99 * numpy.nanmin(sarray_plot)
        max_value = 1.01 * numpy.nanmax(sarray_plot)
    else:
        min_value = cbar_bounds[0]
        max_value = cbar_bounds[1]
    cmap, norm = add_color.create_cmap(
        cmap_name=cmap_name,
        vmin=min_value,
        vmax=max_value,
    )
    im_obj = ax.imshow(
        sarray_plot,
        extent=axis_bounds,
        aspect=axis_aspect_ratio,
        origin="lower",
        cmap=cmap,
        norm=norm,
    )
    if axis_bounds is not None:
        ax.set_xlim([axis_bounds[0], axis_bounds[1]])
        ax.set_ylim([axis_bounds[2], axis_bounds[3]])
    if add_cbar:
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
    ax: plot_manager.Axis,
    field_slice_rows: numpy.ndarray,
    field_slice_cols: numpy.ndarray,
    axis_bounds: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
    num_quivers: int = 25,
    quiver_width: float = 5e-3,
    field_color: str = "white",
):
    if field_slice_rows.shape != field_slice_cols.shape:
        raise ValueError("`field_slice_rows` and `field_slice_cols` must be the same shape.")
    grid_x, grid_y = _generate_grid(
        field_shape=field_slice_rows.shape,
        axis_bounds=axis_bounds,
    )
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
    ax: plot_manager.Axis,
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
    grid_x, grid_y = _generate_grid(
        field_shape=field_slice_rows.shape,
        axis_bounds=axis_bounds,
    )
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
