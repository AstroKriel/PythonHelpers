## { MODULE

##
## === DEPENDENCIES
##

import numpy
from typing import Literal

from jormi.ww_types import type_checks, array_checks, cardinal_anchors
from jormi.ww_plots import plot_manager, add_color

##
## === DATA TYPES
##

DataFormat = Literal["xy", "ij"]
AxisBounds = tuple[tuple[float, float], tuple[float, float]]  # ((xmin, xmax), (ymin, ymax))

##
## === FUNCTIONS
##


def as_plot_view(
    data_array: numpy.ndarray,
    data_format: DataFormat,
) -> numpy.ndarray:
    """
    Convert a 2D array to a plot-ready array[rows, cols].
        - data_format="xy": is indexed [x, y]
        - data_format="ij": is indexed [i:rows, j:cols]
    """
    match data_format:
        case "xy":
            return data_array.T
        case "ij":
            return data_array
        case _:
            raise ValueError(f"Data format `{data_format}` is not supported. Use 'xy' or 'ij'.")


def _as_axis_extent(
    axis_bounds: AxisBounds | None,
) -> tuple[float, float, float, float] | None:
    if axis_bounds is None:
        return None
    type_checks.ensure_nested_tuple(
        param=axis_bounds,
        param_name="axis_bounds",
        outer_length=2,
        inner_length=2,
        valid_elem_types=type_checks.RuntimeTypes.Numerics.NumericLike,
        allow_none=False,
    )
    ## ensure booleans are rejected (bool is a subclass of int)
    type_checks.ensure_tuple_of_numbers(
        param=axis_bounds[0],
        param_name="axis_bounds[0]",
        seq_length=2,
        allow_none=False,
    )
    type_checks.ensure_tuple_of_numbers(
        param=axis_bounds[1],
        param_name="axis_bounds[1]",
        seq_length=2,
        allow_none=False,
    )
    (xmin, xmax), (ymin, ymax) = axis_bounds
    if not (xmin < xmax):
        raise ValueError(f"`axis_bounds[0]` must satisfy xmin < xmax, got ({xmin}, {xmax}).")
    if not (ymin < ymax):
        raise ValueError(f"`axis_bounds[1]` must satisfy ymin < ymax, got ({ymin}, {ymax}).")
    return (
        float(xmin),
        float(xmax),
        float(ymin),
        float(ymax),
    )


def _get_value_range(
    array_2d: numpy.ndarray,
    cbar_bounds: tuple[float, float] | None,
) -> tuple[float, float]:
    if cbar_bounds is not None:
        type_checks.ensure_tuple_of_numbers(
            param=cbar_bounds,
            param_name="cbar_bounds",
            seq_length=2,
            allow_none=False,
        )
        min_value, max_value = cbar_bounds
        min_value = float(min_value)
        max_value = float(max_value)
        if not (numpy.isfinite(min_value) and numpy.isfinite(max_value)):
            raise ValueError(f"`cbar_bounds` must be finite, got ({min_value}, {max_value}).")
        if not (min_value <= max_value):
            raise ValueError(f"`cbar_bounds` must satisfy min <= max, got ({min_value}, {max_value}).")
        return (min_value, max_value)
    finite_mask = numpy.isfinite(array_2d)
    if not numpy.any(finite_mask):
        raise ValueError("Array contains no finite values; cannot infer colorbar bounds.")
    min_value = float(numpy.min(array_2d[finite_mask]))
    max_value = float(numpy.max(array_2d[finite_mask]))
    if min_value == max_value:
        value_pad = 1e-12 if (min_value == 0.0) else 1e-12 * abs(min_value)
        min_value -= value_pad
        max_value += value_pad
    return (
        0.99 * min_value,
        1.01 * max_value,
    )


def plot_2d_array(
    ax: plot_manager.PlotAxis,
    array_2d: numpy.ndarray,
    data_format: DataFormat,
    axis_aspect_ratio: Literal["equal", "auto"] = "equal",
    axis_bounds: AxisBounds | None = None,
    cbar_bounds: tuple[float, float] | None = None,
    cmap_name: str = "cmr.arctic",
    add_cbar: bool = True,
    cbar_label: str | None = None,
    cbar_side: cardinal_anchors.AnchorLike = "right",
):
    array_checks.ensure_dims(
        array=array_2d,
        num_dims=2,
    )
    array_plot = as_plot_view(
        data_array=array_2d,
        data_format=data_format,
    )
    min_value, max_value = _get_value_range(
        array_2d=array_plot,
        cbar_bounds=cbar_bounds,
    )
    cmap_obj = add_color.CMap(
        min_value=min_value,
        max_value=max_value,
        cmap_name=cmap_name,
    )
    axis_extent = _as_axis_extent(axis_bounds)
    im_obj = ax.imshow(
        array_plot,
        extent=axis_extent,
        aspect=axis_aspect_ratio,
        origin="lower",
        cmap=cmap_obj.cmap,
        norm=cmap_obj.norm,
    )
    if axis_extent is not None:
        xmin, xmax, ymin, ymax = axis_extent
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
    if add_cbar:
        add_color.add_cbar_from_cmap(
            ax=ax,
            cmap=cmap_obj,
            label=cbar_label,
            side=cbar_side,
        )
    return im_obj


def _generate_grid(
    field_shape: tuple[int, int],
    axis_bounds: AxisBounds = ((-1.0, 1.0), (-1.0, 1.0)),
):
    axis_extent = _as_axis_extent(axis_bounds)
    if axis_extent is None:
        raise ValueError("`axis_bounds` must not be None.")
    xmin, xmax, ymin, ymax = axis_extent
    num_rows, num_cols = field_shape
    coords_x = numpy.linspace(xmin, xmax, num_cols)
    coords_y = numpy.linspace(ymin, ymax, num_rows)
    grid_x, grid_y = numpy.meshgrid(coords_x, coords_y, indexing="xy")
    return grid_x, grid_y


def plot_2d_quiver(
    ax: plot_manager.PlotAxis,
    array_2d_rows: numpy.ndarray,
    array_2d_cols: numpy.ndarray,
    axis_bounds: AxisBounds = ((-1.0, 1.0), (-1.0, 1.0)),
    num_quivers: int = 25,
    quiver_width: float = 5e-3,
    field_color: str = "white",
):
    array_checks.ensure_dims(
        array=array_2d_rows,
        num_dims=2,
    )
    array_checks.ensure_dims(
        array=array_2d_cols,
        num_dims=2,
    )
    if array_2d_rows.shape != array_2d_cols.shape:
        raise ValueError("`array_2d_rows` and `array_2d_cols` must be the same shape.")
    grid_x, grid_y = _generate_grid(
        field_shape=array_2d_rows.shape,
        axis_bounds=axis_bounds,
    )
    axis_extent = _as_axis_extent(axis_bounds)
    if axis_extent is None:
        raise ValueError("`axis_bounds` must not be None.")
    quiver_step_rows = max(1, array_2d_rows.shape[0] // num_quivers)
    quiver_step_cols = max(1, array_2d_cols.shape[1] // num_quivers)
    field_rows_subset = array_2d_rows[::quiver_step_rows, ::quiver_step_cols]
    field_cols_subset = array_2d_cols[::quiver_step_rows, ::quiver_step_cols]
    quiver_obj = ax.quiver(
        grid_x[::quiver_step_rows, ::quiver_step_cols],
        grid_y[::quiver_step_rows, ::quiver_step_cols],
        field_cols_subset,
        field_rows_subset,
        width=quiver_width,
        color=field_color,
    )
    xmin, xmax, ymin, ymax = axis_extent
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    return quiver_obj


def plot_2d_streamlines(
    ax: plot_manager.PlotAxis,
    array_2d_rows: numpy.ndarray,
    array_2d_cols: numpy.ndarray,
    axis_bounds: AxisBounds = ((-1.0, 1.0), (-1.0, 1.0)),
    streamline_width: float = 1.0,
    field_color: str = "white",
    streamline_density: float = 2.0,
    arrow_size: float = 1.0,
):
    array_checks.ensure_dims(
        array=array_2d_rows,
        num_dims=2,
    )
    array_checks.ensure_dims(
        array=array_2d_cols,
        num_dims=2,
    )
    if array_2d_rows.shape != array_2d_cols.shape:
        raise ValueError("`array_2d_rows` and `array_2d_cols` must have the same shape.")
    grid_x, grid_y = _generate_grid(
        field_shape=array_2d_rows.shape,
        axis_bounds=axis_bounds,
    )
    axis_extent = _as_axis_extent(axis_bounds)
    if axis_extent is None:
        raise ValueError("`axis_bounds` must not be None.")
    stream_obj = ax.streamplot(
        grid_x,
        grid_y,
        array_2d_cols,
        array_2d_rows,
        color=field_color,
        linewidth=streamline_width,
        density=streamline_density,
        arrowsize=arrow_size,
    )
    xmin, xmax, ymin, ymax = axis_extent
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    return stream_obj


## } MODULE
