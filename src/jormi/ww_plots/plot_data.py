## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Any, Literal, cast

## third-party
import numpy
from numpy.typing import NDArray

## local
from jormi.ww_plots import (
    add_color,
    manage_plots,
)
from jormi.ww_types import (
    box_positions,
    check_arrays,
    check_types,
)

##
## === DATA TYPES
##

DataFormat = Literal["xy", "ij"]
AxisBounds = tuple[
    tuple[float, float],  # ((min_x_value, max_x_value)
    tuple[float, float],  # (min_y_value, max_y_value)
]

##
## === INTERNAL HELPERS
##


def as_plot_view(
    data_array: NDArray[Any],
    data_format: DataFormat,
) -> NDArray[Any]:
    """
    Convert a 2D array to a plot-ready array[rows, cols], given its current indexing format.
        - `data_format="xy"`: array is currently indexed [x, y] -> transpose to [rows, cols]
        - `data_format="ij"`: array is currently indexed [i=rows, j=cols] -> pass through unchanged
    """
    match data_format:
        case "xy":
            return data_array.T
        case "ij":
            return data_array
        case _:  # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError(
                f"Data format `{data_format}` is not supported. Use 'xy' or 'ij'.",
            )  # pyright: ignore[reportUnreachable]


def _as_axis_extent(
    axis_bounds: AxisBounds | None,
) -> tuple[float, float, float, float] | None:
    """
    Convert AxisBounds to the flat (xmin, xmax, ymin, ymax) extent format expected by matplotlib.
    Returns None if `axis_bounds` is None.
    """
    if axis_bounds is None:
        return None
    check_types.ensure_nested_tuple(
        param=axis_bounds,
        param_name="axis_bounds",
        outer_length=2,
        inner_length=2,
        valid_elem_types=check_types.RuntimeTypes.Numerics.NumericLike,
        allow_none=False,
    )
    check_types.ensure_ordered_pair(
        param=axis_bounds[0],
        param_name="axis_bounds[0]",
        allow_none=False,
        strict_ordering=True,
    )
    check_types.ensure_ordered_pair(
        param=axis_bounds[1],
        param_name="axis_bounds[1]",
        allow_none=False,
        strict_ordering=True,
    )
    (min_x_value, max_x_value), (min_y_value, max_y_value) = axis_bounds
    return (
        float(min_x_value),
        float(max_x_value),
        float(min_y_value),
        float(max_y_value),
    )


def _get_value_range(
    array_2d: NDArray[Any],
    cbar_bounds: tuple[float, float] | None,
) -> tuple[float, float]:
    """
    Calculate the (min, max) value range for colorbar scaling.
    
    If `cbar_bounds` is provided, validate and use it directly. Otherwise, infer from the finite
    values in `array_2d`, with a small pad applied.
    """
    finite_mask = numpy.isfinite(array_2d)
    ## validate user supplied bounds and return directly
    if cbar_bounds is not None:
        check_types.ensure_ordered_pair(
            param=cbar_bounds,
            param_name="cbar_bounds",
            allow_none=False,
        )
        min_value, max_value = float(cbar_bounds[0]), float(cbar_bounds[1])
        if not (numpy.isfinite(min_value) and numpy.isfinite(max_value)):
            raise ValueError(f"`cbar_bounds` must be finite, got ({min_value}, {max_value}).")
        in_range_mask = finite_mask & (array_2d >= min_value) & (array_2d <= max_value)
        if not numpy.any(in_range_mask):
            raise ValueError(f"`cbar_bounds` ({min_value}, {max_value}) does not overlap with data.")
        return (
            min_value,
            max_value,
        )
    ## infer bounds from data, with a small pad to avoid degenerate colormaps
    if not numpy.any(finite_mask):
        raise ValueError("array contains no finite values; cannot infer colorbar bounds.")
    min_value = float(numpy.min(array_2d[finite_mask]))
    max_value = float(numpy.max(array_2d[finite_mask]))
    if min_value == max_value:
        pad_value = 1e-12 if (min_value == 0.0) else 1e-12 * abs(min_value)
        min_value -= pad_value
        max_value += pad_value
    ## return with a slightly clipped range
    return (
        0.99 * min_value,
        1.01 * max_value,
    )


##
## === PLOT FUNCTIONS
##


def plot_2d_array(
    ax: manage_plots.PlotAxis,
    array_2d: NDArray[Any],
    data_format: DataFormat,
    axis_aspect_ratio: Literal["equal", "auto"] = "equal",
    axis_bounds: AxisBounds | None = None,
    cbar_bounds: tuple[float, float] | None = None,
    palette_config: add_color.PaletteConfig | None = None,
    add_cbar: bool = True,
    cbar_label: str | None = None,
    cbar_side: box_positions.Positions.PositionLike = box_positions.Positions.Box.Side.Right,
):
    if palette_config is None:
        palette_config = add_color.SequentialConfig()
    add_color.ensure_continuous_config(palette_config)
    check_arrays.ensure_dims(
        array=array_2d,
        num_dims=2,
    )
    array_view = as_plot_view(
        data_array=array_2d,
        data_format=data_format,
    )
    min_value, max_value = _get_value_range(
        array_2d=array_view,
        cbar_bounds=cbar_bounds,
    )
    palette = add_color.make_palette(
        config=palette_config,
        value_range=(min_value, max_value),
    )
    axis_extent = _as_axis_extent(axis_bounds)
    im_obj = ax.imshow(
        array_view,
        extent=axis_extent,
        aspect=axis_aspect_ratio,
        origin="lower",
        cmap=palette.mpl_cmap,
        norm=palette.mpl_norm,
    )
    if axis_extent is not None:
        min_x_value, max_x_value, min_y_value, max_y_value = axis_extent
        ax.set_xlim((min_x_value, max_x_value))
        ax.set_ylim((min_y_value, max_y_value))
    if add_cbar:
        add_color.add_colorbar(
            ax=ax,
            palette=palette,
            label=cbar_label,
            cbar_side=cbar_side,
        )
    return im_obj


def _generate_grid(
    field_shape: tuple[int, int],
    axis_extent: tuple[float, float, float, float],
) -> tuple[NDArray[Any], NDArray[Any]]:
    min_x_value, max_x_value, min_y_value, max_y_value = axis_extent
    num_rows, num_cols = field_shape
    coords_x = numpy.linspace(min_x_value, max_x_value, num_cols)
    coords_y = numpy.linspace(min_y_value, max_y_value, num_rows)
    grid_x, grid_y = numpy.meshgrid(coords_x, coords_y, indexing="xy")
    return grid_x, grid_y


def plot_2d_quiver(
    ax: manage_plots.PlotAxis,
    array_2d_rows: NDArray[Any],
    array_2d_cols: NDArray[Any],
    axis_bounds: AxisBounds = ((-1.0, 1.0), (-1.0, 1.0)),
    num_quivers: int = 25,
    quiver_width: float = 5e-3,
    color: str = "white",
):
    check_arrays.ensure_dims(
        array=array_2d_rows,
        num_dims=2,
    )
    check_arrays.ensure_dims(
        array=array_2d_cols,
        num_dims=2,
    )
    check_arrays.ensure_same_shape(
        array_a=array_2d_rows,
        array_b=array_2d_cols,
        param_name_a="array_2d_rows",
        param_name_b="array_2d_cols",
    )
    axis_extent = _as_axis_extent(axis_bounds)
    if axis_extent is None:
        raise ValueError("`axis_bounds` must not be None.")
    grid_x, grid_y = _generate_grid(
        field_shape=cast(tuple[int, int], array_2d_rows.shape),
        axis_extent=axis_extent,
    )
    quiver_step_rows = max(1, array_2d_rows.shape[0] // num_quivers)
    quiver_step_cols = max(1, array_2d_cols.shape[1] // num_quivers)
    quiver_obj = ax.quiver(
        grid_x[::quiver_step_rows, ::quiver_step_cols],
        grid_y[::quiver_step_rows, ::quiver_step_cols],
        array_2d_cols[::quiver_step_rows, ::quiver_step_cols],
        array_2d_rows[::quiver_step_rows, ::quiver_step_cols],
        width=quiver_width,
        color=color,
    )
    min_x_value, max_x_value, min_y_value, max_y_value = axis_extent
    ax.set_xlim((min_x_value, max_x_value))
    ax.set_ylim((min_y_value, max_y_value))
    return quiver_obj


def plot_2d_streamlines(
    ax: manage_plots.PlotAxis,
    array_2d_rows: NDArray[Any],
    array_2d_cols: NDArray[Any],
    axis_bounds: AxisBounds = ((0.0, 1.0), (0.0, 1.0)),
    streamline_width: float = 1.0,
    streamline_density: float = 2.0,
    arrow_size: float = 1.0,
    color: str = "white",
):
    check_arrays.ensure_dims(
        array=array_2d_rows,
        num_dims=2,
    )
    check_arrays.ensure_dims(
        array=array_2d_cols,
        num_dims=2,
    )
    check_arrays.ensure_same_shape(
        array_a=array_2d_rows,
        array_b=array_2d_cols,
        param_name_a="array_2d_rows",
        param_name_b="array_2d_cols",
    )
    axis_extent = _as_axis_extent(axis_bounds)
    if axis_extent is None:
        raise ValueError("`axis_bounds` must not be None.")
    grid_x, grid_y = _generate_grid(
        field_shape=cast(tuple[int, int], array_2d_rows.shape),
        axis_extent=axis_extent,
    )
    stream_obj = ax.streamplot(
        grid_x,
        grid_y,
        array_2d_cols,
        array_2d_rows,
        linewidth=streamline_width,
        density=streamline_density,
        arrowsize=arrow_size,
        color=color,
    )
    min_x_value, max_x_value, min_y_value, max_y_value = axis_extent
    ax.set_xlim((min_x_value, max_x_value))
    ax.set_ylim((min_y_value, max_y_value))
    return stream_obj


## } MODULE
