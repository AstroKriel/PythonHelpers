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
    color_palettes,
    manage_figure,
    style_figure,
)
from jormi.ww_validation import validate_arrays, validate_types
from jormi.ww_types import box_positions

##
## === DATA TYPES
##

DataFormat = Literal["xy", "ij"]
AxisRanges = tuple[
    tuple[float, float],  # (min_x_value, max_x_value)
    tuple[float, float],  # (min_y_value, max_y_value)
]

##
## === INTERNAL HELPERS
##


def as_plot_view(
    *,
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
    axis_ranges: AxisRanges | None,
) -> tuple[float, float, float, float] | None:
    """
    Convert AxisRanges to the flat (xmin, xmax, ymin, ymax) extent format expected by matplotlib.
    Returns None if `axis_ranges` is None.
    """
    if axis_ranges is None:
        return None
    validate_types.ensure_nested_tuple(
        param=axis_ranges,
        param_name="axis_ranges",
        outer_length=2,
        inner_length=2,
        valid_elem_types=validate_types.RuntimeTypes.Numerics.NumericLike,
        allow_none=False,
    )
    validate_types.ensure_ordered_pair(
        param=axis_ranges[0],
        param_name="axis_ranges[0]",
        allow_none=False,
        strict_ordering=True,
    )
    validate_types.ensure_ordered_pair(
        param=axis_ranges[1],
        param_name="axis_ranges[1]",
        allow_none=False,
        strict_ordering=True,
    )
    (min_x_value, max_x_value), (min_y_value, max_y_value) = axis_ranges
    return (
        float(min_x_value),
        float(max_x_value),
        float(min_y_value),
        float(max_y_value),
    )


def _get_value_range(
    *,
    array_2d: NDArray[Any],
    colorbar_range: tuple[float, float] | None,
) -> tuple[float, float]:
    """
    Calculate the (min, max) value range for colorbar scaling.

    If `colorbar_range` is provided, validate and use it directly. Otherwise, infer from the finite
    values in `array_2d`, with a small pad applied.
    """
    finite_mask = numpy.isfinite(array_2d)
    ## validate user supplied bounds and return directly
    if colorbar_range is not None:
        validate_types.ensure_ordered_pair(
            param=colorbar_range,
            param_name="colorbar_range",
            allow_none=False,
        )
        min_value, max_value = float(colorbar_range[0]), float(colorbar_range[1])
        if not (numpy.isfinite(min_value) and numpy.isfinite(max_value)):
            raise ValueError(f"`colorbar_range` must be finite, got ({min_value}, {max_value}).")
        in_range_mask = finite_mask & (array_2d >= min_value) & (array_2d <= max_value)
        if not numpy.any(in_range_mask):
            raise ValueError(f"`colorbar_range` ({min_value}, {max_value}) does not overlap with data.")
        return (
            min_value,
            max_value,
        )
    ## infer bounds from data, with a small pad to avoid degenerate colormaps
    if not numpy.any(finite_mask):
        raise ValueError("array contains no finite values; cannot infer colorbar bounds.")
    min_value = float(
        numpy.min(
            array_2d[finite_mask],
        ),
    )
    max_value = float(
        numpy.max(
            array_2d[finite_mask],
        ),
    )
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
    *,
    panel: manage_figure.Panel,
    array_2d: NDArray[Any],
    data_format: DataFormat,
    data_aspect_ratio: Literal["equal", "auto"] = "equal",
    axis_ranges: AxisRanges | None = None,
    colorbar_range: tuple[float, float] | None = None,
    palette_config: add_color.PaletteConfig | None = None,
    add_colorbar: bool = True,
    colorbar_label: str | None = None,
    colorbar_side: box_positions.Positions.PositionLike = box_positions.Positions.Side.Right,
    figure_params: style_figure.FigureParams | None = None,
) -> color_palettes.ColorPalette:
    """
    Draw `array_2d` onto `panel`, with a colorbar beside it unless one is turned down.

    Returns the palette it was drawn through, so a caller can key a colorbar of their own
    to it: a shared one across panels, or one placed where this function would not put it.
    """
    if palette_config is None:
        palette_config = add_color.SequentialConfig()
    validate_arrays.ensure_dims(
        array=array_2d,
        num_dims=2,
    )
    array_view = as_plot_view(
        data_array=array_2d,
        data_format=data_format,
    )
    ## a discrete palette is bounded by its own bin edges, so there is no range to take
    ## from the data, and none to accept from the caller either
    if isinstance(palette_config, add_color.DiscreteConfig):
        if colorbar_range is not None:
            raise ValueError(
                "`colorbar_range` cannot apply to a discrete palette; its `bin_edges`"
                " already bound it.",
            )
        palette = add_color.make_palette(config=palette_config)
    else:
        min_value, max_value = _get_value_range(
            array_2d=array_view,
            colorbar_range=colorbar_range,
        )
        palette = add_color.make_palette(
            config=palette_config,
            value_range=(min_value, max_value),
        )
    axis_extent = _as_axis_extent(axis_ranges)
    panel.imshow(
        array_view,
        extent=axis_extent,
        aspect=data_aspect_ratio,
        origin="lower",
        cmap=palette.mpl_cmap,
        norm=palette.mpl_norm,
    )
    if axis_extent is not None:
        min_x_value, max_x_value, min_y_value, max_y_value = axis_extent
        panel.set_xlim((min_x_value, max_x_value))
        panel.set_ylim((min_y_value, max_y_value))
    if add_colorbar:
        add_color.add_colorbar(
            panel=panel,
            palette=palette,
            label=colorbar_label,
            colorbar_side=colorbar_side,
            figure_params=figure_params,
        )
    return palette


def _generate_grid(
    *,
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
    *,
    panel: manage_figure.Panel,
    array_2d_rows: NDArray[Any],
    array_2d_cols: NDArray[Any],
    axis_ranges: AxisRanges = ((-1.0, 1.0), (-1.0, 1.0)),
    num_quivers: int = 25,
    quiver_width: float = 5e-3,
    color: str = "white",
):
    validate_arrays.ensure_dims(
        array=array_2d_rows,
        num_dims=2,
    )
    validate_arrays.ensure_dims(
        array=array_2d_cols,
        num_dims=2,
    )
    validate_arrays.ensure_same_shape(
        array_a=array_2d_rows,
        array_b=array_2d_cols,
        param_name_a="array_2d_rows",
        param_name_b="array_2d_cols",
    )
    axis_extent = _as_axis_extent(axis_ranges)
    if axis_extent is None:
        raise ValueError("`axis_ranges` must not be None.")
    grid_x, grid_y = _generate_grid(
        field_shape=cast(tuple[int, int], array_2d_rows.shape),
        axis_extent=axis_extent,
    )
    quiver_step_rows = max(1, array_2d_rows.shape[0] // num_quivers)
    quiver_step_cols = max(1, array_2d_cols.shape[1] // num_quivers)
    quiver_obj = panel.quiver(
        grid_x[::quiver_step_rows, ::quiver_step_cols],
        grid_y[::quiver_step_rows, ::quiver_step_cols],
        array_2d_cols[::quiver_step_rows, ::quiver_step_cols],
        array_2d_rows[::quiver_step_rows, ::quiver_step_cols],
        width=quiver_width,
        color=color,
    )
    min_x_value, max_x_value, min_y_value, max_y_value = axis_extent
    panel.set_xlim((min_x_value, max_x_value))
    panel.set_ylim((min_y_value, max_y_value))
    return quiver_obj


def plot_2d_streamlines(
    *,
    panel: manage_figure.Panel,
    array_2d_rows: NDArray[Any],
    array_2d_cols: NDArray[Any],
    axis_ranges: AxisRanges = ((0.0, 1.0), (0.0, 1.0)),
    streamline_width: float | None = None,
    streamline_density: float = 2.0,
    arrow_size: float = 0.5,
    color: str = "white",
    figure_params: style_figure.FigureParams | None = None,
):
    """`streamline_width` defaults to the width the active style draws data at."""
    if streamline_width is None:
        if figure_params is None:
            figure_params = style_figure.get_figure_params()
        streamline_width = figure_params.data_artist_params.line_width
    validate_arrays.ensure_dims(
        array=array_2d_rows,
        num_dims=2,
    )
    validate_arrays.ensure_dims(
        array=array_2d_cols,
        num_dims=2,
    )
    validate_arrays.ensure_same_shape(
        array_a=array_2d_rows,
        array_b=array_2d_cols,
        param_name_a="array_2d_rows",
        param_name_b="array_2d_cols",
    )
    axis_extent = _as_axis_extent(axis_ranges)
    if axis_extent is None:
        raise ValueError("`axis_ranges` must not be None.")
    grid_x, grid_y = _generate_grid(
        field_shape=cast(tuple[int, int], array_2d_rows.shape),
        axis_extent=axis_extent,
    )
    stream_obj = panel.streamplot(
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
    panel.set_xlim((min_x_value, max_x_value))
    panel.set_ylim((min_y_value, max_y_value))
    return stream_obj


def plot_2d_contours(
    *,
    panel: manage_figure.Panel,
    array_2d: NDArray[Any],
    data_format: DataFormat,
    axis_ranges: AxisRanges = ((-1.0, 1.0), (-1.0, 1.0)),
    levels: int | NDArray[Any] = 10,
    color: str = "white",
    linewidth: float | None = None,
    linestyle: str = "-",
    figure_params: style_figure.FigureParams | None = None,
):
    """`linewidth` defaults to the width the active style draws data at."""
    if linewidth is None:
        if figure_params is None:
            figure_params = style_figure.get_figure_params()
        linewidth = figure_params.data_artist_params.line_width
    validate_arrays.ensure_dims(
        array=array_2d,
        num_dims=2,
    )
    axis_extent = _as_axis_extent(axis_ranges)
    if axis_extent is None:
        raise ValueError("`axis_ranges` must not be None.")
    array_view = as_plot_view(data_array=array_2d, data_format=data_format)
    grid_x, grid_y = _generate_grid(
        field_shape=cast(tuple[int, int], array_view.shape),
        axis_extent=axis_extent,
    )
    contour_obj = panel.contour(
        grid_x,
        grid_y,
        array_view,
        levels=levels,
        colors=color,
        linewidths=linewidth,
        linestyles=linestyle,
    )
    min_x_value, max_x_value, min_y_value, max_y_value = axis_extent
    panel.set_xlim((min_x_value, max_x_value))
    panel.set_ylim((min_y_value, max_y_value))
    return contour_obj


## } MODULE
