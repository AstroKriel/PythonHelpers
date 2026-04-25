## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Any

## third-party
import numpy
from numpy.typing import NDArray

from matplotlib.collections import LineCollection
from matplotlib.legend import Legend as mpl_legend
from matplotlib.lines import Line2D as mpl_line2d

## local
from jormi.ww_plots import manage_plots
from jormi.ww_validation import validate_arrays, validate_box_positions, validate_types
from jormi.ww_types import box_positions

##
## === COLOUR TYPE
##

ColorType = str | tuple[float, float, float] | tuple[float, float, float, float]

##
## === VALID ARTISTS
##

_VALID_MARKERS: list[str] = [
    ".",  # point
    "o",  # circle
    "s",  # square
    "D",  # diamond
    "^",  # triangle up
    "v",  # triangle down
]
_VALID_LINES: list[str] = [
    "-",  # solid
    "--",  # dashed
    "-.",  # dash-dot
    ":",  # dotted
]

##
## === AXIS ANNOTATIONS
##


def add_text(
    *,
    ax: manage_plots.PlotAxis,
    x_pos: float,
    y_pos: float,
    label: str,
    x_alignment: box_positions.Positions.PositionLike = box_positions.Positions.Center.Center,
    y_alignment: box_positions.Positions.PositionLike = box_positions.Positions.Center.Center,
    text_size: float = 20,
    text_color: ColorType = "black",
    box_alpha: float = 0.0,
    box_color: ColorType = "white",
    rotate_deg: float | None = None,
):
    """
    Add a text label to an axis at a position given in axes coordinates [0, 1].
    A background box is drawn when `box_alpha > 0`.
    """
    ## validate position in axes coordinates [0, 1]
    validate_types.ensure_in_bounds(
        param=x_pos,
        param_name="x_pos",
        allow_none=False,
        min_value=0.0,
        max_value=1.0,
    )
    validate_types.ensure_in_bounds(
        param=y_pos,
        param_name="y_pos",
        allow_none=False,
        min_value=0.0,
        max_value=1.0,
    )
    ## validate text style
    validate_types.ensure_finite_scalar(
        param=text_size,
        param_name="text_size",
        allow_none=False,
        require_positive=True,
        allow_zero=False,
    )
    ## validate box opacity; box is not drawn if alpha is zero
    validate_types.ensure_in_bounds(
        param=box_alpha,
        param_name="box_alpha",
        allow_none=False,
        min_value=0.0,
        max_value=1.0,
    )
    ## validate optional rotation
    validate_types.ensure_finite_float(
        param=rotate_deg,
        param_name="rotate_deg",
        allow_none=True,
    )
    x_anchor = validate_box_positions.as_mpl_ha(x_alignment)
    y_anchor = validate_box_positions.as_mpl_va(y_alignment)
    box_params = (
        dict(
            facecolor=box_color,
            edgecolor="black",
            alpha=box_alpha,
            boxstyle="round,pad=0.3",
        ) if box_alpha > 0.0 else None
    )
    ax.text(
        x=x_pos,
        y=y_pos,
        s=label,
        ha=x_anchor.value,
        va=y_anchor.value,
        color=text_color,
        fontsize=text_size,
        rotation=rotate_deg,
        transform=ax.transAxes,
        bbox=box_params,
    )


def add_custom_legend(
    *,
    ax: manage_plots.PlotAxis,
    artists: list[str],
    labels: list[str],
    colors: list[ColorType],
    marker_size: float = 8,
    line_width: float = 1.5,
    text_size: float = 16,
    text_color: ColorType = "black",
    anchor_point: tuple[float, float] = (1.0, 1.0),
    anchor_at_corner: box_positions.Positions.PositionLike = box_positions.Positions.Corner.TopRight,
    frame_alpha: float = 0.0,
    num_cols: int = 1,
    spacing: float = 0.5,
    marker_first: bool = True,
):
    """
    Add a custom legend to an axis, built from explicit style strings rather than plot handles.

    Each entry in `artists` must be a marker (e.g. "o", "s") or line style (e.g. "-", "--"),
    paired with the corresponding entry in `labels` and `colors`. A legend frame is drawn
    when `frame_alpha > 0`.
    """
    ## validate parallel lists
    validate_types.ensure_list_of_strings(
        param=artists,
        param_name="artists",
    )
    validate_types.ensure_list_of_strings(
        param=labels,
        param_name="labels",
    )
    validate_types.ensure_sequence(
        param=colors,
        param_name="colors",
        valid_seq_types=list,
        valid_elem_types=(str, tuple),
    )
    if len(artists) != len(labels) or len(artists) != len(colors):
        raise ValueError("`artists`, `labels`, and `colors` must all have the same length.")
    ## validate frame opacity; frame is skipped when alpha is zero
    validate_types.ensure_in_bounds(
        param=frame_alpha,
        param_name="frame_alpha",
        allow_none=False,
        min_value=0.0,
        max_value=1.0,
    )
    ## validate anchor position in axes coordinates [0, 1]
    validate_types.ensure_tuple_of_numbers(
        param=anchor_point,
        param_name="anchor_point",
        seq_length=2,
    )
    validate_types.ensure_in_bounds(
        param=anchor_point[0],
        param_name="anchor_point[0]",
        allow_none=False,
        min_value=0.0,
        max_value=1.0,
    )
    validate_types.ensure_in_bounds(
        param=anchor_point[1],
        param_name="anchor_point[1]",
        allow_none=False,
        min_value=0.0,
        max_value=1.0,
    )
    anchor_at_corner = validate_box_positions.as_mpl_anchor(position=anchor_at_corner)
    ## build artist handles from style strings
    artists_to_draw = []
    for artist, color in zip(artists, colors):
        if artist in _VALID_MARKERS:
            artist_to_draw = mpl_line2d(
                [0],
                [0],
                marker=artist,
                color=color,
                linewidth=0,
                markeredgecolor="black",
                markersize=marker_size,
            )
        elif artist in _VALID_LINES:
            artist_to_draw = mpl_line2d(
                [0],
                [0],
                linestyle=artist,
                color=color,
                linewidth=line_width,
            )
        else:
            raise ValueError(
                f"Artist `{artist}` is not a recognized marker or line style.\n"
                f"\t- Valid markers: {_VALID_MARKERS}.\n"
                f"\t- Valid line styles: {_VALID_LINES}.",
            )
        artists_to_draw.append(artist_to_draw)
    ## draw legend; use Legend directly so multiple legends can coexist on the same axis
    legend = mpl_legend(
        ax,
        handles=artists_to_draw,
        labels=labels,
        bbox_to_anchor=anchor_point,
        loc=anchor_at_corner.value,
        fontsize=text_size,
        labelcolor=text_color,
        frameon=(frame_alpha > 0.0),
        framealpha=frame_alpha,
        facecolor="white",
        edgecolor="black",
        ncol=num_cols,
        borderpad=0.45,
        handletextpad=spacing,
        labelspacing=spacing,
        columnspacing=spacing,
        markerfirst=marker_first,
    )
    ax.add_artist(legend)


def overlay_curve(
    *,
    ax: manage_plots.PlotAxis,
    x_values: list[float] | NDArray[Any],
    y_values: list[float] | NDArray[Any],
    color: ColorType = "black",
    linestyle: str = ":",
    linewidth: float = 1.0,
    label: str | None = None,
    alpha: float = 1.0,
    zorder: float = 1.0,
):
    """
    Overlay a 2D curve onto an axis without affecting its axis limits.

    `x_values` and `y_values` must be 1D and the same length, with at least two points.
    """
    ## validate line style
    validate_types.ensure_finite_scalar(
        param=linewidth,
        param_name="linewidth",
        allow_none=False,
        require_positive=True,
        allow_zero=False,
    )
    validate_types.ensure_in_bounds(
        param=alpha,
        param_name="alpha",
        allow_none=False,
        min_value=0.0,
        max_value=1.0,
    )
    ## validate curve data
    x_array = validate_arrays.as_1d(
        array_like=x_values,
        param_name="x_values",
    )
    y_array = validate_arrays.as_1d(
        array_like=y_values,
        param_name="y_values",
    )
    validate_arrays.ensure_same_shape(
        array_a=x_array,
        array_b=y_array,
        param_name_a="x_values",
        param_name_b="y_values",
    )
    if x_array.size < 2:
        raise ValueError("need at least 2 points to plot a curve.")
    collection = LineCollection(
        [numpy.column_stack((x_array, y_array))],
        colors=color,
        linestyles=linestyle,
        linewidths=linewidth,
        alpha=alpha,
        zorder=zorder,
        label=label,
    )
    ax.add_collection(collection, autolim=False)


## } MODULE
