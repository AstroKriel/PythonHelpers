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
from jormi.ww_plots import manage_figure, style_figure
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
    panel: manage_figure.Panel,
    x_pos: float,
    y_pos: float,
    label: str,
    x_alignment: box_positions.Positions.PositionLike = box_positions.Positions.Center.Center,
    y_alignment: box_positions.Positions.PositionLike = box_positions.Positions.Center.Center,
    text_size: float | None = None,
    text_color: ColorType | None = None,
    box_alpha: float = 0.0,
    box_color: ColorType | None = None,
    rotate_deg: float | None = None,
    figure_params: style_figure.FigureParams | None = None,
):
    """
    Add a text label to a panel at a position given in panel coordinates [0, 1].
    A background box is drawn when `box_alpha > 0`.

    `text_size` defaults to the active annotation text size, and the colours to the
    active theme, so a label stays legible when the theme changes.
    """
    if figure_params is None:
        figure_params = style_figure.get_figure_params()
    theme_params = figure_params.theme_params
    if text_size is None:
        text_size = figure_params.text_size_params.annotation_size
    if text_color is None:
        text_color = theme_params.foreground_color
    if box_color is None:
        box_color = theme_params.background_color
    ## validate position in panel coordinates [0, 1]
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
            edgecolor=theme_params.foreground_color,
            alpha=box_alpha,
            boxstyle="round,pad=0.3",
        ) if box_alpha > 0.0 else None
    )
    panel.text(
        x=x_pos,
        y=y_pos,
        s=label,
        ha=x_anchor.value,
        va=y_anchor.value,
        color=text_color,
        fontsize=text_size,
        rotation=rotate_deg,
        transform=panel.transAxes,
        bbox=box_params,
    )


def add_custom_legend(
    *,
    panel: manage_figure.Panel,
    artists: list[str | None],
    labels: list[str],
    colors: list[ColorType],
    marker_size: float | None = None,
    line_width: float | None = None,
    text_size: float | None = None,
    text_color: ColorType | None = None,
    anchor_point: tuple[float, float] = (1.0, 1.0),
    anchor_at_corner: box_positions.Positions.PositionLike = box_positions.Positions.Corner.TopRight,
    frame_alpha: float = 0.0,
    num_legend_columns: int = 1,
    marker_first: bool = True,
    figure_params: style_figure.FigureParams | None = None,
):
    """
    Add a custom legend to a panel, built from explicit style strings rather than plot handles.

    Each entry in `artists` is a marker (e.g. "o", "s"), a line style (e.g. "-", "--"), or
    `None` for an entry that is text alone, paired with the corresponding entry in `labels`
    and `colors`. Every entry carries a colour: it draws the swatch where there is one, and
    the label itself where there is not, so a legend can key shapes and colours separately.
    A legend frame is drawn when `frame_alpha > 0`.

    Everything left unset is taken from the active style: the text size, the colours, and
    the marker and line sizes the data is drawn with, so a swatch matches what it stands
    for. How tightly the legend packs is left to Matplotlib to read from the rcParams the
    style sets.
    """
    if figure_params is None:
        figure_params = style_figure.get_figure_params()
    text_size_params = figure_params.text_size_params
    data_artist_params = figure_params.data_artist_params
    theme_params = figure_params.theme_params
    if text_size is None:
        text_size = text_size_params.legend_size
    if text_color is None:
        text_color = theme_params.foreground_color
    if marker_size is None:
        marker_size = data_artist_params.marker_size
    if line_width is None:
        line_width = data_artist_params.line_width
    ## validate parallel lists; an artist may be None, for an entry that is text alone
    validate_types.ensure_sequence(
        param=artists,
        param_name="artists",
        valid_seq_types=list,
        valid_elem_types=(str, type(None)),
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
    ## validate anchor position in panel coordinates [0, 1]
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
    ## build artist handles from style strings, and colour each label by whether its entry
    ## draws a swatch to carry the colour instead
    artists_to_draw = []
    label_colors: list[ColorType] = []
    for artist, color in zip(artists, colors):
        label_colors.append(text_color if artist is not None else color)
        if artist is None:
            artist_to_draw = mpl_line2d(
                [0],
                [0],
                linestyle="",
                marker="",
            )
        elif artist in _VALID_MARKERS:
            artist_to_draw = mpl_line2d(
                [0],
                [0],
                marker=artist,
                color=color,
                linewidth=0,
                markeredgecolor=theme_params.foreground_color,
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
                f"\t- Valid line styles: {_VALID_LINES}.\n"
                "\t- Use None for an entry that is text alone.",
            )
        artists_to_draw.append(artist_to_draw)
    ## Matplotlib holds the handle column open at its configured width whatever the handle
    ## draws, so close it when no entry has a swatch to put there; None leaves both to the
    ## rcParams the style sets
    has_no_swatches = all(artist is None for artist in artists)
    handle_length = 0.0 if has_no_swatches else None
    handle_gap = 0.0 if has_no_swatches else None
    ## draw legend; use Legend directly so multiple legends can coexist on the same panel
    legend = mpl_legend(
        panel,
        handles=artists_to_draw,
        labels=labels,
        bbox_to_anchor=anchor_point,
        loc=anchor_at_corner.value,
        fontsize=text_size,
        labelcolor=label_colors,
        frameon=(frame_alpha > 0.0),
        framealpha=frame_alpha,
        facecolor=theme_params.background_color,
        edgecolor=theme_params.foreground_color,
        ncol=num_legend_columns,
        markerfirst=marker_first,
        handlelength=handle_length,
        handletextpad=handle_gap,
    )
    panel.add_artist(legend)


def overlay_curve(
    *,
    panel: manage_figure.Panel,
    x_values: list[float] | NDArray[Any],
    y_values: list[float] | NDArray[Any],
    color: ColorType | None = None,
    linestyle: str = ":",
    linewidth: float | None = None,
    label: str | None = None,
    alpha: float = 1.0,
    zorder: float = 1.0,
    figure_params: style_figure.FigureParams | None = None,
):
    """
    Overlay a 2D curve onto a panel without affecting its axis limits.

    `x_values` and `y_values` must be 1D and the same length, with at least two points.
    The colour and width default to the active style's.
    """
    if figure_params is None:
        figure_params = style_figure.get_figure_params()
    if color is None:
        color = figure_params.theme_params.foreground_color
    if linewidth is None:
        linewidth = figure_params.data_artist_params.line_width
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
    panel.add_collection(collection, autolim=False)


## } MODULE
