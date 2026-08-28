## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import typing

## third-party
import numpy

from matplotlib import collections as mpl_collections
from matplotlib import legend as mpl_legend
from matplotlib import lines as mpl_lines
from numpy import typing as numpy_typing

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
    x_pos_fraction: float,
    y_pos_fraction: float,
    label: str,
    x_alignment: box_positions.Positions.PositionLike = box_positions.Positions.Center.Center,
    y_alignment: box_positions.Positions.PositionLike = box_positions.Positions.Center.Center,
    text_size_pt: float | None = None,
    text_color: ColorType | None = None,
    box_alpha: float = 0.0,
    box_color: ColorType | None = None,
    box_margin_em: float = 0.3,
    box_corner_style: str = "round",
    rotate_deg: float | None = None,
    figure_params: style_figure.FigureParams | None = None,
):
    """
    Add a text label to a panel at a position given in panel coordinates [0, 1].
    A background box is drawn when `box_alpha > 0`.

    `text_size_pt` defaults to the active annotation text size, and the colours to the
    active theme, so a label stays legible when the theme changes. `box_margin_em`
    is the room the box leaves around its text, as a fraction of that text's size.
    """
    if figure_params is None:
        figure_params = style_figure.get_figure_params()
    theme_params = figure_params.theme_params
    if text_size_pt is None:
        text_size_pt = figure_params.text_size_params.annotation_size_pt
    if text_color is None:
        text_color = theme_params.foreground_color
    if box_color is None:
        box_color = theme_params.background_color
    ## validate position in panel coordinates [0, 1]
    validate_types.ensure_in_bounds(
        param=x_pos_fraction,
        param_name="x_pos_fraction",
        allow_none=False,
        min_value=0.0,
        max_value=1.0,
    )
    validate_types.ensure_in_bounds(
        param=y_pos_fraction,
        param_name="y_pos_fraction",
        allow_none=False,
        min_value=0.0,
        max_value=1.0,
    )
    ## validate text style
    validate_types.ensure_finite_scalar(
        param=text_size_pt,
        param_name="text_size_pt",
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
    ## Matplotlib spells the box's margin `pad` inside its style string, so the house name
    ## is what is written here and the translation happens at the boundary
    box_params = (
        dict(
            facecolor=box_color,
            edgecolor=theme_params.foreground_color,
            alpha=box_alpha,
            boxstyle=f"{box_corner_style},pad={box_margin_em}",
        ) if box_alpha > 0.0 else None
    )
    panel.text(
        x=x_pos_fraction,
        y=y_pos_fraction,
        s=label,
        ha=x_anchor.value,
        va=y_anchor.value,
        color=text_color,
        fontsize=text_size_pt,
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
    marker_size_pt: float | None = None,
    line_width_pt: float | None = None,
    text_size_pt: float | None = None,
    text_color: ColorType | None = None,
    anchor_point_fraction: tuple[float, float] = (1.0, 1.0),
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
    artist_params = figure_params.artist_params
    theme_params = figure_params.theme_params
    if text_size_pt is None:
        text_size_pt = text_size_params.legend_size_pt
    if text_color is None:
        text_color = theme_params.foreground_color
    if marker_size_pt is None:
        marker_size_pt = artist_params.marker_size_pt
    if line_width_pt is None:
        line_width_pt = artist_params.line_width_pt
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
        param=anchor_point_fraction,
        param_name="anchor_point_fraction",
        seq_length=2,
    )
    validate_types.ensure_in_bounds(
        param=anchor_point_fraction[0],
        param_name="anchor_point_fraction[0]",
        allow_none=False,
        min_value=0.0,
        max_value=1.0,
    )
    validate_types.ensure_in_bounds(
        param=anchor_point_fraction[1],
        param_name="anchor_point_fraction[1]",
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
            artist_to_draw = mpl_lines.Line2D(
                [0],
                [0],
                linestyle="",
                marker="",
            )
        elif artist in _VALID_MARKERS:
            artist_to_draw = mpl_lines.Line2D(
                [0],
                [0],
                marker=artist,
                color=color,
                linewidth=0,
                markeredgecolor=theme_params.foreground_color,
                markersize=marker_size_pt,
            )
        elif artist in _VALID_LINES:
            artist_to_draw = mpl_lines.Line2D(
                [0],
                [0],
                linestyle=artist,
                color=color,
                linewidth=line_width_pt,
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
    legend = mpl_legend.Legend(
        panel,
        handles=artists_to_draw,
        labels=labels,
        bbox_to_anchor=anchor_point_fraction,
        loc=anchor_at_corner.value,
        fontsize=text_size_pt,
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
    x_values: list[float] | numpy_typing.NDArray[typing.Any],
    y_values: list[float] | numpy_typing.NDArray[typing.Any],
    color: ColorType | None = None,
    linestyle: str = ":",
    linewidth_pt: float | None = None,
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
    if linewidth_pt is None:
        linewidth_pt = figure_params.artist_params.line_width_pt
    ## validate line style
    validate_types.ensure_finite_scalar(
        param=linewidth_pt,
        param_name="linewidth_pt",
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
    collection = mpl_collections.LineCollection(
        [numpy.column_stack((x_array, y_array))],
        colors=color,
        linestyles=linestyle,
        linewidths=linewidth_pt,
        alpha=alpha,
        zorder=zorder,
        label=label,
    )
    panel.add_collection(collection, autolim=False)


def add_shared_axis_label(
    *,
    panels: manage_figure.Panel | manage_figure.PanelGrid,
    label: str,
    side: box_positions.Positions.PositionLike = box_positions.Positions.Side.Left,
    gap_pt: float | None = None,
    text_size_pt: float | None = None,
    text_color: ColorType | None = None,
    figure_params: style_figure.FigureParams | None = None,
) -> None:
    """
    Add one axis label naming what several panels share, outside their own labels.

    Where it goes is not known when it is added, since it sits beyond tick labels that have
    yet to be drawn; the figure places it when it is fitted. `gap_pt` is the room left
    between it and those labels, defaulting to the gap the style leaves an axis label.
    """
    if figure_params is None:
        figure_params = style_figure.get_figure_params()
    if text_size_pt is None:
        text_size_pt = figure_params.text_size_params.axis_label_size_pt
    if text_color is None:
        text_color = figure_params.theme_params.foreground_color
    if gap_pt is None:
        gap_pt = figure_params.frame_params.axis_label_gap_pt
    validate_types.ensure_finite_float(
        param=gap_pt,
        param_name="gap_pt",
        allow_none=False,
        require_positive=True,
        allow_zero=True,
    )
    side = validate_box_positions.as_box_side(side=side)
    labelled_panels = manage_figure.as_panel_list(panels=panels)
    figure = manage_figure.get_figure(panels=labelled_panels)
    is_beside_panels = side in (
        box_positions.Positions.Side.Left,
        box_positions.Positions.Side.Right,
    )
    text = figure.text(
        0.5,
        0.5,
        label,
        ha="center",
        va="center",
        rotation=90.0 if is_beside_panels else 0.0,
        fontsize=text_size_pt,
        color=text_color,
    )
    manage_figure.register_shared_label(
        text=text,
        panels=labelled_panels,
        side=side,
        gap_pt=gap_pt,
    )


## } MODULE
