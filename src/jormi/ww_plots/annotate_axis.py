## { MODULE

##
## === DEPENDENCIES
##

import numpy

from matplotlib.lines import Line2D as mpl_line2d
from matplotlib.collections import LineCollection

from jormi.ww_types import cardinal_anchors, ordinal_anchors, array_checks, type_checks
from jormi.ww_plots import plot_manager

##
## === FUNCTIONS
##


def add_text(
    ax: plot_manager.PlotAxis,
    x_pos: float,
    y_pos: float,
    label: str,
    x_alignment: cardinal_anchors.HorizontalAnchorLike = cardinal_anchors.HorizontalAnchor.Center,
    y_alignment: cardinal_anchors.VerticalAnchorLike = cardinal_anchors.VerticalAnchor.Center,
    fontsize: float = 20,
    font_color: str = "black",
    add_box: bool = False,
    box_alpha: float = 0.8,
    face_color: str = "white",
    edge_color: str = "black",
    rotate_deg: float | None = None,
):
    x_anchor = cardinal_anchors.as_horizontal_anchor(x_alignment)
    y_anchor = cardinal_anchors.as_vertical_anchor(y_alignment)
    box_params = dict(
        facecolor=face_color,
        edgecolor=edge_color,
        alpha=box_alpha,
        boxstyle="round,pad=0.3",
    )
    ax.text(
        x=x_pos,
        y=y_pos,
        s=label,
        ha=x_anchor.value,
        va=y_anchor.value,
        color=font_color,
        fontsize=fontsize,
        rotation=rotate_deg,
        transform=ax.transAxes,
        bbox=box_params if add_box else None,
    )


def add_custom_legend(
    ax: plot_manager.PlotAxis,
    artists: list[str],
    labels: list[str],
    colors: list[str],
    marker_size: float = 8,
    line_width: float = 1.5,
    fontsize: float = 16,
    text_color: str = "black",
    anchor_corner: ordinal_anchors.CornerAnchorLike = ordinal_anchors.CornerAnchor.TopRight,
    anchor_point: tuple[float, float] = (1.0, 1.0),
    enable_frame: bool = False,
    frame_alpha: float = 0.5,
    face_color: str = "white",
    edge_color: str = "grey",
    num_cols: int = 1,
    text_padding: float = 0.5,
    label_spacing: float = 0.5,
    column_spacing: float = 0.5,
    put_label_first: bool = False,
):
    type_checks.ensure_list_of_strings(
        param=artists,
        param_name="artists",
    )
    type_checks.ensure_list_of_strings(
        param=labels,
        param_name="labels",
    )
    type_checks.ensure_list_of_strings(
        param=colors,
        param_name="colors",
    )
    if len(artists) != len(labels) or len(artists) != len(colors):
        raise ValueError("artists, labels, and colors must have the same length.")
    type_checks.ensure_tuple_of_numbers(
        param=anchor_point,
        param_name="anchor_point",
        seq_length=2,
    )
    anchor_corner = ordinal_anchors.as_corner_anchor(anchor_corner)
    artists_to_draw = []
    valid_markers = [".", "o", "s", "D", "^", "v"]
    valid_lines = ["-", "--", "-.", ":"]
    for artist, color in zip(artists, colors):
        if artist in valid_markers:
            artist_to_draw = mpl_line2d(
                [0],
                [0],
                marker=artist,
                color=color,
                linewidth=0,
                markeredgecolor="black",
                markersize=marker_size,
            )
        elif artist in valid_lines:
            artist_to_draw = mpl_line2d(
                [0],
                [0],
                linestyle=artist,
                color=color,
                linewidth=line_width,
            )
        else:
            raise ValueError(
                f"Artist = `{artist}` is not a recognized marker or line style.\n"
                f"\t- Valid markers: {valid_markers}.\n"
                f"\t- Valid line styles: {valid_lines}.",
            )
        artists_to_draw.append(artist_to_draw)
    legend = ax.legend(
        handles=artists_to_draw,
        labels=labels,
        bbox_to_anchor=anchor_point,
        loc=anchor_corner.value,
        fontsize=fontsize,
        labelcolor=text_color,
        frameon=enable_frame,
        framealpha=frame_alpha,
        facecolor=face_color,
        edgecolor=edge_color,
        ncol=num_cols,
        borderpad=0.45,
        handletextpad=text_padding,
        labelspacing=label_spacing,
        columnspacing=column_spacing,
        markerfirst=not (put_label_first),
    )
    ax.add_artist(legend)
    return legend


def overlay_curve(
    ax: plot_manager.PlotAxis,
    x_values: list[float] | numpy.ndarray,
    y_values: list[float] | numpy.ndarray,
    color: str = "black",
    linestyle: str = ":",
    linewidth: float = 1.0,
    label: str | None = None,
    alpha: float = 1.0,
    zorder: float = 1.0,
):
    x_array = array_checks.as_1d(
        array_like=x_values,
        param_name="x_values",
    )
    y_array = array_checks.as_1d(
        array_like=y_values,
        param_name="y_values",
    )
    array_checks.ensure_same_shape(
        array_a=x_array,
        array_b=y_array,
        param_name_a="x_values",
        param_name_b="y_values",
    )
    if x_array.size < 2:
        raise ValueError("There needs to be at least two points to plot a curve.")
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
