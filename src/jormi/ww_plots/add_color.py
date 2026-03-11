## { MODULE

##
## === DEPENDENCIES
##

import matplotlib.axes as mpl_axes
import matplotlib.cm as mpl_cm
import matplotlib.colorbar as mpl_colorbar

from jormi.ww_types import type_checks, box_positions
from jormi.ww_plots.color_palette._base_palette import ColorPalette

##
## === ADD COLORBAR
##


def add_colorbar(
    ax: mpl_axes.Axes,
    *,
    palette: ColorPalette,
    label: str | None = None,
    anchor_side: box_positions.TypeHints.PositionLike = box_positions.TypeHints.Box.Side.Right,
    ax_percentage: float = 0.1,
    cbar_padding: float = 0.02,
    label_padding: float = 10.0,
    fontsize: float = 20.0,
) -> mpl_colorbar.Colorbar:
    anchor_side = box_positions.as_box_side(side=anchor_side)
    anchor_side_str = anchor_side.value
    type_checks.ensure_finite_float(
        param=ax_percentage,
        param_name="ax_percentage",
        allow_none=False,
        require_positive=True,
        allow_zero=False,
    )
    type_checks.ensure_finite_float(
        param=cbar_padding,
        param_name="cbar_padding",
        allow_none=False,
        require_positive=True,
        allow_zero=True,
    )
    type_checks.ensure_finite_float(
        param=label_padding,
        param_name="label_padding",
        allow_none=False,
        require_positive=True,
        allow_zero=True,
    )
    type_checks.ensure_finite_float(
        param=fontsize,
        param_name="fontsize",
        allow_none=False,
        require_positive=True,
        allow_zero=False,
    )
    fig = ax.figure
    box = ax.get_position()
    if anchor_side_str in ("left", "right"):
        orientation = "vertical"
        cbar_size = box.width * float(ax_percentage)
        if anchor_side_str == "right":
            cbar_bounds = [
                box.x1 + float(cbar_padding),
                box.y0,
                cbar_size,
                box.height,
            ]
        else:
            cbar_bounds = [
                box.x0 - cbar_size - float(cbar_padding),
                box.y0,
                cbar_size,
                box.height,
            ]
    else:
        orientation = "horizontal"
        cbar_size = box.height * float(ax_percentage)
        if anchor_side_str == "top":
            cbar_bounds = [
                box.x0,
                box.y1 + float(cbar_padding),
                box.width,
                cbar_size,
            ]
        else:
            cbar_bounds = [
                box.x0,
                box.y0 - cbar_size - float(cbar_padding),
                box.width,
                cbar_size,
            ]
    ax_cbar = fig.add_axes(cbar_bounds)
    mappable = mpl_cm.ScalarMappable(
        norm=palette.mpl_norm,
        cmap=palette.mpl_cmap,
    )
    mappable.set_array([])
    cbar = fig.colorbar(
        mappable,
        cax=ax_cbar,
        orientation=orientation,
    )
    if orientation == "horizontal":
        if label:
            cbar.set_label(
                label,
                fontsize=float(fontsize),
                labelpad=float(label_padding),
            )
            cbar.ax.xaxis.set_label_position(anchor_side_str)  # type: ignore[arg-type]
        cbar.ax.xaxis.set_ticks_position(anchor_side_str)  # type: ignore[arg-type]
    else:
        if label:
            cbar.set_label(
                label,
                fontsize=float(fontsize),
                labelpad=float(label_padding),
                rotation=90,
            )
            cbar.ax.yaxis.set_label_position(anchor_side_str)  # type: ignore[arg-type]
        cbar.ax.yaxis.set_ticks_position(anchor_side_str)  # type: ignore[arg-type]
        cbar.ax.get_yaxis().label.set_verticalalignment("center")
    return cbar


## } MODULE
