## { MODULE

##
## === DEPENDENCIES
##

import matplotlib.cm as mpl_cm
import matplotlib.axes as mpl_axes
import matplotlib.colorbar as mpl_colorbar

from jormi.ww_types import type_checks, box_positions
from jormi.ww_plots.color_palette import ColorPalette

##
## === INTERNAL HELPERS
##


def _compute_cbar_bounds(
    box,
    anchor_side_str: str,
    ax_percentage: float,
    cbar_padding: float,
) -> tuple[str, tuple[float, float, float, float]]:
    if anchor_side_str in ("left", "right"):
        orientation = "vertical"
        cbar_size = box.width * ax_percentage
        if anchor_side_str == "right":
            bounds = (
                box.x1 + cbar_padding,
                box.y0,
                cbar_size,
                box.height,
            )
        else:
            bounds = (
                box.x0 - cbar_size - cbar_padding,
                box.y0,
                cbar_size,
                box.height,
            )
    else:
        orientation = "horizontal"
        cbar_size = box.height * ax_percentage
        if anchor_side_str == "top":
            bounds = (
                box.x0,
                box.y1 + cbar_padding,
                box.width,
                cbar_size,
            )
        else:
            bounds = (
                box.x0,
                box.y0 - cbar_size - cbar_padding,
                box.width,
                cbar_size,
            )
    return orientation, bounds


def _apply_cbar_label(
    cbar: mpl_colorbar.Colorbar,
    *,
    label: str | None,
    orientation: str,
    anchor_side_str: str,
    fontsize: float,
    label_padding: float,
) -> None:
    if orientation not in ("horizontal", "vertical"):
        raise ValueError(f"Expected orientation to be 'horizontal' or 'vertical', got: {orientation!r}")
    if orientation == "horizontal":
        axis = cbar.ax.xaxis
        if label:
            cbar.set_label(label=label, fontsize=fontsize, labelpad=label_padding)
            axis.set_label_position(anchor_side_str)  # type: ignore[arg-type]
        axis.set_ticks_position(anchor_side_str)  # type: ignore[arg-type]
    else:
        axis = cbar.ax.yaxis
        if label:
            cbar.set_label(label=label, fontsize=fontsize, labelpad=label_padding, rotation=90)
            axis.set_label_position(anchor_side_str)  # type: ignore[arg-type]
        axis.set_ticks_position(anchor_side_str)  # type: ignore[arg-type]
        axis.label.set_verticalalignment("center")


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
    ## validate numeric params
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
    anchor_side = box_positions.as_box_side(side=anchor_side)
    anchor_side_str = anchor_side.value
    orientation, cbar_bounds = _compute_cbar_bounds(
        box=ax.get_position(),
        anchor_side_str=anchor_side_str,
        ax_percentage=float(ax_percentage),
        cbar_padding=float(cbar_padding),
    )
    ax_cbar = ax.figure.add_axes(cbar_bounds)
    mappable = mpl_cm.ScalarMappable(
        norm=palette.mpl_norm,
        cmap=palette.mpl_cmap,
    )
    mappable.set_array([])
    cbar = ax.figure.colorbar(
        mappable=mappable,
        cax=ax_cbar,
        orientation=orientation,
    )
    _apply_cbar_label(
        cbar,
        label=label,
        orientation=orientation,
        anchor_side_str=anchor_side_str,
        fontsize=float(fontsize),
        label_padding=float(label_padding),
    )
    return cbar


## } MODULE
