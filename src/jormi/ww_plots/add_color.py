## { MODULE

##
## === DEPENDENCIES
##

import matplotlib.cm as mpl_cm
import matplotlib.axes as mpl_axes
import matplotlib.colorbar as mpl_colorbar

from jormi.ww_types import type_checks, box_positions
from jormi.ww_plots.color_palette import ColorPalette
from jormi.ww_plots.plot_manager import compute_adjacent_ax_bounds

##
## === INTERNAL HELPERS
##

_Side = box_positions.TypeHints.Box.Side

_SIDE_TO_ORIENTATION: dict[_Side, str] = {
    _Side.Top: "horizontal",
    _Side.Left: "vertical",
    _Side.Right: "vertical",
    _Side.Bottom: "horizontal",
}


def _label_cbar(
    cbar: mpl_colorbar.Colorbar,
    *,
    label: str | None,
    cbar_side: _Side,
    label_size: float,
    label_pad: float,
) -> None:
    if cbar_side in (_Side.Left, _Side.Right):
        axis = cbar.ax.yaxis
        if label:
            cbar.set_label(
                label=label,
                fontsize=label_size,
                labelpad=label_pad,
                rotation=90,
            )
            axis.set_label_position(cbar_side)  # type: ignore[arg-type]
        axis.set_ticks_position(cbar_side)  # type: ignore[arg-type]
        axis.label.set_verticalalignment("center")
    elif cbar_side in (_Side.Top, _Side.Bottom):
        axis = cbar.ax.xaxis
        if label:
            cbar.set_label(
                label=label,
                fontsize=label_size,
                labelpad=label_pad,
            )
            axis.set_label_position(cbar_side)  # type: ignore[arg-type]
        axis.set_ticks_position(cbar_side)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unexpected cbar_side: {cbar_side!r}")  # type: ignore[unreachable]


##
## === ADD COLORBAR
##


def add_colorbar(
    ax: mpl_axes.Axes,
    *,
    palette: ColorPalette,
    label: str | None = None,
    cbar_side: box_positions.TypeHints.PositionLike = box_positions.TypeHints.Box.Side.Right,
    cbar_thickness: float = 0.1,
    cbar_length: float = 1.0,
    cbar_pad: float = 0.02,
    label_pad: float = 10.0,
    label_size: float = 20.0,
) -> mpl_colorbar.Colorbar:
    ## validate numeric params
    type_checks.ensure_finite_float(
        param=cbar_thickness,
        param_name="cbar_thickness",
        allow_none=False,
        require_positive=True,
        allow_zero=False,
    )
    type_checks.ensure_finite_float(
        param=cbar_pad,
        param_name="cbar_pad",
        allow_none=False,
        require_positive=True,
        allow_zero=True,
    )
    type_checks.ensure_finite_float(
        param=label_pad,
        param_name="label_pad",
        allow_none=False,
        require_positive=True,
        allow_zero=True,
    )
    type_checks.ensure_finite_float(
        param=label_size,
        param_name="label_size",
        allow_none=False,
        require_positive=True,
        allow_zero=False,
    )
    cbar_side = box_positions.as_box_side(side=cbar_side)
    cbar_orientation = _SIDE_TO_ORIENTATION[cbar_side]
    ax_bounds = compute_adjacent_ax_bounds(
        ax=ax,
        side=cbar_side,
        thickness=cbar_thickness,
        length=cbar_length,
        gap=cbar_pad,
    )
    cbar_ax = ax.figure.add_axes((
        ax_bounds.x_min,
        ax_bounds.y_min,
        ax_bounds.x_width,
        ax_bounds.y_width,
    ))
    cbar_mappable = mpl_cm.ScalarMappable(
        norm=palette.mpl_norm,
        cmap=palette.mpl_cmap,
    )
    ## required by mpl to suppress warning when ScalarMappable has no data
    cbar_mappable.set_array([])
    cbar = ax.figure.colorbar(
        mappable=cbar_mappable,
        cax=cbar_ax,
        orientation=cbar_orientation,
    )
    _label_cbar(
        cbar,
        label=label,
        cbar_side=cbar_side,
        label_size=label_size,
        label_pad=label_pad,
    )
    return cbar


## } MODULE
