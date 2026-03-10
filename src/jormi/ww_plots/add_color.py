## { MODULE

##
## === DEPENDENCIES
##

import numpy
import cmasher
import matplotlib.axes as mpl_axes
import matplotlib.cm as mpl_cm
import matplotlib.colorbar as mpl_colorbar
import matplotlib.colors as mpl_colors

from dataclasses import dataclass

from jormi.ww_types import type_checks, box_positions


##
## === INTERNAL HELPERS
##


def _ensure_in_unit_interval(
    value: float,
    *,
    param_name: str,
) -> None:
    type_checks.ensure_finite_float(
        param=value,
        param_name=param_name,
        allow_none=False,
    )
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"`{param_name}` must lie in [0, 1], got {value}.")


def _ensure_ordered_pair(
    value_pair: tuple[float | int, float | int],
    *,
    param_name: str,
) -> None:
    min_value, max_value = value_pair
    type_checks.ensure_finite_scalar(
        param=min_value,
        param_name=f"{param_name}[0]",
        allow_none=False,
    )
    type_checks.ensure_finite_scalar(
        param=max_value,
        param_name=f"{param_name}[1]",
        allow_none=False,
    )
    if not (float(min_value) <= float(max_value)):
        raise ValueError(f"`{param_name}` must satisfy [0] <= [1], got {value_pair}.")


def _get_base_cmap(
    cmap_name: str,
) -> mpl_colors.Colormap:
    type_checks.ensure_nonempty_string(
        param=cmap_name,
        param_name="cmap_name",
    )
    if hasattr(CUSTOM_CMAPS, cmap_name):
        return getattr(CUSTOM_CMAPS, cmap_name)
    try:
        return mpl_cm.get_cmap(cmap_name)
    except ValueError:
        return cmasher.get_cmap(cmap_name)


def _subset_cmap(
    cmap: mpl_colors.Colormap,
    *,
    cmap_range: tuple[float, float],
    name: str,
) -> mpl_colors.Colormap:
    cmap_min, cmap_max = cmap_range
    _ensure_in_unit_interval(value=cmap_min, param_name="cmap_range[0]")
    _ensure_in_unit_interval(value=cmap_max, param_name="cmap_range[1]")
    _ensure_ordered_pair(cmap_range, param_name="cmap_range")
    if (cmap_min == 0.0) and (cmap_max == 1.0):
        return cmap
    sample = cmap(numpy.linspace(cmap_min, cmap_max, 256))
    return mpl_colors.LinearSegmentedColormap.from_list(
        name=f"{name}_sub",
        colors=sample,
        N=256,
    )


def _create_norm(
    *,
    value_range: tuple[float, float],
    mid_value: float | None,
) -> mpl_colors.Normalize:
    _ensure_ordered_pair(value_range, param_name="value_range")
    vmin, vmax = float(value_range[0]), float(value_range[1])
    if mid_value is None:
        return mpl_colors.Normalize(
            vmin=vmin,
            vmax=vmax,
        )
    type_checks.ensure_finite_float(
        param=mid_value,
        param_name="mid_value",
        allow_none=False,
    )
    vcenter = float(mid_value)
    if not (vmin < vcenter < vmax):
        raise ValueError(f"`mid_value` must satisfy vmin < mid_value < vmax, got ({vmin}, {vcenter}, {vmax}).")
    return mpl_colors.TwoSlopeNorm(
        vmin=vmin,
        vcenter=vcenter,
        vmax=vmax,
    )


##
## === DATA STRUCTURE
##


@dataclass(frozen=True)
class CMap:
    """
    User-facing configuration for a colormap + normalisation.

    Parameters
    ----------
    value_range:
        Data-space (vmin, vmax) tuple.
    cmap_name:
        Name of the base colormap. This can refer to:
          - a Matplotlib-registered colormap
          - a cmasher colormap
          - a name matching an entry in CUSTOM_CMAPS
    cmap_range:
        Colormap-space (min, max) tuple in [0, 1] used for subsetting.
    mid_value:
        Optional data-space midpoint. If provided, a TwoSlopeNorm is used
        with vcenter=mid_value.
    colors:
        Optional tuple of colors to build a custom colormap from. If provided,
        cmap_name is used as the name for the generated colormap.
    """
    value_range: tuple[float, float]
    cmap_name: str
    cmap_range: tuple[float, float] = (0.0, 1.0)
    mid_value: float | None = None
    colors: tuple[str, ...] | None = None

    @classmethod
    def from_colors(
        cls,
        *,
        value_range: tuple[float, float],
        colors: list[str],
        mid_value: float | None = None,
        name: str = "custom",
        cmap_range: tuple[float, float] = (0.0, 1.0),
    ) -> "CMap":
        return cls(
            value_range=value_range,
            cmap_name=name,
            cmap_range=cmap_range,
            mid_value=mid_value,
            colors=tuple(colors),
        )

    @property
    def cmap(self) -> mpl_colors.Colormap:
        if self.colors is not None:
            base = mpl_colors.LinearSegmentedColormap.from_list(
                name=self.cmap_name,
                colors=list(self.colors),
                N=256,
            )
        else:
            base = _get_base_cmap(self.cmap_name)
        return _subset_cmap(
            cmap=base,
            cmap_range=self.cmap_range,
            name=self.cmap_name,
        )

    @property
    def norm(self) -> mpl_colors.Normalize:
        return _create_norm(
            value_range=self.value_range,
            mid_value=self.mid_value,
        )


##
## === CUSTOM CMAPS
##


class CUSTOM_CMAPS:
    blue_red = mpl_colors.LinearSegmentedColormap.from_list(
        name="blue-red",
        colors=["#024f92", "#067bf1", "#d4d4d4", "#f65d25", "#A41409"],
        N=256,
    )
    white_brown = mpl_colors.LinearSegmentedColormap.from_list(
        name="white-brown",
        colors=["#fdfdfd", "#f49325", "#010101"],
        N=256,
    )
    purple_green = mpl_colors.LinearSegmentedColormap.from_list(
        name="purple-green",
        colors=["#68287d", "#d0a7c7", "#f2f0e0", "#d5e370", "#275b0e"],
        N=256,
    )


##
## === FUNCTIONS
##


def add_cbar_from_cmap(
    ax: mpl_axes.Axes,
    *,
    cmap: CMap,
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
        norm=cmap.norm,
        cmap=cmap.cmap,
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
