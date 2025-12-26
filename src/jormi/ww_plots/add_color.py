## { MODULE

##
## === DEPENDENCIES
##

import numpy
import cmasher
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors

from dataclasses import dataclass, field

from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

from jormi.ww_types import cardinal_anchors, type_checks

##
## === DATA STRUCTURE
##


@dataclass(frozen=True)
class CMap:
    """
    User-facing configuration for a colormap + normalisation.

    Parameters
    ----------
    min_value:
        Data-space minimum (vmin).
    max_value:
        Data-space maximum (vmax).
    cmap_name:
        Name of the base colormap. This can refer to:
          - a Matplotlib-registered colormap
          - a cmasher colormap
          - a name attached to a custom colormap from from_colors
          - a name matching an entry in CUSTOM_CMAPS
    min_cmap_value:
        Lower bound in colormap space (0-1) used for subsetting.
    max_cmap_value:
        Upper bound in colormap space (0-1) used for subsetting.
    mid_value:
        Optional data-space midpoint. If provided, a TwoSlopeNorm is used
        with vcenter=mid_value.
    """
    min_value: float
    max_value: float
    cmap_name: str
    min_cmap_value: float = 0.0
    max_cmap_value: float = 1.0
    mid_value: float | None = None
    _custom_cmap: mpl_colors.Colormap | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_colors(
        cls,
        *,
        min_value: float,
        max_value: float,
        colors: list[str],
        mid_value: float | None = None,
        name: str = "custom",
        min_cmap_value: float = 0.0,
        max_cmap_value: float = 1.0,
    ) -> "CMap":
        custom_cmap = LinearSegmentedColormap.from_list(
            name=name,
            colors=colors,
            N=256,
        )
        obj = cls(
            min_value=min_value,
            max_value=max_value,
            cmap_name=name,
            min_cmap_value=min_cmap_value,
            max_cmap_value=max_cmap_value,
            mid_value=mid_value,
        )
        object.__setattr__(obj, "_custom_cmap", custom_cmap)
        return obj

    @property
    def cmap(self) -> mpl_colors.Colormap:
        if self._custom_cmap is not None:
            return _subset_cmap(
                cmap=self._custom_cmap,
                min_cmap_value=self.min_cmap_value,
                max_cmap_value=self.max_cmap_value,
                name=self.cmap_name,
            )
        base_cmap = _get_base_cmap(self.cmap_name)
        return _subset_cmap(
            cmap=base_cmap,
            min_cmap_value=self.min_cmap_value,
            max_cmap_value=self.max_cmap_value,
            name=self.cmap_name,
        )

    @property
    def norm(self) -> mpl_colors.Normalize:
        return _create_norm(
            vmin=self.min_value,
            vmax=self.max_value,
            vcenter=self.mid_value,
        )


##
## === INTERNAL HELPERS
##


def _resolve_cbar_side(
    side: cardinal_anchors.AnchorLike,
) -> str:
    param_name = "<side>"
    edge_anchor = cardinal_anchors.as_edge_anchor(
        anchor=side,
        param_name=param_name,
    )
    if isinstance(edge_anchor, cardinal_anchors.VerticalAnchor):
        if cardinal_anchors.is_top_edge(edge_anchor, param_name=param_name):
            return "top"
        if cardinal_anchors.is_bottom_edge(edge_anchor, param_name=param_name):
            return "bottom"
    if isinstance(edge_anchor, cardinal_anchors.HorizontalAnchor):
        if cardinal_anchors.is_left_edge(edge_anchor, param_name=param_name):
            return "left"
        if cardinal_anchors.is_right_edge(edge_anchor, param_name=param_name):
            return "right"
    raise ValueError("Something went wrong.")


def _ensure_unit_interval(
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
    min_value: float,
    max_value: float,
    *,
    min_name: str,
    max_name: str,
) -> None:
    type_checks.ensure_finite_float(
        param=min_value,
        param_name=min_name,
        allow_none=False,
    )
    type_checks.ensure_finite_float(
        param=max_value,
        param_name=max_name,
        allow_none=False,
    )
    if not (float(min_value) <= float(max_value)):
        raise ValueError(f"`{min_name}` must be <= `{max_name}`, got ({min_value}, {max_value}).")


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
    min_cmap_value: float,
    max_cmap_value: float,
    name: str,
) -> mpl_colors.Colormap:
    _ensure_unit_interval(
        value=min_cmap_value,
        param_name="min_cmap_value",
    )
    _ensure_unit_interval(
        value=max_cmap_value,
        param_name="max_cmap_value",
    )
    if not (float(min_cmap_value) <= float(max_cmap_value)):
        raise ValueError(
            f"`min_cmap_value` must be <= `max_cmap_value`, got ({min_cmap_value}, {max_cmap_value}).",
        )
    if (float(min_cmap_value) == 0.0) and (float(max_cmap_value) == 1.0):
        return cmap
    sample = cmap(numpy.linspace(float(min_cmap_value), float(max_cmap_value), 256))
    return LinearSegmentedColormap.from_list(
        name=f"{name}_sub",
        colors=sample,
        N=256,
    )


def _create_norm(
    *,
    vmin: float,
    vmax: float,
    vcenter: float | None,
) -> mpl_colors.Normalize:
    _ensure_ordered_pair(
        min_value=vmin,
        max_value=vmax,
        min_name="vmin",
        max_name="vmax",
    )
    vmin = float(vmin)
    vmax = float(vmax)
    if vcenter is None:
        return mpl_colors.Normalize(
            vmin=vmin,
            vmax=vmax,
        )
    type_checks.ensure_finite_float(
        param=vcenter,
        param_name="vcenter",
        allow_none=False,
    )
    vcenter = float(vcenter)
    if not (vmin < vcenter < vmax):
        raise ValueError(f"`vcenter` must satisfy vmin < vcenter < vmax, got ({vmin}, {vcenter}, {vmax}).")
    return mpl_colors.TwoSlopeNorm(
        vmin=vmin,
        vcenter=vcenter,
        vmax=vmax,
    )


##
## === CUSTOM CMAPS
##


class CUSTOM_CMAPS:
    blue_red = LinearSegmentedColormap.from_list(
        name="blue-red",
        colors=["#024f92", "#067bf1", "#d4d4d4", "#f65d25", "#A41409"],
        N=256,
    )
    white_brown = LinearSegmentedColormap.from_list(
        name="white-brown",
        colors=["#fdfdfd", "#f49325", "#010101"],
        N=256,
    )
    purple_green = LinearSegmentedColormap.from_list(
        name="purple-green",
        colors=["#68287d", "#d0a7c7", "#f2f0e0", "#d5e370", "#275b0e"],
        N=256,
    )


##
## === FUNCTIONS
##


def create_cmap(
    *,
    cmap_name: str,
    vmin: float,
    vmax: float,
    vcenter: float | None = None,
    min_cmap_value: float = 0.0,
    max_cmap_value: float = 1.0,
) -> tuple[mpl_colors.Colormap, mpl_colors.Normalize]:
    _ensure_ordered_pair(
        min_value=vmin,
        max_value=vmax,
        min_name="vmin",
        max_name="vmax",
    )
    _ensure_unit_interval(
        value=min_cmap_value,
        param_name="min_cmap_value",
    )
    _ensure_unit_interval(
        value=max_cmap_value,
        param_name="max_cmap_value",
    )
    if not (float(min_cmap_value) <= float(max_cmap_value)):
        raise ValueError(
            f"`min_cmap_value` must be <= `max_cmap_value`, got ({min_cmap_value}, {max_cmap_value}).",
        )
    base_cmap = _get_base_cmap(cmap_name)
    cmap = _subset_cmap(
        cmap=base_cmap,
        min_cmap_value=min_cmap_value,
        max_cmap_value=max_cmap_value,
        name=cmap_name,
    )
    norm = _create_norm(
        vmin=vmin,
        vmax=vmax,
        vcenter=vcenter,
    )
    return cmap, norm


def add_cbar_from_cmap(
    ax,
    *,
    cmap: mpl_colors.Colormap | CMap,
    norm: mpl_colors.Normalize | None = None,
    label: str | None = None,
    side: cardinal_anchors.AnchorLike = "right",
    ax_percentage: float = 0.1,
    cbar_padding: float = 0.02,
    label_padding: float = 10.0,
    fontsize: float = 20.0,
):
    side_str = _resolve_cbar_side(side)
    type_checks.ensure_finite_float(param=ax_percentage, param_name="ax_percentage", allow_none=False, require_positive=True, allow_zero=False)
    type_checks.ensure_finite_float(param=cbar_padding, param_name="cbar_padding", allow_none=False, require_positive=True, allow_zero=True)
    type_checks.ensure_finite_float(param=label_padding, param_name="label_padding", allow_none=False, require_positive=True, allow_zero=True)
    type_checks.ensure_finite_float(param=fontsize, param_name="fontsize", allow_none=False, require_positive=True, allow_zero=False)
    if isinstance(cmap, CMap):
        cmap_obj = cmap.cmap
        norm_obj = cmap.norm
    else:
        cmap_obj = cmap
        if norm is None:
            raise ValueError("If `cmap` is not a CMap instance, you must provide `norm`.")
        norm_obj = norm
    fig = ax.figure
    box = ax.get_position()
    if side_str in ("left", "right"):
        orientation = "vertical"
        cbar_size = box.width * float(ax_percentage)
        if side_str == "right":
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
        if side_str == "top":
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
    mappable = ScalarMappable(
        norm=norm_obj,
        cmap=cmap_obj,
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
            cbar.ax.xaxis.set_label_position(side_str)
        cbar.ax.xaxis.set_ticks_position(side_str)
    else:
        if label:
            cbar.set_label(
                label,
                fontsize=float(fontsize),
                labelpad=float(label_padding),
                rotation=90,
            )
            cbar.ax.yaxis.set_label_position(side_str)
        cbar.ax.yaxis.set_ticks_position(side_str)
        cbar.ax.get_yaxis().label.set_verticalalignment("center")
    return cbar


## } MODULE
