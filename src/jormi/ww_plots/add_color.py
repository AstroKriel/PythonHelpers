## { MODULE

##
## === DEPENDENCIES
##

import cmasher
import matplotlib.colors as mpl_colors

from matplotlib.cm import ScalarMappable

##
## === FUNCTIONS
##


class CUSTOM_CMAPS:

    white_brown = LinearSegmentedColormap.from_list(
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


def create_norm(
    vmin: float = 0.0,
    vmid: float | None = None,
    vmax: float = 1.0,
):
    if vmid is not None:
        return mpl_colors.TwoSlopeNorm(vmin=vmin, vcenter=vmid, vmax=vmax)
    else:
        return mpl_colors.Normalize(vmin=vmin, vmax=vmax)


def create_cmap(
    cmap_name: str,
    cmin: float = 0.0,
    cmax: float = 1.0,
    vmin: float = 0.0,
    vmid: float | None = None,
    vmax: float = 1.0,
):
    cmap = cmasher.get_sub_cmap(cmap_name, cmin, cmax)
    norm = create_norm(vmin=vmin, vmid=vmid, vmax=vmax)
    return cmap, norm


def add_cbar_from_cmap(
    ax,
    cmap,
    norm,
    label: str | None = None,
    side: str = "right",
    ax_percentage: float = 0.1,
    cbar_padding: float = 0.02,
    label_padding: float = 10,
    fontsize: float = 20,
):
    fig = ax.figure
    box = ax.get_position()
    if side in ["left", "right"]:
        orientation = "vertical"
        cbar_size = box.width * ax_percentage
        if side == "right":
            cbar_bounds = [
                box.x1 + cbar_padding,
                box.y0,
                cbar_size,
                box.height,
            ]
        else:
            cbar_bounds = [
                box.x0 - cbar_size - cbar_padding,
                box.y0,
                cbar_size,
                box.height,
            ]
    elif side in ["top", "bottom"]:
        orientation = "horizontal"
        cbar_size = box.height * ax_percentage
        if side == "top":
            cbar_bounds = [
                box.x0,
                box.y1 + cbar_padding,
                box.width,
                cbar_size,
            ]
        else:
            cbar_bounds = [
                box.x0,
                box.y0 - cbar_size - cbar_padding,
                box.width,
                cbar_size,
            ]
    else:
        raise ValueError(f"Unsupported side: {side}")
    ax_cbar = fig.add_axes(cbar_bounds)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, cax=ax_cbar, orientation=orientation)
    if orientation == "horizontal":
        if label:
            cbar.set_label(label, fontsize=fontsize, labelpad=label_padding)
            cbar.ax.xaxis.set_label_position(side)
        cbar.ax.xaxis.set_ticks_position(side)
    else:
        if label:
            cbar.set_label(label, fontsize=fontsize, labelpad=label_padding, rotation=90)
            cbar.ax.yaxis.set_label_position(side)
        cbar.ax.yaxis.set_ticks_position(side)
        cbar.ax.get_yaxis().label.set_verticalalignment("center")  # optional
    return cbar


## } MODULE
