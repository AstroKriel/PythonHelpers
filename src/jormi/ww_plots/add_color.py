## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass

## third-party
import matplotlib.axes as mpl_axes
import matplotlib.cm as mpl_cm
import matplotlib.colorbar as mpl_colorbar

## local
from jormi.ww_plots.color_palettes import (
    ColorPalette,
    DiscretePalette,
    DivergingPalette,
    SequentialPalette,
)
from jormi.ww_plots.manage_plots import compute_adjacent_ax_bounds
from jormi.ww_validation import validate_box_positions, validate_python_types
from jormi.ww_types import box_positions

##
## === PALETTE CONFIGS
##


@dataclass
class SequentialConfig:
    """Lightweight config for a sequential (single-direction) palette."""
    palette_name: str = "cmr.arctic"
    palette_range: tuple[float, float] = (0.0, 1.0)


@dataclass
class DivergingConfig:
    """Lightweight config for a diverging (two-sided) palette."""
    mid_value: float = 0.0
    palette_name: str = "blue-white-red"
    palette_range: tuple[float, float] = (0.0, 1.0)


@dataclass
class DiscreteConfig:
    """Lightweight config for a discrete (binned) palette."""
    bin_edges: tuple[float, ...]
    palette_name: str = "cmr.arctic"
    palette_range: tuple[float, float] = (0.0, 1.0)


ContinuousPaletteConfig = SequentialConfig | DivergingConfig
PaletteConfig = SequentialConfig | DivergingConfig | DiscreteConfig


def ensure_sequential_config(
    config: PaletteConfig,
    param_name: str = "palette_config",
) -> None:
    """Raise TypeError if config is not a SequentialConfig."""
    if not isinstance(config, SequentialConfig):
        raise TypeError(f"`{param_name}` must be a SequentialConfig, got {type(config).__name__}.")


def ensure_diverging_config(
    config: PaletteConfig,
    param_name: str = "palette_config",
) -> None:
    """Raise TypeError if config is not a DivergingConfig."""
    if not isinstance(config, DivergingConfig):
        raise TypeError(f"`{param_name}` must be a DivergingConfig, got {type(config).__name__}.")


def ensure_continuous_config(
    config: PaletteConfig,
    param_name: str = "palette_config",
) -> None:
    """Raise TypeError if config is not a continuous palette config (sequential or diverging)."""
    if not isinstance(config, (SequentialConfig, DivergingConfig)):
        raise TypeError(
            f"`{param_name}` must be a continuous palette config (SequentialConfig or DivergingConfig), got {type(config).__name__}.",
        )


def ensure_discrete_config(
    config: PaletteConfig,
    param_name: str = "palette_config",
) -> None:
    """Raise TypeError if config is not a DiscreteConfig."""
    if not isinstance(config, DiscreteConfig):
        raise TypeError(f"`{param_name}` must be a DiscreteConfig, got {type(config).__name__}.")


def make_palette(
    config: PaletteConfig,
    value_range: tuple[float, float],
) -> ColorPalette:
    """
    Construct a ColorPalette from a PaletteConfig and a data-driven value range.
    For full control over palette construction, use the palette classes directly.
    """
    match config:
        case SequentialConfig():
            return SequentialPalette.from_name(
                palette_name=config.palette_name,
                palette_range=config.palette_range,
                value_range=value_range,
            )
        case DivergingConfig():
            return DivergingPalette.from_name(
                palette_name=config.palette_name,
                palette_range=config.palette_range,
                value_range=value_range,
                mid_value=config.mid_value,
            )
        case DiscreteConfig():
            return DiscretePalette.from_name(
                palette_name=config.palette_name,
                palette_range=config.palette_range,
                bin_edges=config.bin_edges,
            )


##
## === INTERNAL HELPERS
##

_Side = box_positions.Positions.Side

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
    label_size: int | float,
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
            axis.set_label_position(cbar_side)  # pyright: ignore[reportArgumentType]
        axis.set_ticks_position(cbar_side)  # pyright: ignore[reportArgumentType]
        axis.label.set_verticalalignment("center")
    elif cbar_side in (_Side.Top, _Side.Bottom):
        axis = cbar.ax.xaxis
        if label:
            cbar.set_label(
                label=label,
                fontsize=label_size,
                labelpad=label_pad,
            )
            axis.set_label_position(cbar_side)  # pyright: ignore[reportArgumentType]
        axis.set_ticks_position(cbar_side)  # pyright: ignore[reportArgumentType]
    else:
        raise ValueError(f"unexpected cbar_side: {cbar_side!r}.")  # pyright: ignore[reportUnreachable]


##
## === ADD COLORBAR
##


def add_colorbar(
    ax: mpl_axes.Axes,
    *,
    palette: ColorPalette,
    label: str | None = None,
    cbar_side: box_positions.Positions.PositionLike = box_positions.Positions.Side.Right,
    cbar_thickness: float = 0.1,
    cbar_length: float = 1.0,
    cbar_pad: float = 0.02,
    label_pad: float = 10.0,
    label_size: int | float = 20.0,
) -> mpl_colorbar.Colorbar:
    ## validate numeric params
    validate_python_types.ensure_finite_float(
        param=cbar_thickness,
        param_name="cbar_thickness",
        allow_none=False,
        require_positive=True,
        allow_zero=False,
    )
    validate_python_types.ensure_finite_float(
        param=cbar_pad,
        param_name="cbar_pad",
        allow_none=False,
        require_positive=True,
        allow_zero=True,
    )
    validate_python_types.ensure_finite_float(
        param=label_pad,
        param_name="label_pad",
        allow_none=False,
        require_positive=True,
        allow_zero=True,
    )
    validate_python_types.ensure_finite_scalar(
        param=label_size,
        param_name="label_size",
        allow_none=False,
        require_positive=True,
        allow_zero=False,
    )
    cbar_side = validate_box_positions.as_box_side(side=cbar_side)
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
