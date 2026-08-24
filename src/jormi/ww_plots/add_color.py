## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass

## third-party
import matplotlib.cm as mpl_cm
import matplotlib.colorbar as mpl_colorbar

## local
from jormi.ww_plots import color_palettes, manage_figure, style_figure
from jormi.ww_types import box_positions
from jormi.ww_validation import validate_box_positions, validate_types

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
    *,
    param_name: str = "<palette_config>",
) -> None:
    """Raise TypeError if config is not a SequentialConfig."""
    if not isinstance(config, SequentialConfig):
        raise TypeError(
            f"`{param_name}` must be a SequentialConfig, got {type(config).__name__}.",
        )


def ensure_diverging_config(
    config: PaletteConfig,
    *,
    param_name: str = "<palette_config>",
) -> None:
    """Raise TypeError if config is not a DivergingConfig."""
    if not isinstance(config, DivergingConfig):
        raise TypeError(
            f"`{param_name}` must be a DivergingConfig, got {type(config).__name__}.",
        )


def ensure_continuous_config(
    config: PaletteConfig,
    *,
    param_name: str = "<palette_config>",
) -> None:
    """Raise TypeError if config is not a continuous palette config (sequential or diverging)."""
    if not isinstance(config, (SequentialConfig, DivergingConfig)):
        raise TypeError(
            f"`{param_name}` must be a continuous palette config (SequentialConfig or DivergingConfig), got {type(config).__name__}.",
        )


def ensure_discrete_config(
    config: PaletteConfig,
    *,
    param_name: str = "<palette_config>",
) -> None:
    """Raise TypeError if config is not a DiscreteConfig."""
    if not isinstance(config, DiscreteConfig):
        raise TypeError(
            f"`{param_name}` must be a DiscreteConfig, got {type(config).__name__}.",
        )


def _ensure_value_range(
    *,
    config: PaletteConfig,
    value_range: tuple[float, float] | None,
) -> tuple[float, float]:
    """Require the range a continuous palette spans, having none of its own to fall back on."""
    if value_range is None:
        raise ValueError(f"a {type(config).__name__} spans a `value_range`, so one must be given.")
    return value_range


def make_palette(
    *,
    config: PaletteConfig,
    value_range: tuple[float, float] | None = None,
) -> color_palettes.ColorPalette:
    """
    Construct a ColorPalette from a PaletteConfig and a data-driven value range.
    For full control over palette construction, use the palette classes directly.

    A continuous palette spans `value_range`, so it must be given one. A discrete palette
    is bounded by its own `bin_edges` instead, so passing it a range is a contradiction
    rather than something to quietly ignore.
    """
    match config:
        case SequentialConfig():
            return color_palettes.SequentialPalette.from_name(
                palette_name=config.palette_name,
                palette_range=config.palette_range,
                value_range=_ensure_value_range(
                    config=config,
                    value_range=value_range,
                ),
            )
        case DivergingConfig():
            return color_palettes.DivergingPalette.from_name(
                palette_name=config.palette_name,
                palette_range=config.palette_range,
                value_range=_ensure_value_range(
                    config=config,
                    value_range=value_range,
                ),
                mid_value=config.mid_value,
            )
        case DiscreteConfig():
            if value_range is not None:
                raise ValueError(
                    "`value_range` cannot apply to a DiscreteConfig; its `bin_edges`"
                    " already bound the palette.",
                )
            return color_palettes.DiscretePalette.from_name(
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


def _get_figure_shape_pt(
    *,
    panel: manage_figure.Panel,
) -> tuple[float, float]:
    """The root figure's shape in pt, since pt is measured against the page it is drawn at."""
    figure = panel.get_figure(root=True)
    if figure is None:
        raise ValueError("`panel` does not belong to a figure, so it has no size to measure against.")
    figure_width_inches, figure_height_inches = figure.get_size_inches()
    return (
        float(figure_width_inches) * style_figure.PT_PER_INCH,
        float(figure_height_inches) * style_figure.PT_PER_INCH,
    )


def _compute_colorbar_thickness(
    *,
    panel: manage_figure.Panel,
    colorbar_side: _Side,
    colorbar_length: float,
    colorbar_aspect_ratio: float | None,
    figure_params: style_figure.FigureParams,
) -> float:
    """
    Convert the bar's length-over-thickness into the share of the panel Matplotlib
    measures a neighbouring panel's thickness by.

    Length runs along the bar and thickness across it, so which panel dimension each is
    taken from swaps with the side the bar sits on.
    """
    if colorbar_aspect_ratio is None:
        colorbar_aspect_ratio = figure_params.colorbar_layout.aspect_ratio
    validate_types.ensure_finite_float(
        param=colorbar_aspect_ratio,
        param_name="colorbar_aspect_ratio",
        allow_none=False,
        require_positive=True,
        allow_zero=False,
    )
    figure_width_pt, figure_height_pt = _get_figure_shape_pt(panel=panel)
    panel_box = panel.get_position()
    panel_width_pt = panel_box.width * figure_width_pt
    panel_height_pt = panel_box.height * figure_height_pt
    if colorbar_side in (_Side.Left, _Side.Right):
        length_pt, across_panel_pt = (colorbar_length * panel_height_pt), panel_width_pt
    else:
        length_pt, across_panel_pt = (colorbar_length * panel_width_pt), panel_height_pt
    return (length_pt / colorbar_aspect_ratio) / across_panel_pt


def _compute_colorbar_gap(
    *,
    panel: manage_figure.Panel,
    colorbar_side: _Side,
    colorbar_gap: float | None,
    figure_params: style_figure.FigureParams,
) -> float:
    """
    Convert the gap between a panel and its colorbar (pt) into the share of the figure
    Matplotlib places panels in.

    A colorbar is placed as a panel neighbouring its own, so an unset gap is the one the
    figure already spaces its panels by: the column gap beside a panel, the row gap above
    or below one.
    """
    is_beside_panel = colorbar_side in (_Side.Left, _Side.Right)
    if colorbar_gap is None:
        panel_gaps = figure_params.colorbar_layout.gap
        if panel_gaps is None:
            panel_gaps = figure_params.figure_layout.panel_gaps
        colorbar_gap = panel_gaps.column if is_beside_panel else panel_gaps.row
    validate_types.ensure_finite_float(
        param=colorbar_gap,
        param_name="colorbar_gap",
        allow_none=False,
        require_positive=True,
        allow_zero=True,
    )
    figure_width_pt, figure_height_pt = _get_figure_shape_pt(panel=panel)
    return colorbar_gap / (figure_width_pt if is_beside_panel else figure_height_pt)


def _label_colorbar(
    *,
    colorbar: mpl_colorbar.Colorbar,
    label: str | None,
    colorbar_side: _Side,
    text_size: int | float,
    label_gap: float,
) -> None:
    if colorbar_side in (_Side.Left, _Side.Right):
        axis = colorbar.ax.yaxis
        if label:
            colorbar.set_label(
                label=label,
                fontsize=text_size,
                labelpad=label_gap,
                rotation=90,
            )
            axis.set_label_position(colorbar_side)  # pyright: ignore[reportArgumentType]
        axis.set_ticks_position(colorbar_side)  # pyright: ignore[reportArgumentType]
        axis.label.set_verticalalignment("center")
    elif colorbar_side in (_Side.Top, _Side.Bottom):
        axis = colorbar.ax.xaxis
        if label:
            colorbar.set_label(
                label=label,
                fontsize=text_size,
                labelpad=label_gap,
            )
            axis.set_label_position(colorbar_side)  # pyright: ignore[reportArgumentType]
        axis.set_ticks_position(colorbar_side)  # pyright: ignore[reportArgumentType]
    else:
        raise ValueError(f"unexpected colorbar_side: {colorbar_side!r}.")  # pyright: ignore[reportUnreachable]


##
## === ADD COLORBAR
##


def add_colorbar(
    *,
    panel: manage_figure.Panel,
    palette: color_palettes.ColorPalette,
    label: str | None = None,
    colorbar_side: box_positions.Positions.PositionLike = box_positions.Positions.Side.Right,
    colorbar_aspect_ratio: float | None = None,
    colorbar_length: float = 1.0,
    colorbar_gap: float | None = None,
    label_gap: float | None = None,
    text_size: int | float | None = None,
    figure_params: style_figure.FigureParams | None = None,
) -> mpl_colorbar.Colorbar:
    """
    `colorbar_length` is a share of the panel, and may exceed 1 for a bar spanning several;
    `colorbar_aspect_ratio` then sets how thick it is drawn, as its length over its
    thickness. `colorbar_gap` is in pt like the gaps between panels. All three default to
    what the active style asks for, the gap being the one the figure already spaces its
    panels by. `text_size` defaults to the active axis-label text size, and `label_gap` to
    the gap the active style leaves between a panel's tick labels and its axis label.
    """
    if figure_params is None:
        figure_params = style_figure.get_figure_params()
    if text_size is None:
        text_size = figure_params.text_size_params.axis_label_size
    if label_gap is None:
        label_gap = figure_params.panel_frame_params.axis_label_gap
    ## validate numeric params
    validate_types.ensure_finite_float(
        param=label_gap,
        param_name="label_gap",
        allow_none=False,
        require_positive=True,
        allow_zero=True,
    )
    validate_types.ensure_finite_scalar(
        param=text_size,
        param_name="text_size",
        allow_none=False,
        require_positive=True,
        allow_zero=False,
    )
    colorbar_side = validate_box_positions.as_box_side(side=colorbar_side)
    colorbar_orientation = _SIDE_TO_ORIENTATION[colorbar_side]
    panel_bounds = manage_figure.compute_neighbouring_panel_bounds(
        panel=panel,
        side=colorbar_side,
        thickness=_compute_colorbar_thickness(
            panel=panel,
            colorbar_side=colorbar_side,
            colorbar_length=colorbar_length,
            colorbar_aspect_ratio=colorbar_aspect_ratio,
            figure_params=figure_params,
        ),
        length=colorbar_length,
        gap=_compute_colorbar_gap(
            panel=panel,
            colorbar_side=colorbar_side,
            colorbar_gap=colorbar_gap,
            figure_params=figure_params,
        ),
    )
    colorbar_panel = panel.figure.add_axes(
        (
            panel_bounds.x_min,
            panel_bounds.y_min,
            panel_bounds.x_width,
            panel_bounds.y_width,
        ),
    )
    colorbar_mappable = mpl_cm.ScalarMappable(
        norm=palette.mpl_norm,
        cmap=palette.mpl_cmap,
    )
    ## required by mpl to suppress warning when ScalarMappable has no data
    colorbar_mappable.set_array([])
    colorbar = panel.figure.colorbar(
        mappable=colorbar_mappable,
        cax=colorbar_panel,
        orientation=colorbar_orientation,
    )
    _label_colorbar(
        colorbar=colorbar,
        label=label,
        colorbar_side=colorbar_side,
        text_size=text_size,
        label_gap=label_gap,
    )
    return colorbar


## } MODULE
