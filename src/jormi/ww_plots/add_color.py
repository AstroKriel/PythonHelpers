## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import dataclasses

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


@dataclasses.dataclass
class SequentialConfig:
    """Lightweight config for a sequential (single-direction) palette."""

    palette_name: str = "cmr.arctic"
    palette_range: tuple[float, float] = (0.0, 1.0)


@dataclasses.dataclass
class DivergingConfig:
    """Lightweight config for a diverging (two-sided) palette."""

    mid_value: float = 0.0
    palette_name: str = "blue-white-red"
    palette_range: tuple[float, float] = (0.0, 1.0)


@dataclasses.dataclass
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
    """Raise ValueError if `value_range` is not given; a continuous palette has no default."""
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


def _resolve_colorbar_gap_pt(
    *,
    colorbar_side: _Side,
    colorbar_gap_pt: float | None,
    figure_params: style_figure.FigureParams,
) -> float:
    """
    The gap between a panel and its colorbar, in pt.

    A colorbar is placed as a panel neighbouring its own, so an unset gap is the one the
    figure already spaces its panels by: the column gap beside a panel, the row gap above
    or below one.
    """
    if colorbar_gap_pt is None:
        panel_gaps = figure_params.colorbar_layout.gap
        if panel_gaps is None:
            panel_gaps = figure_params.figure_layout.panel_gaps
        is_beside_panel = colorbar_side in (_Side.Left, _Side.Right)
        colorbar_gap_pt = panel_gaps.col_pt if is_beside_panel else panel_gaps.row_pt
    validate_types.ensure_finite_float(
        param=colorbar_gap_pt,
        param_name="colorbar_gap_pt",
        allow_none=False,
        require_positive=True,
        allow_zero=True,
    )
    return colorbar_gap_pt


def _compute_colorbar_gap_fraction(
    *,
    panel: manage_figure.Panel,
    colorbar_side: _Side,
    colorbar_gap_pt: float | None,
    figure_params: style_figure.FigureParams,
) -> float:
    """
    Convert the gap between a panel and its colorbar (in pt) into the share of the figure
    Matplotlib places panels in.
    """
    is_beside_panel = colorbar_side in (_Side.Left, _Side.Right)
    resolved_gap_pt = _resolve_colorbar_gap_pt(
        colorbar_side=colorbar_side,
        colorbar_gap_pt=colorbar_gap_pt,
        figure_params=figure_params,
    )
    ## the root figure, since a gap in pt is measured against the page the figure is drawn at
    figure = manage_figure.get_figure(
        panels=[panel],
        param_name="panel",
    )
    figure_size = manage_figure.FigureSize(figure=figure)
    figure_length_pt = figure_size.width_pt if is_beside_panel else figure_size.height_pt
    return resolved_gap_pt / figure_length_pt


def _label_colorbar(
    *,
    colorbar: mpl_colorbar.Colorbar,
    label: str | None,
    colorbar_side: _Side,
    text_size_pt: int | float,
    label_gap_pt: float,
) -> None:
    if colorbar_side in (_Side.Left, _Side.Right):
        axis = colorbar.ax.yaxis
        if label:
            colorbar.set_label(
                label=label,
                fontsize=text_size_pt,
                labelpad=label_gap_pt,
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
                fontsize=text_size_pt,
                labelpad=label_gap_pt,
            )
            axis.set_label_position(colorbar_side)  # pyright: ignore[reportArgumentType]
        axis.set_ticks_position(colorbar_side)  # pyright: ignore[reportArgumentType]
    else:
        raise ValueError(
            f"unexpected colorbar_side: {colorbar_side!r}.",
        )  # pyright: ignore[reportUnreachable]


##
## === ADD COLORBAR
##


def add_colorbar(
    *,
    panels: manage_figure.Panel | manage_figure.PanelGrid,
    palette: color_palettes.ColorPalette,
    label: str | None = None,
    colorbar_side: box_positions.Positions.PositionLike = box_positions.Positions.Side.Right,
    colorbar_length_fraction: float = 1.0,
    colorbar_aspect_ratio: float | None = None,
    colorbar_gap_pt: float | None = None,
    label_gap_pt: float | None = None,
    text_size_pt: int | float | None = None,
    figure_params: style_figure.FigureParams | None = None,
) -> mpl_colorbar.Colorbar:
    """
    Add a colorbar next to `panels`.

    `colorbar_length_fraction` is a share of what the bar describes; `colorbar_aspect_ratio`
    is its length over its thickness, so a bar keeps its proportions whatever it spans.
    `colorbar_gap_pt` defaults to the panel gap the figure already uses, `text_size_pt` to
    the active axis-label size, and `label_gap_pt` to the active tick-to-axis-label gap.
    """
    if figure_params is None:
        figure_params = style_figure.get_figure_params()
    if text_size_pt is None:
        text_size_pt = figure_params.text_size_params.axis_label_size_pt
    if label_gap_pt is None:
        label_gap_pt = figure_params.frame_params.axis_label_gap_pt
    ## validate numeric params
    validate_types.ensure_finite_float(
        param=colorbar_length_fraction,
        param_name="colorbar_length_fraction",
        allow_none=False,
        require_positive=True,
        allow_zero=False,
    )
    validate_types.ensure_finite_float(
        param=label_gap_pt,
        param_name="label_gap_pt",
        allow_none=False,
        require_positive=True,
        allow_zero=True,
    )
    validate_types.ensure_finite_scalar(
        param=text_size_pt,
        param_name="text_size_pt",
        allow_none=False,
        require_positive=True,
        allow_zero=False,
    )
    colorbar_side = validate_box_positions.as_box_side(side=colorbar_side)
    colorbar_orientation = _SIDE_TO_ORIENTATION[colorbar_side]
    neighbouring_panels = manage_figure.as_panel_list(panels=panels)
    manage_figure.get_figure(panels=neighbouring_panels)
    panel = neighbouring_panels[0]
    if colorbar_aspect_ratio is None:
        colorbar_aspect_ratio = figure_params.colorbar_layout.aspect_ratio
    validate_types.ensure_finite_float(
        param=colorbar_aspect_ratio,
        param_name="colorbar_aspect_ratio",
        allow_none=False,
        require_positive=True,
        allow_zero=False,
    )
    panel_bounds = manage_figure.compute_panel_fractional_bounds(
        neighbouring_panels=neighbouring_panels,
        side=colorbar_side,
        gap_fraction=_compute_colorbar_gap_fraction(
            panel=panel,
            colorbar_side=colorbar_side,
            colorbar_gap_pt=colorbar_gap_pt,
            figure_params=figure_params,
        ),
        length_fraction=colorbar_length_fraction,
        thickness_fraction=manage_figure.compute_colorbar_thickness_fraction(
            neighbouring_panels=neighbouring_panels,
            side=colorbar_side,
            length_fraction=colorbar_length_fraction,
            aspect_ratio=colorbar_aspect_ratio,
        ),
    )
    colorbar_panel = panel.figure.add_axes(
        (
            panel_bounds.x_min_fraction,
            panel_bounds.y_min_fraction,
            panel_bounds.x_width_fraction,
            panel_bounds.y_width_fraction,
        ),
    )
    ## a bar is placed beyond its panel, so a figure fitted to its contents has to know the
    ## bar is there: to leave room for it, and to place it again once the panels have moved
    manage_figure.register_colorbar(
        colorbar_panel=colorbar_panel,
        neighbouring_panels=neighbouring_panels,
        side=colorbar_side,
        length_fraction=colorbar_length_fraction,
        aspect_ratio=colorbar_aspect_ratio,
        gap_pt=_resolve_colorbar_gap_pt(
            colorbar_side=colorbar_side,
            colorbar_gap_pt=colorbar_gap_pt,
            figure_params=figure_params,
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
        text_size_pt=text_size_pt,
        label_gap_pt=label_gap_pt,
    )
    return colorbar


## } MODULE
