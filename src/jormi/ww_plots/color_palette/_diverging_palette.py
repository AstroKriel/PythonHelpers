## { MODULE

##
## === DEPENDENCIES
##

import dataclasses
import matplotlib.colors as mpl_colors

from dataclasses import dataclass

from jormi.ww_types import type_checks
from jormi.ww_plots.color_palette import _base_palette

##
## === DIVERGING PALETTE
##


@dataclass(frozen=True, kw_only=True)
class DivergingPalette(_base_palette.ColorPalette):
    """
    A continuous, two-sided color palette anchored at a midpoint.

    Use `from_name` or `from_colors` to construct.
    """

    value_range: tuple[float, float]  # (vmin, vmax)
    mid_value: float  # midpoint; must satisfy vmin < mid_value < vmax
    palette_range: tuple[float, float] = (0.0, 1.0)  # portion of palette to use, in [0, 1]

    def __post_init__(
        self,
    ) -> None:
        type_checks.ensure_ordered_pair(
            param=self.value_range,
            param_name="value_range",
        )
        type_checks.ensure_finite_float(
            param=self.mid_value,
            param_name="mid_value",
        )
        value_min, value_max = float(self.value_range[0]), float(self.value_range[1])
        value_center = float(self.mid_value)
        if not (value_min < value_center < value_max):
            raise ValueError(
                f"`mid_value` must satisfy vmin < mid_value < vmax, got ({value_min}, {value_center}, {value_max}).",
            )
        _base_palette.validate_palette_range(self.palette_range)

    @classmethod
    def from_name(
        cls,
        *,
        value_range: tuple[float, float],
        mid_value: float,
        palette_name: str = "blue-red",
        palette_range: tuple[float, float] = (0.0, 1.0),
    ) -> "DivergingPalette":
        return cls(
            value_range=value_range,
            mid_value=mid_value,
            palette_range=palette_range,
            _base_colormap=_base_palette.resolve_palette(palette_name),
        )

    @classmethod
    def from_colors(
        cls,
        *,
        value_range: tuple[float, float],
        mid_value: float,
        colors: list[str],
        palette_range: tuple[float, float] = (0.0, 1.0),
    ) -> "DivergingPalette":
        base_colormap = mpl_colors.LinearSegmentedColormap.from_list(
            name="custom",
            colors=colors,
            N=256,  # LUT size: 256 matches 8-bit color depth and exceeds perceptual resolution
        )
        return cls(
            value_range=value_range,
            mid_value=mid_value,
            palette_range=palette_range,
            _base_colormap=base_colormap,
        )

    @property
    def _mpl_norm(
        self,
    ) -> mpl_colors.TwoSlopeNorm:
        type_checks.ensure_ordered_pair(
            param=self.value_range,
            param_name="value_range",
        )
        value_min, value_max = float(self.value_range[0]), float(self.value_range[1])
        value_center = float(self.mid_value)
        if not (value_min < value_center < value_max):
            raise ValueError(
                f"`mid_value` must satisfy vmin < mid_value < vmax, got ({value_min}, {value_center}, {value_max}).",
            )
        return mpl_colors.TwoSlopeNorm(
            vmin=value_min,
            vcenter=value_center,
            vmax=value_max,
        )

    @property
    def _mpl_colormap(
        self,
    ) -> mpl_colors.Colormap:
        return _base_palette.subset_palette(
            palette=self._base_colormap,
            palette_range=self.palette_range,
            name="diverging",
        )

    def with_value_range(
        self,
        value_range: tuple[float, float],
    ) -> "DivergingPalette":
        return dataclasses.replace(self, value_range=value_range)

    def with_mid_value(
        self,
        mid_value: float,
    ) -> "DivergingPalette":
        return dataclasses.replace(self, mid_value=mid_value)


## } MODULE
