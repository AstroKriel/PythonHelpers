## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import dataclasses

from dataclasses import dataclass

## third-party
import matplotlib.colors as mpl_colors

## local
## import directly from the module file (not via the package __init__) to avoid a static import cycle
from jormi.ww_plots.color_palettes._base_palette import (
    ColorPalette,
    validate_palette_range,
    resolve_palette,
    subset_palette,
)
from jormi.ww_validation import validate_python_types

##
## === DIVERGING PALETTE
##


@dataclass(frozen=True, kw_only=True)
class DivergingPalette(ColorPalette):
    """
    A continuous, two-sided color palette anchored at a midpoint.

    Use `from_name` or `from_colors` to construct.
    """

    value_range: tuple[float, float]  # (min_value, max_value)
    mid_value: float  # midpoint; must satisfy min_value < mid_value < max_value
    palette_range: tuple[float, float] = (0.0, 1.0)  # portion of palette to use, in [0, 1]

    def __post_init__(
        self,
    ) -> None:
        validate_python_types.validate_ordered_pair(
            param=self.value_range,
            param_name="value_range",
        )
        validate_python_types.validate_finite_float(
            param=self.mid_value,
            param_name="mid_value",
        )
        min_value, max_value = float(self.value_range[0]), float(self.value_range[1])
        mid_value = float(self.mid_value)
        if not (min_value < mid_value < max_value):
            raise ValueError(
                f"`mid_value` must satisfy min_value < mid_value < max_value, got ({min_value}, {mid_value}, {max_value}).",
            )
        validate_palette_range(self.palette_range)

    @classmethod
    def from_name(
        cls,
        *,
        value_range: tuple[float, float],
        mid_value: float,
        palette_name: str = "blue-white-red",
        palette_range: tuple[float, float] = (0.0, 1.0),
    ) -> "DivergingPalette":
        return cls(
            value_range=value_range,
            mid_value=mid_value,
            palette_range=palette_range,
            _base_cmap=resolve_palette(palette_name),
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
        ## look-up table size should be 256 to match 8-bit color depth (this exceeds perceptual resolution)
        cmap = mpl_colors.LinearSegmentedColormap.from_list(
            name="custom-diverging-cmap",
            colors=colors,
            N=256,
        )
        return cls(
            value_range=value_range,
            mid_value=mid_value,
            palette_range=palette_range,
            _base_cmap=cmap,
        )

    @property
    def mpl_norm(
        self,
    ) -> mpl_colors.TwoSlopeNorm:
        validate_python_types.validate_ordered_pair(
            param=self.value_range,
            param_name="value_range",
        )
        min_value, max_value = float(self.value_range[0]), float(self.value_range[1])
        mid_value = float(self.mid_value)
        if not (min_value < mid_value < max_value):
            raise ValueError(
                f"`mid_value` must satisfy min_value < mid_value < max_value, got ({min_value}, {mid_value}, {max_value}).",
            )
        return mpl_colors.TwoSlopeNorm(
            vmin=min_value,
            vcenter=mid_value,
            vmax=max_value,
        )

    @property
    def mpl_cmap(
        self,
    ) -> mpl_colors.Colormap:
        return subset_palette(
            palette_cmap=self._base_cmap,
            palette_range=self.palette_range,
            palette_label="subset-diverging-cmap",
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
