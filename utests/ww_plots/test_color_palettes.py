## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## local
from jormi.ww_plots import add_color, color_palettes

##
## === TEST SUITE
##


class TestGetColor(unittest.TestCase):

    def _make_palette(
        self,
    ) -> color_palettes.ColorPalette:
        return add_color.make_palette(
            config=add_color.SequentialPaletteConfig(palette_name="cmr.lavender"),
            value_range=(0.0, 10.0),
        )

    def test_matches_composing_mpl_norm_and_mpl_cmap(
        self,
    ):
        palette = self._make_palette()
        self.assertEqual(
            palette.get_color(5.0),
            palette.mpl_cmap(palette.mpl_norm(5.0)),
        )

    def test_returns_rgba_tuple(
        self,
    ):
        palette = self._make_palette()
        color = palette.get_color(5.0)
        self.assertIsInstance(color, tuple)
        assert isinstance(color, tuple)
        self.assertEqual(len(color), 4)
        for channel in color:
            self.assertGreaterEqual(channel, 0.0)
            self.assertLessEqual(channel, 1.0)


## } U-TEST
