## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import pathlib
import tempfile
import unittest

## third-party
from matplotlib import colorbar as mpl_colorbar
from matplotlib import pyplot as mpl_pyplot

## local
from jormi.ww_plots import add_color, manage_figure
from jormi.ww_types import box_positions

##
## === HELPER FUNCTIONS
##


def make_panel_with_inset_colorbar(
    *,
    colorbar_corner: box_positions.Positions.PositionLike,
    corner_gaps_pt: tuple[float, float] | None = None,
) -> tuple[manage_figure.Panel, mpl_colorbar.Colorbar]:
    _, panel_grid = manage_figure.create_figure_grid(
        num_panel_rows=1,
        num_panel_cols=1,
        panel_aspect_ratio=1.5,
    )
    panel = panel_grid[0, 0]
    palette = add_color.make_palette(
        config=add_color.SequentialPaletteConfig(),
        value_range=(0.0, 1.0),
    )
    colorbar = add_color.add_inset_colorbar(
        panel=panel,
        palette=palette,
        label="value",
        colorbar_corner=colorbar_corner,
        corner_gaps_pt=corner_gaps_pt,
    )
    return panel, colorbar


def get_bar_bounds_in_panel(
    *,
    panel: manage_figure.Panel,
    colorbar: mpl_colorbar.Colorbar,
) -> tuple[float, float, float, float]:
    """Return the drawn bar's (x_min, y_min, x_max, y_max) as fractions of the drawn panel."""
    renderer = panel.figure.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
    panel_box = panel.get_window_extent(renderer)
    bar_box = colorbar.ax.get_window_extent(renderer)
    return (
        (bar_box.x0 - panel_box.x0) / panel_box.width,
        (bar_box.y0 - panel_box.y0) / panel_box.height,
        (bar_box.x1 - panel_box.x0) / panel_box.width,
        (bar_box.y1 - panel_box.y0) / panel_box.height,
    )


##
## === TEST SUITE
##


class TestResolveContinuousConfig(unittest.TestCase):

    def test_no_pivot_value_gives_sequential_config(
        self,
    ):
        config = add_color.resolve_continuous_palette_config(
            pivot_value=None,
            value_range=(0.0, 1.0),
        )
        self.assertIsInstance(config, add_color.SequentialPaletteConfig)

    def test_pivot_value_gives_diverging_config_centred_there(
        self,
    ):
        config = add_color.resolve_continuous_palette_config(
            pivot_value=1.0,
            value_range=(0.0, 2.0),
        )
        self.assertIsInstance(config, add_color.DivergingPaletteConfig)
        assert isinstance(config, add_color.DivergingPaletteConfig)
        self.assertEqual(config.mid_value, 1.0)

    def test_zero_pivot_value_still_gives_diverging_config(
        self,
    ):
        ## `0.0` is falsy but must not be treated the same as `None`
        config = add_color.resolve_continuous_palette_config(
            pivot_value=0.0,
            value_range=(-1.0, 1.0),
        )
        self.assertIsInstance(config, add_color.DivergingPaletteConfig)
        assert isinstance(config, add_color.DivergingPaletteConfig)
        self.assertEqual(config.mid_value, 0.0)

    def test_one_sided_value_range_falls_back_to_sequential(
        self,
    ):
        ## a signed quantity can still have an instance that comes out one-sided; a diverging
        ## palette cannot render a range that misses its own pivot
        config = add_color.resolve_continuous_palette_config(
            pivot_value=0.0,
            value_range=(-6.4e-4, -2.96e-9),
        )
        self.assertIsInstance(config, add_color.SequentialPaletteConfig)

    def test_flat_value_range_falls_back_to_sequential(
        self,
    ):
        ## a uniform initial condition (eg. density = 1.0 everywhere) gives a zero-width
        ## range; nothing can straddle it, so this must fall back rather than raise
        config = add_color.resolve_continuous_palette_config(
            pivot_value=0.0,
            value_range=(1.0, 1.0),
        )
        self.assertIsInstance(config, add_color.SequentialPaletteConfig)

    def test_custom_palette_names_are_used(
        self,
    ):
        config = add_color.resolve_continuous_palette_config(
            pivot_value=None,
            value_range=(0.0, 1.0),
            sequential_palette_name="custom-sequential",
        )
        self.assertIsInstance(config, add_color.SequentialPaletteConfig)
        assert isinstance(config, add_color.SequentialPaletteConfig)
        self.assertEqual(config.palette_name, "custom-sequential")


class TestAddInsetColorbar(unittest.TestCase):

    def tearDown(
        self,
    ) -> None:
        mpl_pyplot.close("all")

    def test_top_left_bar_sits_in_top_left_of_panel(
        self,
    ):
        panel, colorbar = make_panel_with_inset_colorbar(
            colorbar_corner=box_positions.Positions.Corner.TopLeft
        )
        x_min, y_min, x_max, y_max = get_bar_bounds_in_panel(
            panel=panel,
            colorbar=colorbar,
        )
        self.assertGreater(x_min, 0.0)
        self.assertLess(x_max, 0.5)
        self.assertLess(y_max, 1.0)
        self.assertGreater(y_min, 0.5)

    def test_bottom_right_bar_sits_in_bottom_right_of_panel(
        self,
    ):
        panel, colorbar = make_panel_with_inset_colorbar(
            colorbar_corner=box_positions.Positions.Corner.BottomRight
        )
        x_min, y_min, x_max, y_max = get_bar_bounds_in_panel(
            panel=panel,
            colorbar=colorbar,
        )
        self.assertGreater(x_min, 0.5)
        self.assertLess(x_max, 1.0)
        self.assertGreater(y_min, 0.0)
        self.assertLess(y_max, 0.5)

    def test_ticks_and_label_face_into_the_panel(
        self,
    ):
        _, top_colorbar = make_panel_with_inset_colorbar(
            colorbar_corner=box_positions.Positions.Corner.TopLeft
        )
        _, bottom_colorbar = make_panel_with_inset_colorbar(
            colorbar_corner=box_positions.Positions.Corner.BottomLeft
        )
        self.assertEqual(top_colorbar.ax.xaxis.get_label_position(), "bottom")
        self.assertEqual(top_colorbar.ax.xaxis.get_ticks_position(), "bottom")
        self.assertEqual(bottom_colorbar.ax.xaxis.get_label_position(), "top")
        self.assertEqual(bottom_colorbar.ax.xaxis.get_ticks_position(), "top")

    def test_bar_stays_inside_panel_after_figure_is_fitted_and_saved(
        self,
    ):
        panel, colorbar = make_panel_with_inset_colorbar(
            colorbar_corner=box_positions.Positions.Corner.TopRight
        )
        figure = manage_figure.get_figure(panels=[panel])
        with tempfile.TemporaryDirectory() as temp_dir:
            manage_figure.save_figure(
                figure=figure,
                figure_path=pathlib.Path(temp_dir) / "inset-colorbar.png",
                verbose=False,
            )
        renderer = figure.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
        panel_box = panel.get_window_extent(renderer)
        bar_box = colorbar.ax.get_window_extent(renderer)
        self.assertGreaterEqual(bar_box.x0, panel_box.x0)
        self.assertLessEqual(bar_box.x1, panel_box.x1)
        self.assertGreaterEqual(bar_box.y0, panel_box.y0)
        self.assertLessEqual(bar_box.y1, panel_box.y1)

    def test_horizontal_gap_moves_bar_sideways_only(
        self,
    ):
        narrow_panel, narrow_colorbar = make_panel_with_inset_colorbar(
            colorbar_corner=box_positions.Positions.Corner.TopLeft,
            corner_gaps_pt=(5.0, 5.0),
        )
        wide_panel, wide_colorbar = make_panel_with_inset_colorbar(
            colorbar_corner=box_positions.Positions.Corner.TopLeft,
            corner_gaps_pt=(30.0, 5.0),
        )
        narrow_bounds = get_bar_bounds_in_panel(
            panel=narrow_panel,
            colorbar=narrow_colorbar,
        )
        wide_bounds = get_bar_bounds_in_panel(
            panel=wide_panel,
            colorbar=wide_colorbar,
        )
        self.assertGreater(wide_bounds[0], narrow_bounds[0])
        self.assertAlmostEqual(wide_bounds[1], narrow_bounds[1])
        self.assertAlmostEqual(wide_bounds[3], narrow_bounds[3])

    def test_rejects_a_side_instead_of_a_corner(
        self,
    ):
        with self.assertRaises(ValueError):
            make_panel_with_inset_colorbar(colorbar_corner=box_positions.Positions.Side.Top)

    def test_rejects_length_fraction_above_one(
        self,
    ):
        _, panel_grid = manage_figure.create_figure_grid(
            num_panel_rows=1,
            num_panel_cols=1,
            panel_aspect_ratio=1.5,
        )
        palette = add_color.make_palette(
            config=add_color.SequentialPaletteConfig(),
            value_range=(0.0, 1.0),
        )
        with self.assertRaises(ValueError):
            add_color.add_inset_colorbar(
                panel=panel_grid[0, 0],
                palette=palette,
                colorbar_length_fraction=1.2,
            )


## } U-TEST
