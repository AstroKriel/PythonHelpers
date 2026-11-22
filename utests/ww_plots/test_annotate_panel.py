## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
from matplotlib import legend as mpl_legend
from matplotlib import pyplot as mpl_pyplot

## local
from jormi.ww_plots import annotate_panel, manage_figure

##
## === TESTS
##


class TestAddCustomLegend(unittest.TestCase):

    def tearDown(
        self,
    ):
        mpl_pyplot.close("all")

    def test_returns_the_legend_it_adds_to_the_panel(
        self,
    ):
        _, panel_grid = manage_figure.create_figure_grid(
            num_panel_rows=1,
            num_panel_cols=1,
            panel_aspect_ratio=1.5,
        )
        panel = panel_grid[0, 0]
        legend = annotate_panel.add_custom_legend(
            panel=panel,
            artists=["o", None],
            labels=["marker", "text alone"],
            colors=["black", "red"],
        )
        self.assertIsInstance(legend, mpl_legend.Legend)
        self.assertIn(legend, panel.artists)
        self.assertEqual(
            [_text.get_text() for _text in legend.get_texts()],
            ["marker", "text alone"],
        )


## } U-TEST
