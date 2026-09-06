## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## local
from jormi.ww_plots import add_color

##
## === TEST SUITE
##


class TestResolveContinuousConfig(unittest.TestCase):

    def test_no_pivot_value_gives_sequential_config(
        self,
    ):
        config = add_color.resolve_continuous_config(
            pivot_value=None,
            value_range=(0.0, 1.0),
        )
        self.assertIsInstance(config, add_color.SequentialConfig)

    def test_pivot_value_gives_diverging_config_centred_there(
        self,
    ):
        config = add_color.resolve_continuous_config(
            pivot_value=1.0,
            value_range=(0.0, 2.0),
        )
        self.assertIsInstance(config, add_color.DivergingConfig)
        assert isinstance(config, add_color.DivergingConfig)
        self.assertEqual(config.mid_value, 1.0)

    def test_zero_pivot_value_still_gives_diverging_config(
        self,
    ):
        ## `0.0` is falsy but must not be treated the same as `None`
        config = add_color.resolve_continuous_config(
            pivot_value=0.0,
            value_range=(-1.0, 1.0),
        )
        self.assertIsInstance(config, add_color.DivergingConfig)
        assert isinstance(config, add_color.DivergingConfig)
        self.assertEqual(config.mid_value, 0.0)

    def test_one_sided_value_range_falls_back_to_sequential(
        self,
    ):
        ## a signed quantity can still have an instance that comes out one-sided; a diverging
        ## palette cannot render a range that misses its own pivot
        config = add_color.resolve_continuous_config(
            pivot_value=0.0,
            value_range=(-6.4e-4, -2.96e-9),
        )
        self.assertIsInstance(config, add_color.SequentialConfig)

    def test_custom_palette_names_are_used(
        self,
    ):
        config = add_color.resolve_continuous_config(
            pivot_value=None,
            value_range=(0.0, 1.0),
            sequential_palette_name="custom-sequential",
        )
        self.assertIsInstance(config, add_color.SequentialConfig)
        assert isinstance(config, add_color.SequentialConfig)
        self.assertEqual(config.palette_name, "custom-sequential")


## } U-TEST
