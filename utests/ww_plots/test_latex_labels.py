## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## local
from jormi.ww_plots import latex_labels

##
## === TEST SUITE
##


class Tests(unittest.TestCase):

    def test_get_label_wraps_content(
        self,
    ):
        label = latex_labels.LatexLabel(content=r"\rho")
        self.assertEqual(label.get_label(), r"$\rho$")

    def test_combining_labels(
        self,
    ):
        field_label = latex_labels.LatexLabel(content=r"\rho")
        combined = latex_labels.LatexLabel(content=rf"\mathrm{{rms}}\big({field_label.content}\big)")
        self.assertEqual(combined.get_label(), r"$\mathrm{rms}\big(\rho\big)$")

    def test_rejects_empty_content(
        self,
    ):
        with self.assertRaises(ValueError):
            latex_labels.LatexLabel(content="")

    def test_rejects_content_with_dollar_sign(
        self,
    ):
        with self.assertRaises(ValueError):
            latex_labels.LatexLabel(content=r"$\rho$")

    def test_rejects_unbalanced_braces(
        self,
    ):
        with self.assertRaises(ValueError):
            latex_labels.LatexLabel(content=r"\mathrm{rms")


## } U-TEST
