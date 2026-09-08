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

    def test_label_wraps_content(
        self,
    ):
        latex_label = latex_labels.LatexLabel(content=r"\rho")
        self.assertEqual(latex_label.label, r"$\rho$")

    def test_combining_labels(
        self,
    ):
        field_latex_label = latex_labels.LatexLabel(content=r"\rho")
        combined_latex_label = latex_labels.LatexLabel(
            content=rf"\mathrm{{rms}}\big({field_latex_label.content}\big)",
        )
        self.assertEqual(combined_latex_label.label, r"$\mathrm{rms}\big(\rho\big)$")

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
