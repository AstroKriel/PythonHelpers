## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from jormi.ww_arrays.farrays_3d.compute_spectra import IsotropicPowerSpectrum

_N = 8


##
## === TEST SUITES
##


class TestIsotropicPowerSpectrum(unittest.TestCase):

    def test_accepts_matching_lengths(
        self,
    ) -> None:
        IsotropicPowerSpectrum(
            k_bin_centers_1d=numpy.arange(_N, dtype=float),
            spectrum_1d=numpy.zeros(_N),
        )

    def test_rejects_mismatched_lengths(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            IsotropicPowerSpectrum(
                k_bin_centers_1d=numpy.arange(_N, dtype=float),
                spectrum_1d=numpy.zeros(_N + 1),
            )

    def test_rejects_2d_k_bin_centers(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            IsotropicPowerSpectrum(
                k_bin_centers_1d=numpy.zeros((_N, _N)),
                spectrum_1d=numpy.zeros(_N),
            )

    def test_rejects_2d_spectrum(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            IsotropicPowerSpectrum(
                k_bin_centers_1d=numpy.arange(_N, dtype=float),
                spectrum_1d=numpy.zeros((_N, _N)),
            )

    def test_lengths_match_after_construction(
        self,
    ) -> None:
        k = numpy.arange(_N, dtype=float)
        s = numpy.zeros(_N)
        spectrum = IsotropicPowerSpectrum(k_bin_centers_1d=k, spectrum_1d=s)
        self.assertEqual(
            spectrum.k_bin_centers_1d.shape[0],
            spectrum.spectrum_1d.shape[0],
        )


## } U-TEST
