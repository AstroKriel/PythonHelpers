## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from jormi.ww_arrays.farrays_1d import compute_spectra

##
## === TEST SUITES
##


class TestSarrayPowerSpectrum(unittest.TestCase):

    def test_peak_at_pure_mode(
        self,
    ) -> None:
        num_cells = 16
        resolution_1d = (num_cells,)
        cell_indices = numpy.arange(num_cells)
        sarray_1d = numpy.cos(2.0 * numpy.pi * 2 * cell_indices / num_cells)
        spectrum = compute_spectra.compute_isotropic_power_spectrum_sarray(
            sarray_1d=sarray_1d,
            resolution_1d=resolution_1d,
        )
        peak_index = int(numpy.argmax(spectrum.power_spectrum_1d))
        self.assertEqual(spectrum.k_bin_centers_1d[peak_index], 2)

    def test_folds_positive_and_negative_frequency_together(
        self,
    ) -> None:
        ## a real-valued signal has mirror-symmetric +k/-k content; the 1D "shell" at each
        ## |k| is exactly that pair, so a single pure mode must land in exactly one bin, not
        ## be split or double-counted across two
        num_cells = 16
        resolution_1d = (num_cells,)
        cell_indices = numpy.arange(num_cells)
        sarray_1d = numpy.cos(2.0 * numpy.pi * 3 * cell_indices / num_cells)
        spectrum = compute_spectra.compute_isotropic_power_spectrum_sarray(
            sarray_1d=sarray_1d,
            resolution_1d=resolution_1d,
        )
        nonzero_bins = numpy.flatnonzero(spectrum.power_spectrum_1d > 1e-10)
        self.assertEqual(nonzero_bins.size, 1)


class TestVarrayPowerSpectrum(unittest.TestCase):

    def test_reduces_to_scalar_spectrum_for_single_component(
        self,
    ) -> None:
        num_cells = 16
        resolution_1d = (num_cells,)
        rng = numpy.random.default_rng(0)
        sarray_1d = rng.standard_normal(resolution_1d)
        varray_1d = sarray_1d[numpy.newaxis, :]
        vector_spectrum = compute_spectra.compute_isotropic_power_spectrum_varray(
            varray_1d=varray_1d,
            resolution_1d=resolution_1d,
        )
        scalar_spectrum = compute_spectra.compute_isotropic_power_spectrum_sarray(
            sarray_1d=sarray_1d,
            resolution_1d=resolution_1d,
        )
        numpy.testing.assert_allclose(
            vector_spectrum.power_spectrum_1d,
            scalar_spectrum.power_spectrum_1d,
        )

    def test_rejects_wrong_leading_axis_length(
        self,
    ) -> None:
        num_cells = 8
        resolution_1d = (num_cells,)
        rng = numpy.random.default_rng(1)
        with self.assertRaises(ValueError):
            compute_spectra.compute_isotropic_power_spectrum_varray(
                varray_1d=rng.standard_normal((2, num_cells)),
                resolution_1d=resolution_1d,
            )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
