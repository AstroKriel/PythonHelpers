## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from jormi.ww_arrays.farrays_2d import compute_spectra

##
## === TEST SUITES
##


class TestSarrayPowerSpectrum(unittest.TestCase):

    def test_peak_at_pure_mode(
        self,
    ) -> None:
        num_cells = 16
        resolution_2d = (num_cells, num_cells)
        cell_indices = numpy.arange(num_cells)
        sarray_2d = numpy.broadcast_to(
            numpy.cos(2.0 * numpy.pi * 2 * cell_indices / num_cells)[:, numpy.newaxis],
            resolution_2d,
        ).copy()
        spectrum = compute_spectra.compute_isotropic_power_spectrum_sarray(
            sarray_2d=sarray_2d,
            resolution_2d=resolution_2d,
        )
        peak_index = int(numpy.argmax(spectrum.power_spectrum_1d))
        self.assertEqual(spectrum.k_bin_centers_1d[peak_index], 2)


class TestVarrayPowerSpectrum(unittest.TestCase):
    """
    Verify the vector spectrum is built by FFT-ing each component and summing
    |.|^2 in k-space, not by FFT-ing the real-space magnitude.
    """

    def test_reduces_to_scalar_spectrum_for_single_nonzero_component(
        self,
    ) -> None:
        for num_cells in (8, 16):
            resolution_2d = (num_cells, num_cells)
            rng = numpy.random.default_rng(0)
            comp_x = rng.standard_normal(resolution_2d)
            varray_2d = numpy.zeros((2, *resolution_2d))
            varray_2d[0] = comp_x
            vector_spectrum = compute_spectra.compute_isotropic_power_spectrum_varray(
                varray_2d=varray_2d,
                resolution_2d=resolution_2d,
            )
            scalar_spectrum = compute_spectra.compute_isotropic_power_spectrum_sarray(
                sarray_2d=comp_x,
                resolution_2d=resolution_2d,
            )
            numpy.testing.assert_allclose(
                vector_spectrum.power_spectrum_1d,
                scalar_spectrum.power_spectrum_1d,
                err_msg=f"Vector spectrum with one nonzero component should match the scalar spectrum for N={num_cells}",
            )

    def test_parseval_total_power_matches_real_space_energy(
        self,
    ) -> None:
        for num_cells in (8, 16):
            resolution_2d = (num_cells, num_cells)
            rng = numpy.random.default_rng(1)
            varray_2d = rng.standard_normal((2, *resolution_2d))
            power_spectrum_2d = compute_spectra.compute_power_spectrum_farray(
                farray_2d=varray_2d,
                resolution_2d=resolution_2d,
            )
            total_power_k_space = numpy.sum(power_spectrum_2d)
            total_power_real_space = numpy.sum(numpy.square(varray_2d)) / (num_cells**2)
            numpy.testing.assert_allclose(
                total_power_k_space,
                total_power_real_space,
                rtol=1e-10,
                err_msg=f"Parseval mismatch for N={num_cells}",
            )

    def test_rejects_wrong_leading_axis_length(
        self,
    ) -> None:
        num_cells = 8
        resolution_2d = (num_cells, num_cells)
        rng = numpy.random.default_rng(2)
        with self.assertRaises(ValueError):
            compute_spectra.compute_isotropic_power_spectrum_varray(
                varray_2d=rng.standard_normal((3, *resolution_2d)),
                resolution_2d=resolution_2d,
            )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
