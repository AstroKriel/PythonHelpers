## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from jormi.ww_arrays.farrays_3d import compute_spectra

##
## === TEST SUITES
##


class TestVarrayPowerSpectrum(unittest.TestCase):
    """
    Verify the vector spectrum is built by FFT-ing each component and summing
    |.|^2 in k-space, not by FFT-ing the real-space magnitude.
    """

    def test_reduces_to_scalar_spectrum_for_single_nonzero_component(
        self,
    ) -> None:
        for num_cells in (8, 16):
            resolution_3d = (num_cells, num_cells, num_cells)
            rng = numpy.random.default_rng(0)
            comp_x = rng.standard_normal(resolution_3d)
            varray_3d = numpy.zeros((3, *resolution_3d))
            varray_3d[0] = comp_x
            vector_spectrum = compute_spectra.compute_isotropic_power_spectrum_varray(
                varray_3d=varray_3d,
                resolution_3d=resolution_3d,
            )
            scalar_spectrum = compute_spectra.compute_isotropic_power_spectrum_sarray(
                sarray_3d=comp_x,
                resolution_3d=resolution_3d,
            )
            numpy.testing.assert_allclose(
                vector_spectrum.power_spectrum_1d,
                scalar_spectrum.power_spectrum_1d,
                err_msg=f"Vector spectrum with one nonzero component should match the scalar spectrum for N={num_cells}",
            )

    def test_parseval_total_power_matches_real_space_energy(
        self,
    ) -> None:
        ## sum_k sum_i |FFT(v_i)(k)|^2 must equal (1/N) sum_x |v(x)|^2 for norm="forward";
        ## this is the property that makes the spectrum an energy decomposition by scale,
        ## which FFT-ing the real-space magnitude first does not preserve per-k.
        for num_cells in (8, 16):
            resolution_3d = (num_cells, num_cells, num_cells)
            rng = numpy.random.default_rng(1)
            varray_3d = rng.standard_normal((3, *resolution_3d))
            power_spectrum_3d = compute_spectra.compute_power_spectrum_farray(
                farray_3d=varray_3d,
                resolution_3d=resolution_3d,
                num_ranks=1,
            )
            total_power_k_space = numpy.sum(power_spectrum_3d)
            total_power_real_space = numpy.sum(numpy.square(varray_3d)) / (num_cells**3)
            numpy.testing.assert_allclose(
                total_power_k_space,
                total_power_real_space,
                rtol=1e-10,
                err_msg=f"Parseval mismatch for N={num_cells}",
            )

    def test_differs_from_spectrum_of_magnitude(
        self,
    ) -> None:
        ## the correct vector spectrum (sum of per-component FFTs) must not equal the
        ## spectrum of the real-space magnitude field |v| for a generic vector field;
        ## the two coincide only in degenerate cases (e.g. a single nonzero component).
        num_cells = 16
        resolution_3d = (num_cells, num_cells, num_cells)
        rng = numpy.random.default_rng(2)
        varray_3d = rng.standard_normal((3, *resolution_3d))
        vector_spectrum = compute_spectra.compute_isotropic_power_spectrum_varray(
            varray_3d=varray_3d,
            resolution_3d=resolution_3d,
        )
        magnitude_sarray_3d = numpy.sqrt(numpy.sum(numpy.square(varray_3d), axis=0))
        magnitude_spectrum = compute_spectra.compute_isotropic_power_spectrum_sarray(
            sarray_3d=magnitude_sarray_3d,
            resolution_3d=resolution_3d,
        )
        self.assertFalse(
            numpy.allclose(
                vector_spectrum.power_spectrum_1d,
                magnitude_spectrum.power_spectrum_1d,
            ),
            msg="Vector spectrum should differ from the spectrum of |v| for a generic field",
        )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
