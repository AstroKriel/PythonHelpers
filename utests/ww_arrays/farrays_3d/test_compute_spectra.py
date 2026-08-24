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
            power_spectrum_1d=numpy.zeros(_N),
        )

    def test_rejects_mismatched_lengths(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            IsotropicPowerSpectrum(
                k_bin_centers_1d=numpy.arange(_N, dtype=float),
                power_spectrum_1d=numpy.zeros(_N + 1),
            )

    def test_rejects_2d_k_bin_centers(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            IsotropicPowerSpectrum(
                k_bin_centers_1d=numpy.zeros((_N, _N)),
                power_spectrum_1d=numpy.zeros(_N),
            )

    def test_rejects_2d_spectrum(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            IsotropicPowerSpectrum(
                k_bin_centers_1d=numpy.arange(_N, dtype=float),
                power_spectrum_1d=numpy.zeros((_N, _N)),
            )

    def test_lengths_match_after_construction(
        self,
    ) -> None:
        spectrum = IsotropicPowerSpectrum(
            k_bin_centers_1d=numpy.arange(_N, dtype=float),
            power_spectrum_1d=numpy.zeros(_N),
        )
        self.assertEqual(
            spectrum.k_bin_centers_1d.shape[0],
            spectrum.power_spectrum_1d.shape[0],
        )


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


class TestGenericPowerSpectrumKernel(unittest.TestCase):
    """
    compute_power_spectrum_farray must serve any rank through the same code path:
    zero leading axes for a scalar, one for a vector, two for a rank-2 tensor, etc.
    No per-rank reimplementation.
    """

    def test_scalar_vector_tensor_share_one_kernel(
        self,
    ) -> None:
        num_cells = 8
        resolution_3d = (num_cells, num_cells, num_cells)
        rng = numpy.random.default_rng(3)
        for leading_shape in [(), (3,), (3, 3), (2, 4)]:
            farray_3d = rng.standard_normal((*leading_shape, *resolution_3d))
            power_spectrum_3d = compute_spectra.compute_power_spectrum_farray(
                farray_3d=farray_3d,
                resolution_3d=resolution_3d,
            )
            self.assertEqual(
                power_spectrum_3d.shape,
                resolution_3d,
                msg=f"Leading component axes {leading_shape} did not collapse to a pure spatial spectrum",
            )
            ## Parseval must hold regardless of how many leading component axes are summed over
            total_power_k_space = numpy.sum(power_spectrum_3d)
            total_power_real_space = numpy.sum(numpy.square(farray_3d)) / (num_cells**3)
            numpy.testing.assert_allclose(
                total_power_k_space,
                total_power_real_space,
                rtol=1e-10,
                err_msg=f"Parseval mismatch for leading_shape={leading_shape}",
            )

    def test_rank2_tensor_reduces_to_scalar_for_single_nonzero_component(
        self,
    ) -> None:
        num_cells = 8
        resolution_3d = (num_cells, num_cells, num_cells)
        rng = numpy.random.default_rng(4)
        comp_00 = rng.standard_normal(resolution_3d)
        r2tarray_3d = numpy.zeros((3, 3, *resolution_3d))
        r2tarray_3d[0, 0] = comp_00
        tensor_centered_spectrum = compute_spectra.compute_power_spectrum_farray(
            farray_3d=r2tarray_3d,
            resolution_3d=resolution_3d,
        )
        scalar_spectrum = compute_spectra.compute_isotropic_power_spectrum_sarray(
            sarray_3d=comp_00,
            resolution_3d=resolution_3d,
        )
        tensor_spectrum = compute_spectra._integrate_over_spherical_shells(
            power_spectrum_3d=tensor_centered_spectrum,
            resolution_3d=resolution_3d,
        )
        numpy.testing.assert_allclose(
            tensor_spectrum.power_spectrum_1d,
            scalar_spectrum.power_spectrum_1d,
            err_msg="Rank-2 tensor spectrum with one nonzero component should match the scalar spectrum",
        )

    def test_rejects_non_ndarray_farray(
        self,
    ) -> None:
        with self.assertRaises(TypeError):
            compute_spectra.compute_power_spectrum_farray(
                farray_3d=[[1.0, 2.0], [3.0, 4.0]],  # pyright: ignore[reportArgumentType]
                resolution_3d=(8, 8, 8),
            )


## } U-TEST
