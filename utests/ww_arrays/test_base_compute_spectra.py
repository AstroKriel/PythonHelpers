## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from jormi.ww_arrays import _compute_spectra
from jormi.ww_arrays._compute_spectra import IsotropicPowerSpectrum

_N = 8

## resolutions covering every supported spatial dimensionality
_RESOLUTIONS = ((8, 8, 8), (8, 8))

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


class TestGenericPowerSpectrumKernel(unittest.TestCase):
    """
    compute_power_spectrum_farray must serve any rank and any number of spatial
    dimensions through the same code path: zero leading axes for a scalar, one for a
    vector, two for a rank-2 tensor, etc. No per-rank or per-dimension reimplementation.
    """

    def test_scalar_vector_tensor_share_one_kernel(
        self,
    ) -> None:
        rng = numpy.random.default_rng(3)
        for resolution in _RESOLUTIONS:
            num_cells = resolution[0]
            num_spatial_dims = len(resolution)
            for leading_shape in [(), (3,), (3, 3), (2, 4)]:
                farray = rng.standard_normal((*leading_shape, *resolution))
                power_spectrum = _compute_spectra.compute_power_spectrum_farray(
                    farray=farray,
                    resolution=resolution,
                )
                self.assertEqual(
                    power_spectrum.shape,
                    resolution,
                    msg=(
                        f"Leading component axes {leading_shape} did not collapse to a"
                        f" pure spatial spectrum for resolution={resolution}"
                    ),
                )
                ## Parseval must hold regardless of how many leading component axes are summed over
                total_power_k_space = numpy.sum(power_spectrum)
                total_power_real_space = numpy.sum(numpy.square(farray)) / (num_cells**num_spatial_dims)
                numpy.testing.assert_allclose(
                    total_power_k_space,
                    total_power_real_space,
                    rtol=1e-10,
                    err_msg=f"Parseval mismatch for leading_shape={leading_shape}, resolution={resolution}",
                )

    def test_leading_rank_reduces_to_scalar_for_single_nonzero_component(
        self,
    ) -> None:
        rng = numpy.random.default_rng(4)
        for resolution in _RESOLUTIONS:
            comp_sarray = rng.standard_normal(resolution)
            farray = numpy.zeros((3, 3, *resolution))
            farray[0, 0] = comp_sarray
            centered_spectrum = _compute_spectra.compute_power_spectrum_farray(
                farray=farray,
                resolution=resolution,
            )
            scalar_spectrum = _compute_spectra.compute_isotropic_power_spectrum_farray(
                farray=comp_sarray,
                resolution=resolution,
            )
            rank2_spectrum = _compute_spectra._integrate_over_shells(
                power_spectrum=centered_spectrum,
                resolution=resolution,
            )
            numpy.testing.assert_allclose(
                rank2_spectrum.power_spectrum_1d,
                scalar_spectrum.power_spectrum_1d,
                err_msg=(
                    "Rank-2 leading shape with one nonzero component should match the"
                    f" scalar spectrum for resolution={resolution}"
                ),
            )

    def test_rejects_non_ndarray_farray(
        self,
    ) -> None:
        with self.assertRaises(TypeError):
            _compute_spectra.compute_power_spectrum_farray(
                farray=[[1.0, 2.0], [3.0, 4.0]],  # pyright: ignore[reportArgumentType]
                resolution=(8, 8, 8),
            )

    def test_rejects_non_isotropic_resolution(
        self,
    ) -> None:
        rng = numpy.random.default_rng(5)
        with self.assertRaises(ValueError):
            _compute_spectra.compute_power_spectrum_farray(
                farray=rng.standard_normal((8, 16)),
                resolution=(8, 16),
            )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
