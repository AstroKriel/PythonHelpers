## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from jormi.ww_arrays import _compute_spectra, _filter_spectra

## resolutions covering every supported spatial dimensionality
_RESOLUTIONS = ((8, 8, 8), (8, 8), (8,))


def _naive_bandpass_filter(
    *,
    sarray: numpy.ndarray,
    k_min: float,
    k_max: float,
) -> numpy.ndarray:
    """Reference implementation: unshifted fftfreq*shape convention, no shared helpers."""
    freq_axes = [numpy.fft.fftfreq(num_cells) * num_cells for num_cells in sarray.shape]
    k_grids = numpy.meshgrid(*freq_axes, indexing="ij")
    k_magnitude = numpy.sqrt(sum(k_grid**2 for k_grid in k_grids))
    k_mask = (k_magnitude >= k_min) & (k_magnitude <= k_max)
    fft_sarray = numpy.fft.fftn(sarray, norm="forward")
    return numpy.fft.ifftn(fft_sarray * k_mask, norm="forward").real


##
## === TEST SUITES
##


class TestGenericFilterKernel(unittest.TestCase):
    """
    compute_bandpass_filtered_farray must serve any rank and any number of spatial
    dimensions through the same code path: zero leading axes for a scalar, one for a
    vector, two for a rank-2 tensor, etc. No per-rank or per-dimension reimplementation.
    """

    def test_output_shape_matches_input_shape(
        self,
    ) -> None:
        rng = numpy.random.default_rng(0)
        for resolution in _RESOLUTIONS:
            for leading_shape in [(), (3,), (3, 3), (2, 4)]:
                farray = rng.standard_normal((*leading_shape, *resolution))
                filtered_farray = _filter_spectra.compute_bandpass_filtered_farray(
                    farray=farray,
                    resolution=resolution,
                    num_ranks=len(leading_shape),
                    k_min=1.0,
                    k_max=float(resolution[0]),
                )
                self.assertEqual(filtered_farray.shape, farray.shape)

    def test_components_are_filtered_independently(
        self,
    ) -> None:
        ## a batched (leading-shape) call must equal looping the scalar kernel over
        ## each leading component: no cross-component mixing in the mask.
        rng = numpy.random.default_rng(1)
        resolution = (8, 8, 8)
        varray = rng.standard_normal((3, *resolution))
        batched = _filter_spectra.compute_bandpass_filtered_farray(
            farray=varray,
            resolution=resolution,
            num_ranks=1,
            k_min=1.0,
            k_max=3.0,
        )
        for comp_index in range(3):
            per_component = _filter_spectra.compute_bandpass_filtered_farray(
                farray=varray[comp_index],
                resolution=resolution,
                num_ranks=0,
                k_min=1.0,
                k_max=3.0,
            )
            numpy.testing.assert_allclose(
                batched[comp_index],
                per_component,
                err_msg=f"Component {comp_index} diverged from its independent filter",
            )

    def test_matches_naive_unshifted_implementation(
        self,
    ) -> None:
        rng = numpy.random.default_rng(2)
        for resolution in _RESOLUTIONS:
            sarray = rng.standard_normal(resolution)
            filtered = _filter_spectra.compute_bandpass_filtered_farray(
                farray=sarray,
                resolution=resolution,
                num_ranks=0,
                k_min=1.0,
                k_max=2.0,
            )
            expected = _naive_bandpass_filter(
                sarray=sarray,
                k_min=1.0,
                k_max=2.0,
            )
            numpy.testing.assert_allclose(
                filtered,
                expected,
                atol=1e-10,
                err_msg=f"Diverged from the naive fftfreq convention for resolution={resolution}",
            )

    def test_full_range_reconstructs_original_field(
        self,
    ) -> None:
        rng = numpy.random.default_rng(3)
        resolution = (8, 8, 8)
        sarray = rng.standard_normal(resolution)
        max_possible_k = float(numpy.sqrt(3) * (resolution[0] // 2))
        filtered = _filter_spectra.compute_bandpass_filtered_farray(
            farray=sarray,
            resolution=resolution,
            num_ranks=0,
            k_min=0.0,
            k_max=max_possible_k,
        )
        numpy.testing.assert_allclose(filtered, sarray, atol=1e-10)

    def test_range_beyond_max_mode_gives_zero(
        self,
    ) -> None:
        rng = numpy.random.default_rng(4)
        resolution = (8, 8, 8)
        sarray = rng.standard_normal(resolution)
        filtered = _filter_spectra.compute_bandpass_filtered_farray(
            farray=sarray,
            resolution=resolution,
            num_ranks=0,
            k_min=100.0,
            k_max=200.0,
        )
        numpy.testing.assert_allclose(filtered, numpy.zeros(resolution), atol=1e-10)

    def test_rejects_non_isotropic_resolution(
        self,
    ) -> None:
        rng = numpy.random.default_rng(5)
        with self.assertRaises(ValueError):
            _filter_spectra.compute_bandpass_filtered_farray(
                farray=rng.standard_normal((8, 16)),
                resolution=(8, 16),
                num_ranks=0,
                k_min=1.0,
                k_max=2.0,
            )

    def test_rejects_num_ranks_mismatch(
        self,
    ) -> None:
        ## an array with the wrong number of leading axes must raise, not be silently
        ## reinterpreted as a different, valid rank
        rng = numpy.random.default_rng(6)
        for resolution in _RESOLUTIONS:
            farray = rng.standard_normal((3, 4, *resolution))
            with self.assertRaises(ValueError):
                _filter_spectra.compute_bandpass_filtered_farray(
                    farray=farray,
                    resolution=resolution,
                    num_ranks=1,
                    k_min=1.0,
                    k_max=2.0,
                )

    def test_rejects_k_min_greater_than_k_max(
        self,
    ) -> None:
        rng = numpy.random.default_rng(7)
        with self.assertRaises(ValueError):
            _filter_spectra.compute_bandpass_filtered_farray(
                farray=rng.standard_normal((8, 8, 8)),
                resolution=(8, 8, 8),
                num_ranks=0,
                k_min=5.0,
                k_max=1.0,
            )

    def test_rejects_negative_k_min(
        self,
    ) -> None:
        rng = numpy.random.default_rng(8)
        with self.assertRaises(ValueError):
            _filter_spectra.compute_bandpass_filtered_farray(
                farray=rng.standard_normal((8, 8, 8)),
                resolution=(8, 8, 8),
                num_ranks=0,
                k_min=-1.0,
                k_max=1.0,
            )

    def test_reuses_shared_radial_k_magnitude_helper(
        self,
    ) -> None:
        ## the filter must not build its own k-grid convention alongside the spectra
        ## tool's; both should be reading from the same cached helper.
        resolution = (8, 8, 8)
        before = _compute_spectra._compute_radial_k_magnitude.cache_info().hits
        _filter_spectra.compute_bandpass_filtered_farray(
            farray=numpy.zeros(resolution),
            resolution=resolution,
            num_ranks=0,
            k_min=1.0,
            k_max=2.0,
        )
        _compute_spectra.compute_isotropic_power_spectrum_farray(
            farray=numpy.zeros(resolution),
            resolution=resolution,
            num_ranks=0,
        )
        after = _compute_spectra._compute_radial_k_magnitude.cache_info().hits
        self.assertGreater(after, before)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
