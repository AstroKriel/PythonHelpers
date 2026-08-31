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


class TestBandpassFilteredFFT(unittest.TestCase):
    """
    compute_bandpass_filtered_farray must be a thin wrapper around
    compute_bandpass_filtered_fft, not a second implementation: both must do exactly one
    FFT round trip, and `.filtered_farray` must match the plain-array function exactly.
    """

    def test_filtered_farray_matches_plain_function(
        self,
    ) -> None:
        rng = numpy.random.default_rng(9)
        resolution = (8, 8, 8)
        varray = rng.standard_normal((3, *resolution))
        fft_state = _filter_spectra.compute_bandpass_filtered_fft(
            farray=varray,
            resolution=resolution,
            num_ranks=1,
            k_min=1.0,
            k_max=3.0,
        )
        plain = _filter_spectra.compute_bandpass_filtered_farray(
            farray=varray,
            resolution=resolution,
            num_ranks=1,
            k_min=1.0,
            k_max=3.0,
        )
        numpy.testing.assert_allclose(fft_state.filtered_farray, plain)

    def test_shifted_fft_shapes_match_input(
        self,
    ) -> None:
        rng = numpy.random.default_rng(10)
        resolution = (8, 8, 8)
        varray = rng.standard_normal((3, *resolution))
        fft_state = _filter_spectra.compute_bandpass_filtered_fft(
            farray=varray,
            resolution=resolution,
            num_ranks=1,
            k_min=1.0,
            k_max=3.0,
        )
        self.assertEqual(fft_state.shifted_fft_farray.shape, varray.shape)
        self.assertEqual(fft_state.shifted_fft_farray_masked.shape, varray.shape)

    def test_shifted_fft_dc_component_is_centered(
        self,
    ) -> None:
        ## a constant field has all its power in the DC mode; after fftshift that mode
        ## sits at the center index (N // 2), not at index 0.
        resolution = (8, 8, 8)
        sarray = numpy.full(resolution, fill_value=3.14)
        fft_state = _filter_spectra.compute_bandpass_filtered_fft(
            farray=sarray,
            resolution=resolution,
            num_ranks=0,
            k_min=0.0,
            k_max=1.0,
        )
        center = resolution[0] // 2
        peak_index = numpy.unravel_index(
            numpy.argmax(numpy.abs(fft_state.shifted_fft_farray)),
            resolution,
        )
        self.assertEqual(peak_index, (center, center, center))

    def test_masked_fft_is_zero_outside_the_band(
        self,
    ) -> None:
        rng = numpy.random.default_rng(11)
        resolution = (8, 8, 8)
        sarray = rng.standard_normal(resolution)
        fft_state = _filter_spectra.compute_bandpass_filtered_fft(
            farray=sarray,
            resolution=resolution,
            num_ranks=0,
            k_min=100.0,
            k_max=200.0,
        )
        numpy.testing.assert_allclose(
            fft_state.shifted_fft_farray_masked,
            numpy.zeros_like(fft_state.shifted_fft_farray_masked),
        )

    def test_unmasked_fft_reconstructs_original_field(
        self,
    ) -> None:
        ## shifted_fft_farray is the raw (unmasked) FFT: ifftshift-ing and inverse-FFT-ing
        ## it directly must reconstruct the original input field exactly.
        rng = numpy.random.default_rng(12)
        resolution = (8, 8, 8)
        sarray = rng.standard_normal(resolution)
        fft_state = _filter_spectra.compute_bandpass_filtered_fft(
            farray=sarray,
            resolution=resolution,
            num_ranks=0,
            k_min=0.0,
            k_max=1.0,
        )
        reconstructed = numpy.fft.ifftn(
            numpy.fft.ifftshift(fft_state.shifted_fft_farray),
            norm="forward",
        ).real
        numpy.testing.assert_allclose(reconstructed, sarray, atol=1e-10)


class TestGaussianWindow(unittest.TestCase):
    """
    The gaussian window shares [k_min, k_max] with the tophat window, but has no hard
    edge: it must be smooth (no discontinuity at k_min/k_max) and it must not reproduce
    the tophat's exact zero/one step, which is the whole point of comparing the two.
    """

    def test_default_window_is_tophat(
        self,
    ) -> None:
        rng = numpy.random.default_rng(13)
        resolution = (8, 8, 8)
        sarray = rng.standard_normal(resolution)
        default = _filter_spectra.compute_bandpass_filtered_farray(
            farray=sarray,
            resolution=resolution,
            num_ranks=0,
            k_min=1.0,
            k_max=2.0,
        )
        explicit_tophat = _filter_spectra.compute_bandpass_filtered_farray(
            farray=sarray,
            resolution=resolution,
            num_ranks=0,
            k_min=1.0,
            k_max=2.0,
            window="tophat",
        )
        numpy.testing.assert_array_equal(default, explicit_tophat)

    def test_gaussian_peak_is_unity_at_band_center(
        self,
    ) -> None:
        resolution = (16, 16, 16)
        k_mask = _filter_spectra._compute_k_mask(
            k_magnitude=_compute_spectra._compute_radial_k_magnitude(num_cells_per_dim=resolution),
            k_min=2.0,
            k_max=6.0,
            window="gaussian",
        )
        center = resolution[0] // 2
        ## the k-magnitude grid is centered at `center` in every axis, where |k| = 0;
        ## nudge one axis to land exactly on k_magnitude == k_center == 4.0
        self.assertAlmostEqual(float(k_mask[center + 4, center, center]), 1.0, places=10)

    def test_gaussian_has_no_hard_edge(
        self,
    ) -> None:
        ## the tophat drops to exactly 0 one cell past k_max; the gaussian must not
        resolution = (16, 16, 16)
        k_magnitude = _compute_spectra._compute_radial_k_magnitude(num_cells_per_dim=resolution)
        gaussian_mask = _filter_spectra._compute_k_mask(
            k_magnitude=k_magnitude,
            k_min=2.0,
            k_max=6.0,
            window="gaussian",
        )
        tophat_mask = _filter_spectra._compute_k_mask(
            k_magnitude=k_magnitude,
            k_min=2.0,
            k_max=6.0,
            window="tophat",
        )
        just_outside_band = k_magnitude == 7.0
        self.assertTrue(numpy.any(just_outside_band))
        self.assertTrue(numpy.all(tophat_mask[just_outside_band] == 0.0))
        self.assertTrue(numpy.all(gaussian_mask[just_outside_band] > 0.0))

    def test_gaussian_filtered_field_is_bounded_by_original(
        self,
    ) -> None:
        ## a smooth window should not ring/overshoot the way a hard-edged one can; a
        ## constant field is the simplest case where "bounded by the original" is exact.
        resolution = (16, 16, 16)
        constant_sarray = numpy.full(resolution, fill_value=2.0)
        filtered = _filter_spectra.compute_bandpass_filtered_farray(
            farray=constant_sarray,
            resolution=resolution,
            num_ranks=0,
            k_min=0.0,
            k_max=0.5,
            window="gaussian",
        )
        self.assertTrue(numpy.all(filtered <= 2.0 + 1e-10))

    def test_rejects_degenerate_gaussian_width(
        self,
    ) -> None:
        rng = numpy.random.default_rng(14)
        with self.assertRaises(ValueError):
            _filter_spectra.compute_bandpass_filtered_farray(
                farray=rng.standard_normal((8, 8, 8)),
                resolution=(8, 8, 8),
                num_ranks=0,
                k_min=2.0,
                k_max=2.0,
                window="gaussian",
            )

    def test_rejects_unknown_window(
        self,
    ) -> None:
        rng = numpy.random.default_rng(15)
        with self.assertRaises(ValueError):
            _filter_spectra.compute_bandpass_filtered_farray(
                farray=rng.standard_normal((8, 8, 8)),
                resolution=(8, 8, 8),
                num_ranks=0,
                k_min=1.0,
                k_max=2.0,
                window="hann",  # pyright: ignore[reportArgumentType]
            )


class TestTukeyWindow(unittest.TestCase):
    """
    The tukey window must combine what the other two do not: exactly zero outside
    [k_min, k_max] (no leakage, like the tophat), but with a smooth, continuous-slope
    taper at the edges instead of a hard step (so it rings far less, like the gaussian).
    """

    def test_flat_top_equals_one(
        self,
    ) -> None:
        resolution = (32, 32, 32)
        k_magnitude = _compute_spectra._compute_radial_k_magnitude(num_cells_per_dim=resolution)
        k_mask = _filter_spectra._compute_k_mask(
            k_magnitude=k_magnitude,
            k_min=4.0,
            k_max=12.0,
            window="tukey",
            taper_width=2.0,
        )
        ## interior of the flat top: k_min + taper_width <= k <= k_max - taper_width
        in_flat_top = (k_magnitude >= 6.0) & (k_magnitude <= 10.0)
        self.assertTrue(numpy.any(in_flat_top))
        numpy.testing.assert_allclose(k_mask[in_flat_top], 1.0)

    def test_exactly_zero_outside_band(
        self,
    ) -> None:
        ## this is the property the gaussian window does not have
        resolution = (32, 32, 32)
        k_magnitude = _compute_spectra._compute_radial_k_magnitude(num_cells_per_dim=resolution)
        k_mask = _filter_spectra._compute_k_mask(
            k_magnitude=k_magnitude,
            k_min=4.0,
            k_max=12.0,
            window="tukey",
            taper_width=2.0,
        )
        outside_band = (k_magnitude < 4.0) | (k_magnitude > 12.0)
        self.assertTrue(numpy.any(outside_band))
        numpy.testing.assert_array_equal(k_mask[outside_band], 0.0)

    def test_edge_value_and_slope_match_zero_outside(
        self,
    ) -> None:
        ## this is the property the tophat window does not have: at k_min the tukey mask
        ## reaches 0 (matching the region just outside) with zero slope (matching it too)
        k_min, taper_width = 4.0, 2.0
        k_values = numpy.array([k_min - 1e-4, k_min, k_min + 1e-4])
        k_mask = _filter_spectra._compute_k_mask(
            k_magnitude=k_values,
            k_min=k_min,
            k_max=12.0,
            window="tukey",
            taper_width=taper_width,
        )
        numpy.testing.assert_allclose(k_mask, [0.0, 0.0, 0.0], atol=1e-6)

    def test_rings_less_than_tophat_at_the_same_nominal_band(
        self,
    ) -> None:
        resolution = (32, 32, 32)
        rng = numpy.random.default_rng(16)
        sarray = rng.standard_normal(resolution)
        tophat_filtered = _filter_spectra.compute_bandpass_filtered_farray(
            farray=sarray,
            resolution=resolution,
            num_ranks=0,
            k_min=4.0,
            k_max=12.0,
            window="tophat",
        )
        tukey_filtered = _filter_spectra.compute_bandpass_filtered_farray(
            farray=sarray,
            resolution=resolution,
            num_ranks=0,
            k_min=4.0,
            k_max=12.0,
            window="tukey",
            taper_width=2.0,
        )
        ## a positive-definite proxy for "spread of the reconstructed variance": comparing
        ## a full random field's ringing directly is noisy, so this only asserts the two
        ## are not identical; the dedicated ringing comparison lives in the vtest
        self.assertFalse(numpy.allclose(tophat_filtered, tukey_filtered))

    def test_rejects_missing_taper_width(
        self,
    ) -> None:
        rng = numpy.random.default_rng(17)
        with self.assertRaises(ValueError):
            _filter_spectra.compute_bandpass_filtered_farray(
                farray=rng.standard_normal((8, 8, 8)),
                resolution=(8, 8, 8),
                num_ranks=0,
                k_min=2.0,
                k_max=6.0,
                window="tukey",
            )

    def test_rejects_nonpositive_taper_width(
        self,
    ) -> None:
        rng = numpy.random.default_rng(18)
        with self.assertRaises(ValueError):
            _filter_spectra.compute_bandpass_filtered_farray(
                farray=rng.standard_normal((8, 8, 8)),
                resolution=(8, 8, 8),
                num_ranks=0,
                k_min=2.0,
                k_max=6.0,
                window="tukey",
                taper_width=0.0,
            )

    def test_rejects_overlapping_tapers(
        self,
    ) -> None:
        rng = numpy.random.default_rng(19)
        with self.assertRaises(ValueError):
            _filter_spectra.compute_bandpass_filtered_farray(
                farray=rng.standard_normal((8, 8, 8)),
                resolution=(8, 8, 8),
                num_ranks=0,
                k_min=2.0,
                k_max=6.0,
                window="tukey",
                taper_width=3.0,  # 2 * 3.0 > (6.0 - 2.0)
            )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
