## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest
from typing import Any

## third-party
import numpy

## local
from jormi.ww_fields.fields_3d import (
    compute_spectra,
    domain_types,
    field_types,
)

##
## === HELPERS
##


def _make_3d_udomain(
    resolution: tuple[int, int, int],
) -> domain_types.UniformDomain_3D:
    return domain_types.UniformDomain_3D(
        periodicity=(True, True, True),
        resolution=resolution,
        domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    )


def _make_sfield(
    sarray_3d: numpy.ndarray[Any, numpy.dtype[Any]],
) -> field_types.ScalarField_3D:
    resolution = sarray_3d.shape
    return field_types.ScalarField_3D.from_3d_sarray(
        sarray_3d=sarray_3d,
        udomain_3d=_make_3d_udomain(resolution),
        field_label="f",
    )


def _cosine_mode_field(
    *,
    mode_k: int,
    num_cells: int,
) -> numpy.ndarray[Any, numpy.dtype[numpy.float64]]:
    """Return an N^3 array with a pure cosine wave at integer mode `mode_k` along x_0."""
    cell_indices = numpy.arange(num_cells)
    sarray_1d = numpy.cos(2.0 * numpy.pi * mode_k * cell_indices / num_cells)
    return numpy.broadcast_to(
        sarray_1d[:, numpy.newaxis, numpy.newaxis],
        (num_cells, num_cells, num_cells),
    ).copy()


##
## === TEST SUITES
##


class TestKBinCenters(unittest.TestCase):

    def test_bin_centers_are_integer_modes(
        self,
    ):
        ## k_bin_centers should be [1, 2, 3, ..., N//2] for all supported N
        for num_cells in (8, 16, 32):
            sfield = _make_sfield(numpy.ones((num_cells, num_cells, num_cells)))
            spectrum = compute_spectra.compute_isotropic_power_spectrum_sfield(sfield)
            expected_centers = numpy.arange(1, num_cells // 2 + 1, dtype=float)
            numpy.testing.assert_array_equal(
                spectrum.k_bin_centers_1d,
                expected_centers,
                err_msg=f"k_bin_centers mismatch for N={num_cells}",
            )

    def test_spectrum_and_centers_have_same_length(
        self,
    ):
        for num_cells in (8, 16):
            sfield = _make_sfield(numpy.ones((num_cells, num_cells, num_cells)))
            spectrum = compute_spectra.compute_isotropic_power_spectrum_sfield(sfield)
            self.assertEqual(
                spectrum.k_bin_centers_1d.shape,
                spectrum.spectrum_1d.shape,
            )


class TestDCExclusion(unittest.TestCase):

    def test_constant_field_produces_zero_spectrum(
        self,
    ):
        ## a constant field has all power in the DC mode (k=0), which is
        ## excluded from the output. The remaining spectrum should be zero.
        for num_cells in (8, 16):
            sarray_3d = numpy.full((num_cells, num_cells, num_cells), fill_value=3.14)
            sfield = _make_sfield(sarray_3d)
            spectrum = compute_spectra.compute_isotropic_power_spectrum_sfield(sfield)
            numpy.testing.assert_allclose(
                spectrum.spectrum_1d,
                numpy.zeros(num_cells // 2),
                atol=1e-10,
                err_msg=f"Constant field should produce zero spectrum for N={num_cells}",
            )


class TestPureModeBinPlacement(unittest.TestCase):
    """
    Verify that a pure cosine wave at integer mode k has its spectral
    peak in bin k (i.e., k_bin_centers[argmax] == k).

    This is the core correctness test. The k_center bug (using (N-1)/2
    instead of N//2) caused the DC component to leak into bin 1 and
    shifted all modes up by ~1 bin for even-N grids.
    """

    def _assert_peak_at_mode(
        self,
        *,
        mode_k: int,
        num_cells: int,
    ) -> None:
        sarray_3d = _cosine_mode_field(mode_k=mode_k, num_cells=num_cells)
        sfield = _make_sfield(sarray_3d)
        spectrum = compute_spectra.compute_isotropic_power_spectrum_sfield(sfield)
        peak_index = int(numpy.argmax(spectrum.spectrum_1d))
        self.assertEqual(
            spectrum.k_bin_centers_1d[peak_index],
            mode_k,
            msg=
            f"Expected peak at k={mode_k}, got k={spectrum.k_bin_centers_1d[peak_index]} for N={num_cells}",
        )

    def test_k1_even_N(
        self,
    ):
        ## regression: old code placed k=1 in bin 2 for even N
        ## because k_center used (N-1)/2 = 127.5 instead of N//2 = 128.
        for num_cells in (8, 16, 32):
            self._assert_peak_at_mode(mode_k=1, num_cells=num_cells)

    def test_k1_odd_N(
        self,
    ):
        ## for odd N, (N-1)/2 == N//2 (both integer), so the old code
        ## was correct. Verify correctness is preserved.
        for num_cells in (9, 15):
            self._assert_peak_at_mode(mode_k=1, num_cells=num_cells)

    def test_k2_mode(
        self,
    ):
        for num_cells in (8, 16):
            self._assert_peak_at_mode(mode_k=2, num_cells=num_cells)

    def test_k3_mode(
        self,
    ):
        self._assert_peak_at_mode(mode_k=3, num_cells=16)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
