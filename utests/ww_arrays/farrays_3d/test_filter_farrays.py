## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from jormi.ww_arrays.farrays_3d import filter_farrays

##
## === TEST SUITES
##


class TestSarrayMatchesGenericKernel(unittest.TestCase):

    def test_matches_array_level_kernel(
        self,
    ) -> None:
        num_cells = 8
        resolution_3d = (num_cells, num_cells, num_cells)
        rng = numpy.random.default_rng(0)
        sarray_3d = rng.standard_normal(resolution_3d)
        via_sarray = filter_farrays.compute_bandpass_filtered_sarray(
            sarray_3d=sarray_3d,
            resolution_3d=resolution_3d,
            k_min=1.0,
            k_max=2.0,
        )
        via_farray = filter_farrays.compute_bandpass_filtered_farray(
            farray_3d=sarray_3d,
            resolution_3d=resolution_3d,
            num_ranks=0,
            k_min=1.0,
            k_max=2.0,
        )
        numpy.testing.assert_allclose(via_sarray, via_farray)

    def test_rejects_non_3d_array(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            filter_farrays.compute_bandpass_filtered_sarray(
                sarray_3d=numpy.ones((8, 8)),
                resolution_3d=(8, 8),  # pyright: ignore[reportArgumentType]
                k_min=1.0,
                k_max=2.0,
            )


class TestVarrayMatchesGenericKernel(unittest.TestCase):

    def test_matches_array_level_kernel(
        self,
    ) -> None:
        num_cells = 8
        resolution_3d = (num_cells, num_cells, num_cells)
        rng = numpy.random.default_rng(1)
        varray_3d = rng.standard_normal((3, *resolution_3d))
        via_varray = filter_farrays.compute_bandpass_filtered_varray(
            varray_3d=varray_3d,
            resolution_3d=resolution_3d,
            k_min=1.0,
            k_max=2.0,
        )
        via_farray = filter_farrays.compute_bandpass_filtered_farray(
            farray_3d=varray_3d,
            resolution_3d=resolution_3d,
            num_ranks=1,
            k_min=1.0,
            k_max=2.0,
        )
        numpy.testing.assert_allclose(via_varray, via_farray)

    def test_rejects_wrong_num_components(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            filter_farrays.compute_bandpass_filtered_varray(
                varray_3d=numpy.ones((2, 8, 8, 8)),
                resolution_3d=(8, 8, 8),
                k_min=1.0,
                k_max=2.0,
            )


class TestR2TarrayMatchesGenericKernel(unittest.TestCase):

    def test_matches_array_level_kernel(
        self,
    ) -> None:
        num_cells = 8
        resolution_3d = (num_cells, num_cells, num_cells)
        rng = numpy.random.default_rng(2)
        r2tarray_3d = rng.standard_normal((3, 3, *resolution_3d))
        via_r2tarray = filter_farrays.compute_bandpass_filtered_r2tarray(
            r2tarray_3d=r2tarray_3d,
            resolution_3d=resolution_3d,
            k_min=1.0,
            k_max=2.0,
        )
        via_farray = filter_farrays.compute_bandpass_filtered_farray(
            farray_3d=r2tarray_3d,
            resolution_3d=resolution_3d,
            num_ranks=2,
            k_min=1.0,
            k_max=2.0,
        )
        numpy.testing.assert_allclose(via_r2tarray, via_farray)

    def test_rejects_wrong_num_components(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            filter_farrays.compute_bandpass_filtered_r2tarray(
                r2tarray_3d=numpy.ones((3, 2, 8, 8, 8)),
                resolution_3d=(8, 8, 8),
                k_min=1.0,
                k_max=2.0,
            )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
