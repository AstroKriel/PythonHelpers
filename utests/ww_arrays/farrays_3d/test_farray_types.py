## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from jormi.ww_arrays.farrays_3d import farray_types

_N = 4
_SSHAPE = (_N, _N, _N)
_VSHAPE = (3, _N, _N, _N)
_R2TSHAPE = (3, 3, _N, _N, _N)

##
## === TEST SUITES
##


class TestEnsure3dSarray(unittest.TestCase):

    def test_accepts_3d_array(
        self,
    ) -> None:
        farray_types.ensure_3d_sarray(
            numpy.zeros(
                _SSHAPE,
            ),
        )

    def test_rejects_2d_array(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            farray_types.ensure_3d_sarray(
                numpy.zeros((_N, _N), ),
            )

    def test_rejects_4d_array(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            farray_types.ensure_3d_sarray(
                numpy.zeros((_N, _N, _N, _N), ),
            )

    def test_rejects_1d_array(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            farray_types.ensure_3d_sarray(
                numpy.zeros((_N, ), ),
            )


class TestEnsure3dVarray(unittest.TestCase):

    def test_accepts_valid_varray(
        self,
    ) -> None:
        farray_types.ensure_3d_varray(
            numpy.zeros(
                _VSHAPE,
            ),
        )

    def test_rejects_wrong_leading_axis(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            farray_types.ensure_3d_varray(
                numpy.zeros((2, _N, _N, _N), ),
            )

    def test_rejects_3d_array(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            farray_types.ensure_3d_varray(
                numpy.zeros(
                    _SSHAPE,
                ),
            )

    def test_rejects_5d_array(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            farray_types.ensure_3d_varray(
                numpy.zeros((3, _N, _N, _N, _N), ),
            )


class TestEnsure3dR2tarray(unittest.TestCase):

    def test_accepts_valid_r2tarray(
        self,
    ) -> None:
        farray_types.ensure_3d_r2tarray(
            numpy.zeros(
                _R2TSHAPE,
            ),
        )

    def test_rejects_wrong_first_axis(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            farray_types.ensure_3d_r2tarray(
                numpy.zeros((2, 3, _N, _N, _N), ),
            )

    def test_rejects_wrong_second_axis(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            farray_types.ensure_3d_r2tarray(
                numpy.zeros((3, 2, _N, _N, _N), ),
            )

    def test_rejects_4d_array(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            farray_types.ensure_3d_r2tarray(
                numpy.zeros(
                    _VSHAPE,
                ),
            )


class TestEnsureFarrayMetadata(unittest.TestCase):

    def test_allocates_when_farray_is_none(
        self,
    ) -> None:
        out = farray_types.ensure_farray_metadata(
            farray_shape=_SSHAPE,
            farray=None,
        )
        self.assertEqual(
            out.shape,
            _SSHAPE,
        )

    def test_default_dtype_is_float64(
        self,
    ) -> None:
        out = farray_types.ensure_farray_metadata(
            farray_shape=_SSHAPE,
            farray=None,
        )
        self.assertEqual(
            out.dtype,
            numpy.float64,
        )

    def test_respects_explicit_dtype(
        self,
    ) -> None:
        out = farray_types.ensure_farray_metadata(
            farray_shape=_SSHAPE,
            farray=None,
            dtype=numpy.float32,
        )
        self.assertEqual(
            out.dtype,
            numpy.float32,
        )

    def test_reuses_compatible_farray(
        self,
    ) -> None:
        existing = numpy.zeros(_SSHAPE, dtype=numpy.float64)
        out = farray_types.ensure_farray_metadata(
            farray_shape=_SSHAPE,
            farray=existing,
        )
        self.assertIs(
            out,
            existing,
        )

    def test_reallocates_on_shape_mismatch(
        self,
    ) -> None:
        existing = numpy.zeros((_N, _N), dtype=numpy.float64)
        out = farray_types.ensure_farray_metadata(
            farray_shape=_SSHAPE,
            farray=existing,
        )
        self.assertEqual(
            out.shape,
            _SSHAPE,
        )
        self.assertIsNot(
            out,
            existing,
        )

    def test_reallocates_on_dtype_mismatch(
        self,
    ) -> None:
        existing = numpy.zeros(_SSHAPE, dtype=numpy.float32)
        out = farray_types.ensure_farray_metadata(
            farray_shape=_SSHAPE,
            farray=existing,
            dtype=numpy.float64,
        )
        self.assertIsNot(
            out,
            existing,
        )
        self.assertEqual(
            out.dtype,
            numpy.float64,
        )


## } U-TEST
