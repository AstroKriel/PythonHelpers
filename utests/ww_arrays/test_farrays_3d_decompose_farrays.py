## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from jormi.ww_arrays.farrays_3d import decompose_farrays

_N = 4
_SSHAPE = (_N, _N, _N)
_VSHAPE = (3, _N, _N, _N)

##
## === HELPERS
##


def _zeros_varray() -> numpy.ndarray:
    return numpy.zeros(_VSHAPE)


def _zeros_sarray() -> numpy.ndarray:
    return numpy.zeros(_SSHAPE)


##
## === TEST SUITES
##


class TestHelmholtzDecomposedFArrays3D(unittest.TestCase):

    def test_accepts_valid_components(
        self,
    ) -> None:
        decompose_farrays.HelmholtzDecomposedFArrays_3D(
            varray_3d_div=_zeros_varray(),
            varray_3d_sol=_zeros_varray(),
            varray_3d_bulk=_zeros_varray(),
        )

    def test_rejects_mismatched_shapes(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            decompose_farrays.HelmholtzDecomposedFArrays_3D(
                varray_3d_div=numpy.zeros((3, _N, _N, _N)),
                varray_3d_sol=numpy.zeros((3, _N + 1, _N, _N)),
                varray_3d_bulk=_zeros_varray(),
            )

    def test_rejects_non_varray(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            decompose_farrays.HelmholtzDecomposedFArrays_3D(
                varray_3d_div=_zeros_sarray(),  # type: ignore
                varray_3d_sol=_zeros_varray(),
                varray_3d_bulk=_zeros_varray(),
            )


class TestTNBDecomposedFArrays3D(unittest.TestCase):

    def test_accepts_valid_components(
        self,
    ) -> None:
        decompose_farrays.TNBDecomposedFArrays_3D(
            uvarray_3d_tangent=_zeros_varray(),
            uvarray_3d_normal=_zeros_varray(),
            uvarray_3d_binormal=_zeros_varray(),
            sarray_3d_curvature=_zeros_sarray(),
        )

    def test_rejects_mismatched_vector_shapes(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            decompose_farrays.TNBDecomposedFArrays_3D(
                uvarray_3d_tangent=numpy.zeros((3, _N, _N, _N)),
                uvarray_3d_normal=numpy.zeros((3, _N + 1, _N, _N)),
                uvarray_3d_binormal=_zeros_varray(),
                sarray_3d_curvature=_zeros_sarray(),
            )

    def test_rejects_mismatched_curvature_shape(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            decompose_farrays.TNBDecomposedFArrays_3D(
                uvarray_3d_tangent=_zeros_varray(),
                uvarray_3d_normal=_zeros_varray(),
                uvarray_3d_binormal=_zeros_varray(),
                sarray_3d_curvature=numpy.zeros((_N + 1, _N, _N)),
            )


class TestMagneticCurvatureFArrays3D(unittest.TestCase):

    def test_accepts_valid_components(
        self,
    ) -> None:
        decompose_farrays.MagneticCurvatureFArrays_3D(
            sarray_3d_curvature=_zeros_sarray(),
            sarray_3d_stretching=_zeros_sarray(),
            sarray_3d_compression=_zeros_sarray(),
        )

    def test_rejects_mismatched_shapes(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            decompose_farrays.MagneticCurvatureFArrays_3D(
                sarray_3d_curvature=_zeros_sarray(),
                sarray_3d_stretching=numpy.zeros((_N + 1, _N, _N)),
                sarray_3d_compression=_zeros_sarray(),
            )


class TestLorentzForceFArrays3D(unittest.TestCase):

    def test_accepts_valid_components(
        self,
    ) -> None:
        decompose_farrays.LorentzForceFArrays_3D(
            varray_3d_lorentz=_zeros_varray(),
            varray_3d_tension=_zeros_varray(),
            varray_3d_grad_p_perp=_zeros_varray(),
        )

    def test_rejects_mismatched_shapes(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            decompose_farrays.LorentzForceFArrays_3D(
                varray_3d_lorentz=_zeros_varray(),
                varray_3d_tension=numpy.zeros((3, _N + 1, _N, _N)),
                varray_3d_grad_p_perp=_zeros_varray(),
            )

    def test_rejects_non_varray(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            decompose_farrays.LorentzForceFArrays_3D(
                varray_3d_lorentz=_zeros_sarray(),  # type: ignore
                varray_3d_tension=_zeros_varray(),
                varray_3d_grad_p_perp=_zeros_varray(),
            )


## } U-TEST
