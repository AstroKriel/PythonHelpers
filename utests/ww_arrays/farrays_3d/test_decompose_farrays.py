## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest
from typing import Any

## third-party
import numpy
from numpy.typing import NDArray

## local
from jormi.ww_arrays.farrays_3d import decompose_farrays

_N = 4
_SSHAPE = (_N, _N, _N)
_VSHAPE = (3, _N, _N, _N)

##
## === HELPERS
##


def _zeros_varray() -> NDArray[Any]:
    return numpy.zeros(_VSHAPE)


def _zeros_sarray() -> NDArray[Any]:
    return numpy.zeros(_SSHAPE)


##
## === TEST SUITES
##


class TestHelmholtzDecomposedFArrays3D(unittest.TestCase):

    def test_accepts_valid_components(
        self,
    ) -> None:
        decompose_farrays.HelmholtzDecomposedFArrays_3D(
            div_varray_3d=_zeros_varray(),
            sol_varray_3d=_zeros_varray(),
            bulk_varray_3d=_zeros_varray(),
        )

    def test_rejects_mismatched_shapes(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            decompose_farrays.HelmholtzDecomposedFArrays_3D(
                div_varray_3d=numpy.zeros((3, _N, _N, _N)),
                sol_varray_3d=numpy.zeros((3, _N + 1, _N, _N)),
                bulk_varray_3d=_zeros_varray(),
            )

    def test_rejects_non_varray(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            decompose_farrays.HelmholtzDecomposedFArrays_3D(
                div_varray_3d=_zeros_sarray(),  # pyright: ignore[reportArgumentType]
                sol_varray_3d=_zeros_varray(),
                bulk_varray_3d=_zeros_varray(),
            )


class TestTNBDecomposedFArrays3D(unittest.TestCase):

    def test_accepts_valid_components(
        self,
    ) -> None:
        decompose_farrays.TNBDecomposedFArrays_3D(
            tangent_uvarray_3d=_zeros_varray(),
            normal_uvarray_3d=_zeros_varray(),
            binormal_uvarray_3d=_zeros_varray(),
            curvature_sarray_3d=_zeros_sarray(),
        )

    def test_rejects_mismatched_vector_shapes(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            decompose_farrays.TNBDecomposedFArrays_3D(
                tangent_uvarray_3d=numpy.zeros((3, _N, _N, _N)),
                normal_uvarray_3d=numpy.zeros((3, _N + 1, _N, _N)),
                binormal_uvarray_3d=_zeros_varray(),
                curvature_sarray_3d=_zeros_sarray(),
            )

    def test_rejects_mismatched_curvature_shape(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            decompose_farrays.TNBDecomposedFArrays_3D(
                tangent_uvarray_3d=_zeros_varray(),
                normal_uvarray_3d=_zeros_varray(),
                binormal_uvarray_3d=_zeros_varray(),
                curvature_sarray_3d=numpy.zeros((_N + 1, _N, _N)),
            )


class TestMagneticCurvatureFArrays3D(unittest.TestCase):

    def test_accepts_valid_components(
        self,
    ) -> None:
        decompose_farrays.MagneticCurvatureFArrays_3D(
            curvature_sarray_3d=_zeros_sarray(),
            stretching_sarray_3d=_zeros_sarray(),
            compression_sarray_3d=_zeros_sarray(),
        )

    def test_rejects_mismatched_shapes(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            decompose_farrays.MagneticCurvatureFArrays_3D(
                curvature_sarray_3d=_zeros_sarray(),
                stretching_sarray_3d=numpy.zeros((_N + 1, _N, _N)),
                compression_sarray_3d=_zeros_sarray(),
            )


class TestLorentzForceFArrays3D(unittest.TestCase):

    def test_accepts_valid_components(
        self,
    ) -> None:
        decompose_farrays.LorentzForceFArrays_3D(
            lorentz_varray_3d=_zeros_varray(),
            tension_varray_3d=_zeros_varray(),
            grad_p_perp_varray_3d=_zeros_varray(),
        )

    def test_rejects_mismatched_shapes(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            decompose_farrays.LorentzForceFArrays_3D(
                lorentz_varray_3d=_zeros_varray(),
                tension_varray_3d=numpy.zeros((3, _N + 1, _N, _N)),
                grad_p_perp_varray_3d=_zeros_varray(),
            )

    def test_rejects_non_varray(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            decompose_farrays.LorentzForceFArrays_3D(
                lorentz_varray_3d=_zeros_sarray(),  # pyright: ignore[reportArgumentType]
                tension_varray_3d=_zeros_varray(),
                grad_p_perp_varray_3d=_zeros_varray(),
            )


## } U-TEST
