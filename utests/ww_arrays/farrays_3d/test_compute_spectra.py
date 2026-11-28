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


def _create_abc_varray(
    *,
    num_cells: int,
) -> numpy.ndarray:
    """ABC field, which satisfies curl(v) = +v and so is right-handed."""
    coordinates = 2.0 * numpy.pi * numpy.arange(num_cells) / num_cells
    x_grid, y_grid, z_grid = numpy.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    return numpy.stack(
        [
            numpy.sin(z_grid) + numpy.cos(y_grid),
            numpy.sin(x_grid) + numpy.cos(z_grid),
            numpy.sin(y_grid) + numpy.cos(x_grid),
        ],
    )


def _mirror_varray_in_x(
    *,
    varray_3d: numpy.ndarray,
) -> numpy.ndarray:
    """Reflect a periodic vector field through x = 0, which flips its handedness."""
    reflected_varray = numpy.roll(numpy.flip(varray_3d, axis=1), shift=1, axis=1)
    reflected_varray[0] = -reflected_varray[0]
    return reflected_varray


class TestVarrayHelicalPowerSpectrum(unittest.TestCase):

    def _compute_both_spectra(
        self,
        *,
        varray_3d: numpy.ndarray,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        resolution_3d = (varray_3d.shape[1], varray_3d.shape[2], varray_3d.shape[3])
        power_spectrum = compute_spectra.compute_isotropic_power_spectrum_varray(
            varray_3d=varray_3d,
            resolution_3d=resolution_3d,
        )
        helical_spectrum = compute_spectra.compute_isotropic_helical_power_spectrum_varray(
            varray_3d=varray_3d,
            resolution_3d=resolution_3d,
        )
        return power_spectrum.power_spectrum_1d, helical_spectrum.power_spectrum_1d

    def test_right_handed_field_matches_power_spectrum(
        self,
    ) -> None:
        abc_varray = _create_abc_varray(num_cells=16)
        power_1d, helical_1d = self._compute_both_spectra(varray_3d=abc_varray)
        numpy.testing.assert_allclose(helical_1d, power_1d, atol=1e-12)
        self.assertGreater(helical_1d[0], 0.0)

    def test_left_handed_field_matches_negative_power_spectrum(
        self,
    ) -> None:
        mirrored_varray = _mirror_varray_in_x(varray_3d=_create_abc_varray(num_cells=16))
        power_1d, helical_1d = self._compute_both_spectra(varray_3d=mirrored_varray)
        numpy.testing.assert_allclose(helical_1d, -power_1d, atol=1e-12)

    def test_random_field_is_bounded_by_power_spectrum(
        self,
    ) -> None:
        rng = numpy.random.default_rng(7)
        random_varray = rng.standard_normal((3, 16, 16, 16))
        power_1d, helical_1d = self._compute_both_spectra(varray_3d=random_varray)
        self.assertTrue(numpy.all(numpy.abs(helical_1d) <= power_1d + 1e-12))
        self.assertLess(numpy.max(numpy.abs(helical_1d) / power_1d), 0.9)


class TestR2TarrayPowerSpectrum(unittest.TestCase):
    """Same checks as TestVarrayPowerSpectrum, one rank up: the tensor spectrum is built
    by FFT-ing each of the 9 components and summing |.|^2 in k-space."""

    def test_reduces_to_scalar_spectrum_for_single_nonzero_component(
        self,
    ) -> None:
        for num_cells in (8, 16):
            resolution_3d = (num_cells, num_cells, num_cells)
            rng = numpy.random.default_rng(3)
            comp_xy = rng.standard_normal(resolution_3d)
            r2tarray_3d = numpy.zeros((3, 3, *resolution_3d))
            r2tarray_3d[0, 1] = comp_xy
            tensor_spectrum = compute_spectra.compute_isotropic_power_spectrum_r2tarray(
                r2tarray_3d=r2tarray_3d,
                resolution_3d=resolution_3d,
            )
            scalar_spectrum = compute_spectra.compute_isotropic_power_spectrum_sarray(
                sarray_3d=comp_xy,
                resolution_3d=resolution_3d,
            )
            numpy.testing.assert_allclose(
                tensor_spectrum.power_spectrum_1d,
                scalar_spectrum.power_spectrum_1d,
                err_msg=f"Tensor spectrum with one nonzero component should match the scalar spectrum for N={num_cells}",
            )

    def test_parseval_total_power_matches_real_space_energy(
        self,
    ) -> None:
        for num_cells in (8, 16):
            resolution_3d = (num_cells, num_cells, num_cells)
            rng = numpy.random.default_rng(4)
            r2tarray_3d = rng.standard_normal((3, 3, *resolution_3d))
            power_spectrum_3d = compute_spectra.compute_power_spectrum_farray(
                farray_3d=r2tarray_3d,
                resolution_3d=resolution_3d,
                num_ranks=2,
            )
            total_power_k_space = numpy.sum(power_spectrum_3d)
            total_power_real_space = numpy.sum(numpy.square(r2tarray_3d)) / (num_cells**3)
            numpy.testing.assert_allclose(
                total_power_k_space,
                total_power_real_space,
                rtol=1e-10,
                err_msg=f"Parseval mismatch for N={num_cells}",
            )

    def test_matches_generic_kernel_directly(
        self,
    ) -> None:
        ## the farrays_3d wrapper must not diverge from the shared N-dimensional core it
        ## delegates to
        num_cells = 16
        resolution_3d = (num_cells, num_cells, num_cells)
        rng = numpy.random.default_rng(5)
        r2tarray_3d = rng.standard_normal((3, 3, *resolution_3d))
        via_wrapper = compute_spectra.compute_isotropic_power_spectrum_r2tarray(
            r2tarray_3d=r2tarray_3d,
            resolution_3d=resolution_3d,
        )
        via_farray = compute_spectra.compute_isotropic_power_spectrum_farray(
            farray_3d=r2tarray_3d,
            resolution_3d=resolution_3d,
            num_ranks=2,
        )
        numpy.testing.assert_allclose(
            via_wrapper.power_spectrum_1d,
            via_farray.power_spectrum_1d,
        )

    def test_rejects_wrong_leading_shape(
        self,
    ) -> None:
        rng = numpy.random.default_rng(6)
        with self.assertRaises(ValueError):
            compute_spectra.compute_isotropic_power_spectrum_r2tarray(
                r2tarray_3d=rng.standard_normal((3, 2, 8, 8, 8)),
                resolution_3d=(8, 8, 8),
            )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
