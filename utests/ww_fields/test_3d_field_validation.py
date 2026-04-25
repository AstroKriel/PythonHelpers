## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from jormi.ww_fields.fields_3d import (
    domain_types,
    field_types,
)

##
## === HELPERS
##


def _make_3d_udomain(
    *,
    resolution: tuple[int, int, int] = (4, 4, 4),
    domain_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
) -> domain_types.UniformDomain_3D:
    return domain_types.UniformDomain_3D(
        periodicity=(True, True, True),
        resolution=resolution,
        domain_bounds=domain_bounds,
    )


def _make_sfield_3d(
    *,
    resolution: tuple[int, int, int] = (4, 4, 4),
    domain: domain_types.UniformDomain_3D | None = None,
) -> field_types.ScalarField_3D:
    if domain is None:
        domain = _make_3d_udomain(resolution=resolution)
    return field_types.ScalarField_3D.from_3d_sarray(
        sarray_3d=numpy.ones(resolution),
        udomain_3d=domain,
        field_label="f",
    )


def _make_vfield_3d(
    *,
    resolution: tuple[int, int, int] = (4, 4, 4),
    domain: domain_types.UniformDomain_3D | None = None,
) -> field_types.VectorField_3D:
    if domain is None:
        domain = _make_3d_udomain(resolution=resolution)
    return field_types.VectorField_3D.from_3d_varray(
        varray_3d=numpy.ones((3, ) + resolution),
        udomain_3d=domain,
        field_label="v",
    )


def _make_unit_vfield_3d(
    resolution: tuple[int, int, int] = (4, 4, 4),
) -> field_types.UnitVectorField_3D:
    varray = numpy.zeros((3, ) + resolution)
    varray[0] = 1.0
    vfield = field_types.VectorField_3D.from_3d_varray(
        varray_3d=varray,
        udomain_3d=_make_3d_udomain(resolution=resolution),
        field_label="uv",
    )
    return field_types.UnitVectorField_3D.from_3d_vfield(vfield)


##
## === TEST SUITES
##


class TestEnsureFieldTypes(unittest.TestCase):

    def test_ensure_3d_sfield_accepts_scalar_field(
        self,
    ):
        sfield = _make_sfield_3d()
        field_types.ensure_3d_sfield(sfield_3d=sfield)

    def test_ensure_3d_sfield_rejects_vector_field(
        self,
    ):
        vfield = _make_vfield_3d()
        with self.assertRaises(TypeError):
            field_types.ensure_3d_sfield(sfield_3d=vfield)  # type: ignore

    def test_ensure_3d_sfield_rejects_none(
        self,
    ):
        with self.assertRaises(TypeError):
            field_types.ensure_3d_sfield(sfield_3d=None)  # type: ignore

    def test_ensure_3d_vfield_accepts_vector_field(
        self,
    ):
        vfield = _make_vfield_3d()
        field_types.ensure_3d_vfield(vfield_3d=vfield)

    def test_ensure_3d_vfield_rejects_scalar_field(
        self,
    ):
        sfield = _make_sfield_3d()
        with self.assertRaises(TypeError):
            field_types.ensure_3d_vfield(vfield_3d=sfield)  # type: ignore

    def test_ensure_3d_vfield_rejects_none(
        self,
    ):
        with self.assertRaises(TypeError):
            field_types.ensure_3d_vfield(vfield_3d=None)  # type: ignore

    def test_ensure_3d_uvfield_accepts_unit_vector_field(
        self,
    ):
        uvfield = _make_unit_vfield_3d()
        field_types.ensure_3d_uvfield(uvfield_3d=uvfield)

    def test_ensure_3d_uvfield_accepts_uvfield_as_vfield(
        self,
    ):
        uvfield = _make_unit_vfield_3d()
        field_types.ensure_3d_vfield(vfield_3d=uvfield)

    def test_ensure_3d_uvfield_rejects_plain_vector_field(
        self,
    ):
        varray = numpy.zeros((3, 4, 4, 4))
        varray[0] = 1.0
        vfield = field_types.VectorField_3D.from_3d_varray(
            varray_3d=varray,
            udomain_3d=_make_3d_udomain(),
            field_label="unit_but_not_uvfield",
        )
        with self.assertRaises(TypeError):
            field_types.ensure_3d_uvfield(uvfield_3d=vfield)  # type: ignore

    def test_ensure_3d_uvfield_rejects_none(
        self,
    ):
        with self.assertRaises(TypeError):
            field_types.ensure_3d_uvfield(uvfield_3d=None)  # type: ignore


class TestEnsureUdomainMatchesField(unittest.TestCase):

    def test_udomain_matches_sfield_passes(
        self,
    ):
        domain = _make_3d_udomain()
        sfield = _make_sfield_3d(domain=domain)
        field_types.ensure_3d_udomain_matches_sfield(
            sfield_3d=sfield,
            udomain_3d=domain,
        )

    def test_udomain_matches_sfield_fails_with_different_domain(
        self,
    ):
        domain_a = _make_3d_udomain(domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
        domain_b = _make_3d_udomain(domain_bounds=((0.0, 2.0), (0.0, 2.0), (0.0, 2.0)))
        sfield = _make_sfield_3d(domain=domain_a)
        with self.assertRaises(ValueError):
            field_types.ensure_3d_udomain_matches_sfield(
                sfield_3d=sfield,
                udomain_3d=domain_b,
            )

    def test_udomain_matches_sfield_rejects_wrong_field_type(
        self,
    ):
        domain = _make_3d_udomain()
        vfield = _make_vfield_3d(domain=domain)
        with self.assertRaises(TypeError):
            field_types.ensure_3d_udomain_matches_sfield(
                sfield_3d=vfield,  # type: ignore
                udomain_3d=domain,
            )

    def test_udomain_matches_vfield_passes(
        self,
    ):
        domain = _make_3d_udomain()
        vfield = _make_vfield_3d(domain=domain)
        field_types.ensure_3d_udomain_matches_vfield(
            vfield_3d=vfield,
            udomain_3d=domain,
        )

    def test_udomain_matches_vfield_fails_with_different_domain(
        self,
    ):
        domain_a = _make_3d_udomain(domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
        domain_b = _make_3d_udomain(domain_bounds=((0.0, 2.0), (0.0, 2.0), (0.0, 2.0)))
        vfield = _make_vfield_3d(domain=domain_a)
        with self.assertRaises(ValueError):
            field_types.ensure_3d_udomain_matches_vfield(
                vfield_3d=vfield,
                udomain_3d=domain_b,
            )

    def test_udomain_matches_vfield_rejects_wrong_field_type(
        self,
    ):
        domain = _make_3d_udomain()
        sfield = _make_sfield_3d(domain=domain)
        with self.assertRaises(TypeError):
            field_types.ensure_3d_udomain_matches_vfield(
                vfield_3d=sfield,  # type: ignore
                udomain_3d=domain,
            )


class TestEnsureSameShape(unittest.TestCase):

    def test_same_shape_passes_for_two_sfields(
        self,
    ):
        sfield_a = _make_sfield_3d(resolution=(4, 4, 4))
        sfield_b = _make_sfield_3d(resolution=(4, 4, 4))
        field_types.ensure_same_3d_field_shape(
            field_3d_a=sfield_a,
            field_3d_b=sfield_b,
        )

    def test_same_shape_passes_for_two_vfields(
        self,
    ):
        vfield_a = _make_vfield_3d(resolution=(4, 4, 4))
        vfield_b = _make_vfield_3d(resolution=(4, 4, 4))
        field_types.ensure_same_3d_field_shape(
            field_3d_a=vfield_a,
            field_3d_b=vfield_b,
        )

    def test_different_resolution_raises(
        self,
    ):
        sfield_a = _make_sfield_3d(resolution=(4, 4, 4))
        sfield_b = _make_sfield_3d(resolution=(8, 8, 8))
        with self.assertRaises(ValueError):
            field_types.ensure_same_3d_field_shape(
                field_3d_a=sfield_a,
                field_3d_b=sfield_b,
            )

    def test_scalar_and_vector_same_resolution_raises(
        self,
    ):
        sfield = _make_sfield_3d(resolution=(4, 4, 4))
        vfield = _make_vfield_3d(resolution=(4, 4, 4))
        with self.assertRaises(ValueError):
            field_types.ensure_same_3d_field_shape(
                field_3d_a=sfield,
                field_3d_b=vfield,
            )


class TestEnsureSameUdomains(unittest.TestCase):

    def test_same_domain_passes(
        self,
    ):
        domain = _make_3d_udomain()
        sfield_a = _make_sfield_3d(domain=domain)
        sfield_b = _make_sfield_3d(domain=domain)
        field_types.ensure_same_3d_field_udomains(
            field_3d_a=sfield_a,
            field_3d_b=sfield_b,
        )

    def test_equivalent_domain_params_pass(
        self,
    ):
        sfield_a = _make_sfield_3d(resolution=(4, 4, 4))
        sfield_b = _make_sfield_3d(resolution=(4, 4, 4))
        field_types.ensure_same_3d_field_udomains(
            field_3d_a=sfield_a,
            field_3d_b=sfield_b,
        )

    def test_different_bounds_raises(
        self,
    ):
        domain_a = _make_3d_udomain(domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
        domain_b = _make_3d_udomain(domain_bounds=((0.0, 2.0), (0.0, 2.0), (0.0, 2.0)))
        sfield_a = _make_sfield_3d(domain=domain_a)
        sfield_b = _make_sfield_3d(domain=domain_b)
        with self.assertRaises(ValueError):
            field_types.ensure_same_3d_field_udomains(
                field_3d_a=sfield_a,
                field_3d_b=sfield_b,
            )

    def test_different_periodicity_raises(
        self,
    ):
        domain_a = domain_types.UniformDomain_3D(
            periodicity=(True, True, True),
            resolution=(4, 4, 4),
            domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        )
        domain_b = domain_types.UniformDomain_3D(
            periodicity=(True, True, False),
            resolution=(4, 4, 4),
            domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        )
        sfield_a = _make_sfield_3d(domain=domain_a)
        sfield_b = _make_sfield_3d(domain=domain_b)
        with self.assertRaises(ValueError):
            field_types.ensure_same_3d_field_udomains(
                field_3d_a=sfield_a,
                field_3d_b=sfield_b,
            )


class TestEnsureSameShapeAndUdomains(unittest.TestCase):

    def test_identical_fields_pass(
        self,
    ):
        domain = _make_3d_udomain()
        sfield_a = _make_sfield_3d(domain=domain)
        sfield_b = _make_sfield_3d(domain=domain)
        field_types.ensure_same_3d_field_shape_and_udomains(
            field_3d_a=sfield_a,
            field_3d_b=sfield_b,
        )

    def test_different_resolution_raises(
        self,
    ):
        sfield_a = _make_sfield_3d(resolution=(4, 4, 4))
        sfield_b = _make_sfield_3d(resolution=(8, 8, 8))
        with self.assertRaises(ValueError):
            field_types.ensure_same_3d_field_shape_and_udomains(
                field_3d_a=sfield_a,
                field_3d_b=sfield_b,
            )

    def test_different_bounds_raises(
        self,
    ):
        sfield_a = _make_sfield_3d(
            domain=_make_3d_udomain(
                domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            ),
        )
        sfield_b = _make_sfield_3d(
            domain=_make_3d_udomain(
                domain_bounds=((0.0, 2.0), (0.0, 2.0), (0.0, 2.0)),
            ),
        )
        with self.assertRaises(ValueError):
            field_types.ensure_same_3d_field_shape_and_udomains(
                field_3d_a=sfield_a,
                field_3d_b=sfield_b,
            )


class TestExtractArrays(unittest.TestCase):

    def test_extract_3d_sarray_returns_correct_array(
        self,
    ):
        sarray = numpy.arange(64, dtype=float).reshape((4, 4, 4))
        sfield = field_types.ScalarField_3D.from_3d_sarray(
            sarray_3d=sarray,
            udomain_3d=_make_3d_udomain(),
            field_label="f",
        )
        result = field_types.extract_3d_sarray(sfield_3d=sfield)
        self.assertEqual(
            result.shape,
            (4, 4, 4),
        )
        self.assertTrue(
            numpy.array_equal(
                result,
                sarray,
            ),
        )

    def test_extract_3d_sarray_rejects_vector_field(
        self,
    ):
        vfield = _make_vfield_3d()
        with self.assertRaises(TypeError):
            field_types.extract_3d_sarray(sfield_3d=vfield)  # type: ignore

    def test_extract_3d_sarray_rejects_none(
        self,
    ):
        with self.assertRaises(TypeError):
            field_types.extract_3d_sarray(sfield_3d=None)  # type: ignore

    def test_extract_3d_varray_returns_correct_array(
        self,
    ):
        varray = numpy.arange(192, dtype=float).reshape((3, 4, 4, 4))
        vfield = field_types.VectorField_3D.from_3d_varray(
            varray_3d=varray,
            udomain_3d=_make_3d_udomain(),
            field_label="v",
        )
        result = field_types.extract_3d_varray(vfield_3d=vfield)
        self.assertEqual(
            result.shape,
            (3, 4, 4, 4),
        )
        self.assertTrue(
            numpy.array_equal(
                result,
                varray,
            ),
        )

    def test_extract_3d_varray_rejects_scalar_field(
        self,
    ):
        sfield = _make_sfield_3d()
        with self.assertRaises(TypeError):
            field_types.extract_3d_varray(vfield_3d=sfield)  # type: ignore

    def test_extract_3d_varray_rejects_none(
        self,
    ):
        with self.assertRaises(TypeError):
            field_types.extract_3d_varray(vfield_3d=None)  # type: ignore

    def test_extract_3d_varray_accepts_uvfield(
        self,
    ):
        uvfield = _make_unit_vfield_3d()
        result = field_types.extract_3d_varray(vfield_3d=uvfield)
        self.assertEqual(
            result.shape,
            (3, 4, 4, 4),
        )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
