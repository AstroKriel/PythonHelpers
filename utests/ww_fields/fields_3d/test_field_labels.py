## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_3d import domain_models, field_models
from jormi.ww_plots import latex_labels

##
## === HELPERS
##

_RESOLUTION = (4, 4, 4)


def _make_3d_uniform_domain() -> domain_models.UniformDomain_3D:
    return domain_models.UniformDomain_3D(
        periodicity=(True, True, True),
        resolution=_RESOLUTION,
        domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    )


def _make_sfield_3d() -> field_models.ScalarField_3D:
    return field_models.ScalarField_3D.from_3d_sarray(
        sarray_3d=numpy.ones(_RESOLUTION),
        uniform_domain_3d=_make_3d_uniform_domain(),
        field_name="density",
        latex_label=r"\rho",
    )


def _make_vfield_3d() -> field_models.VectorField_3D:
    return field_models.VectorField_3D.from_3d_varray(
        varray_3d=numpy.ones((3, ) + _RESOLUTION),
        uniform_domain_3d=_make_3d_uniform_domain(),
        field_name="velocity",
        latex_label=r"\vec{v}",
    )


##
## === TEST SUITES
##


class TestGetLabel(unittest.TestCase):

    def test_returns_latex_label(
        self,
    ):
        label = field_models.get_label(_make_sfield_3d())
        self.assertIsInstance(label, latex_labels.LatexLabel)

    def test_content_matches_field_latex_label(
        self,
    ):
        sfield_3d = _make_sfield_3d()
        label = field_models.get_label(sfield_3d)
        self.assertEqual(label.content, sfield_3d.latex_label)

    def test_get_label_wraps_content_in_dollars(
        self,
    ):
        label = field_models.get_label(_make_sfield_3d())
        self.assertEqual(label.label, r"$\rho$")


class TestGetVcompLabel(unittest.TestCase):

    def test_returns_latex_label(
        self,
    ):
        label = field_models.get_vcomp_label(
            vfield_3d=_make_vfield_3d(),
            comp_axis=cartesian_axes.CartesianAxis_3D.X0,
        )
        self.assertIsInstance(label, latex_labels.LatexLabel)

    def test_get_label_uses_bracket_subscript_notation(
        self,
    ):
        label = field_models.get_vcomp_label(
            vfield_3d=_make_vfield_3d(),
            comp_axis=cartesian_axes.CartesianAxis_3D.X0,
        )
        self.assertEqual(label.label, r"$\left[\vec{v}\right]_0$")


## } U-TEST
