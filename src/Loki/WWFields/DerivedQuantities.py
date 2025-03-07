## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from Loki.WWFields import FieldOperators
from Loki.Utils import Utils4Funcs


## ###############################################################
## FUNCTIONS
## ###############################################################
@Utils4Funcs.time_function
def computeTNBTerms(vfield_b, box_width=1.0, grad_order=2):
  ## format: (vector-component, x, y, z)
  vfield_b = numpy.array(vfield_b)
  ## ---- COMPUTE TANGENT BASIS
  ## (f_k f_k)^(1/2)
  sfield_magn_b = FieldOperators.vfieldMagnitude(vfield_b)
  ## f_i / (f_k f_k)^(1/2)
  vbasis_tangent = vfield_b * sfield_magn_b**(-1)
  ## ---- COMPUTE NORMAL BASIS
  ## df_j/dx_i: (component-j, gradient-direction-i, x, y, z)
  r2tensor_grad_b = FieldOperators.vfieldGradient(vfield_b, box_width, grad_order)
  ## f_i df_j/dx_i
  vbasis_normal_term1 = numpy.einsum("ixyz,jixyz->jxyz", vfield_b, r2tensor_grad_b)
  ## f_i f_j f_m df_m/dx_i
  vbasis_normal_term2 = numpy.einsum("ixyz,jxyz,mxyz,mixyz->jxyz", vfield_b, vfield_b, vfield_b, r2tensor_grad_b)
  ## (f_i df_j/dx_i) / (f_k f_k) - (f_i f_j f_m df_m/dx_i) / (f_k f_k)^2
  vfield_kappa = vbasis_normal_term1 * sfield_magn_b**(-2) - vbasis_normal_term2 * sfield_magn_b**(-4)
  ## clean up temporary quantities
  del vbasis_normal_term1, vbasis_normal_term2
  ## field curvature
  sfield_curvature = FieldOperators.vfieldMagnitude(vfield_kappa)
  ## normal basis
  vbasis_normal = vfield_kappa / sfield_curvature
  ## ---- COMPUTE BINORMAL BASIS
  ## by definition it is orthogonal to both t- and n-basis
  vbasis_binormal = FieldOperators.vfieldCrossProduct(vbasis_tangent, vbasis_normal)
  return vbasis_tangent, vbasis_normal, vbasis_binormal, sfield_curvature

@Utils4Funcs.time_function
def computeCurvatureTerms(vbasis_normal, vbasis_tangent, vfield_u, box_width=1.0, grad_order=2):
  ## du_j/dx_i: (component-j, gradient-direction-i, x, y, z)
  r2tensor_grad_u = FieldOperators.vfieldGradient(vfield_u, box_width, grad_order)
  ## n_i n_j du_j/dx_i
  sfield_curvature = numpy.einsum("ixyz,jxyz,jixyz->xyz", vbasis_normal, vbasis_normal, r2tensor_grad_u)
  ## t_i t_j du_j/dx_i
  sfield_stretching = numpy.einsum("ixyz,jxyz,jixyz->xyz", vbasis_tangent, vbasis_tangent, r2tensor_grad_u)
  ## du_i/dx_i
  sfield_compression = numpy.einsum("iixyz->xyz", r2tensor_grad_u)
  return sfield_curvature, sfield_stretching, sfield_compression

@Utils4Funcs.time_function
def computeLorentzForce(vfield_b, box_width=1.0, grad_order=2):
  vfield_b = numpy.array(vfield_b)
  vbasis_tangent, vbasis_normal, _, sfield_kappa = computeTNBTerms(vfield_b, box_width, grad_order)
  sfield_sq_magn_b           = FieldOperators.vfieldMagnitude(vfield_b)**2
  vfield_tot_grad_pressure   = 0.5 * FieldOperators.sfieldGradient(sfield_sq_magn_b, box_width, grad_order)
  vfield_align_grad_pressure = numpy.einsum("ixyz,jxyz,jxyz->ixyz", vbasis_tangent, vbasis_tangent, vfield_tot_grad_pressure)
  vfield_tension_force       = sfield_sq_magn_b * sfield_kappa * vbasis_normal
  vfield_ortho_grad_pressure = vfield_tot_grad_pressure - vfield_align_grad_pressure
  vfield_lorentz_force       = vfield_tension_force - vfield_ortho_grad_pressure
  del vbasis_tangent, vbasis_normal, sfield_kappa, sfield_sq_magn_b, vfield_tot_grad_pressure, vfield_align_grad_pressure
  return vfield_lorentz_force, vfield_tension_force, vfield_ortho_grad_pressure

def computeDissipationFunction(vfield_u):
  r2tensor_gradj_ui = FieldOperators.vfieldGradient(vfield_u)
  sfield_div_u = numpy.einsum("iixyz->xyz", r2tensor_gradj_ui)
  r2tensor_bulk = 1/3 * numpy.einsum("xyz,ij->ijxyz", sfield_div_u, numpy.identity(3))
  ## S_ij = 0.5 ( \partial_i f_j + \partial_j f_i ) - 1/3 \delta_{ij} \partial_k f_k
  r2tensor_srt = 0.5 * (r2tensor_gradj_ui.transpose(1, 0, 2, 3, 4) + r2tensor_gradj_ui) - r2tensor_bulk
  ## \partial_j S_ij
  vfield_df = numpy.array([
    numpy.sum(FieldOperators.vfieldGradient(r2tensor_srt[:,0,:,:,:])[0], axis=0),
    numpy.sum(FieldOperators.vfieldGradient(r2tensor_srt[:,1,:,:,:])[1], axis=0),
    numpy.sum(FieldOperators.vfieldGradient(r2tensor_srt[:,2,:,:,:])[2], axis=0),
  ])
  del vfield_u, r2tensor_gradj_ui, sfield_div_u, r2tensor_bulk, r2tensor_srt, vfield_df
  return vfield_df


## END OF MODULE