## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import numpy as np

from Loki.TheUsefulModule import WWFuncs


## ###############################################################
## FUNCTIONS
## ###############################################################
def genGaussianRandomField(size: int, correlation_length: float, num_dims:int =3):
  if num_dims not in [2, 3]: raise ValueError("`num_dims` must be either `2` or `3`.")
  ## generate white noise in Fourier space
  white_noise = np.random.normal(0, 1, (size,)*num_dims)
  ## create a grid of frequencies
  array_k   = np.fft.fftfreq(size)
  mg_k_vect = np.meshgrid(*(array_k for _ in range(num_dims)), indexing="ij")
  ## compute the magnitude of the wave vector
  mg_k_magnitude = np.sqrt(np.sum(mg_k_comp**2 for mg_k_comp in mg_k_vect))
  ## create a Gaussian filter in Fourier space
  filter_fft = np.exp(-0.5 * (mg_k_magnitude * correlation_length)**2)
  ## apply the filter to the noise in Fourier space
  sfield_fft = filter_fft * np.fft.fftn(white_noise)
  ## transform back to real space
  sfield = np.real(np.fft.ifftn(sfield_fft))
  return sfield

def vfieldCrossProduct(vfield_q1, vfield_q2):
  return np.array([
    vfield_q1[1] * vfield_q2[2] - vfield_q1[2] * vfield_q2[1],
    vfield_q1[2] * vfield_q2[0] - vfield_q1[0] * vfield_q2[2],
    vfield_q1[0] * vfield_q2[1] - vfield_q1[1] * vfield_q2[0]
  ])

def vfieldDotProduct(vfield_q1, vfield_q2):
  return np.einsum("ixyz,ixyz->xyz", vfield_q1, vfield_q2)

def vfieldMagnitude(vfield_q):
  vfield_q = np.array(vfield_q)
  return np.sqrt(np.sum(vfield_q*vfield_q, axis=0))

def gradient_2ocd(sfield_q, cell_width, gradient_dir):
  F = -1 # shift forwards
  B = +1 # shift backwards
  q_p = np.roll(sfield_q, int(1*F), axis=gradient_dir)
  q_m = np.roll(sfield_q, int(1*B), axis=gradient_dir)
  return (q_p - q_m) / (2*cell_width)

def gradient_4ocd(sfield_q, cell_width, gradient_dir):
  F = -1 # shift forwards
  B = +1 # shift backwards
  q_p1 = np.roll(sfield_q, int(1*F), axis=gradient_dir)
  q_p2 = np.roll(sfield_q, int(2*F), axis=gradient_dir)
  q_m1 = np.roll(sfield_q, int(1*B), axis=gradient_dir)
  q_m2 = np.roll(sfield_q, int(2*B), axis=gradient_dir)
  return (-1.0*q_p2 + 8.0*q_p1 - 8.0*q_m1 + 1.0*q_m2) / (12.0*cell_width)

def gradient_6ocd(sfield_q, cell_width, gradient_dir):
  F = -1 # shift forwards
  B = +1 # shift backwards
  q_p1 = np.roll(sfield_q, int(1*F), axis=gradient_dir)
  q_p2 = np.roll(sfield_q, int(2*F), axis=gradient_dir)
  q_p3 = np.roll(sfield_q, int(3*F), axis=gradient_dir)
  q_m1 = np.roll(sfield_q, int(1*B), axis=gradient_dir)
  q_m2 = np.roll(sfield_q, int(2*B), axis=gradient_dir)
  q_m3 = np.roll(sfield_q, int(3*B), axis=gradient_dir)
  return (1.0*q_p3 - 9.0*q_p2 + 45.0*q_p1 - 45.0*q_m1 + 9.0*q_m2 - 1.0*q_m3) / (60.0*cell_width)

def sfieldRMS(sfield_q):
  sfield_q = np.array(sfield_q)
  return np.sqrt(np.mean(sfield_q**2))

def vfieldCurl(vfield_q, box_width=1.0, grad_order=2, bool_verbose=False):
  if   int(grad_order) == int(2): func_gradient = gradient_2ocd; str_order = "2nd"
  elif int(grad_order) == int(4): func_gradient = gradient_4ocd; str_order = "4th"
  elif int(grad_order) == int(6): func_gradient = gradient_6ocd; str_order = "6th"
  if bool_verbose: print(f"Computing {str_order} gradient of scalar field...")
  ## input format: (vector-component, x, y, z), assuming cubic domain with uniform grid
  ## output format: (curl-component, x, y, z)
  vfield_q = np.array(vfield_q)
  cell_width = box_width / vfield_q.shape[1]
  vfield_curl_q = np.zeros_like(vfield_q)
  ## curl components
  return np.array([
    func_gradient(vfield_q[2], cell_width, 1) - func_gradient(vfield_q[1], cell_width, 2),
    func_gradient(vfield_q[0], cell_width, 2) - func_gradient(vfield_q[2], cell_width, 0),
    func_gradient(vfield_q[1], cell_width, 0) - func_gradient(vfield_q[0], cell_width, 1)
  ])

def sfieldGradient(sfield_q, box_width=1.0, grad_order=2, bool_verbose=False):
  if   int(grad_order) == int(2): func_gradient = gradient_2ocd; str_order = "2nd"
  elif int(grad_order) == int(4): func_gradient = gradient_4ocd; str_order = "4th"
  elif int(grad_order) == int(6): func_gradient = gradient_6ocd; str_order = "6th"
  if bool_verbose: print(f"Computing {str_order} gradient of scalar field...")
  ## input format: (x, y, z), assuming cubic domain with uniform grid
  ## output format: (gradient-direction, x, y, z)
  sfield_q = np.array(sfield_q)
  cell_width = box_width / sfield_q.shape[0]
  return np.array([
    func_gradient(sfield_q, cell_width, gradient_dir)
    for gradient_dir in [0, 1, 2]
  ])

def vfieldGradient(vfield_q, box_width=1.0, grad_order=2):
  ## df_i/dx_j: (component-i, gradient-direction-j, x, y, z)
  return np.array([
    sfieldGradient(sfield_qi, box_width, grad_order)
    for sfield_qi in vfield_q
  ])

def vfieldDivergence(vfield_q, box_width=1.0, grad_order=2):
  r2tensor_grad_q = vfieldGradient(vfield_q, box_width, grad_order)
  # return np.sum(np.array([
  #   r2tensor_grad_q[index_i,index_i,:,:,:]
  #   for index_i in range(3)
  # ]), axis=0)
  return np.einsum("iixyz->xyz", r2tensor_grad_q)

@WWFuncs.time_function
def vfieldTNB(vfield_b, box_width=1.0, grad_order=2):
  ## format: (vector-component, x, y, z)
  vfield_b = np.array(vfield_b)
  ## ---- COMPUTE TANGENT BASIS
  ## (f_k f_k)^(1/2)
  sfield_magn_b = vfieldMagnitude(vfield_b)
  ## f_i / (f_k f_k)^(1/2)
  vbasis_tangent = vfield_b * sfield_magn_b**(-1)
  ## ---- COMPUTE NORMAL BASIS
  ## df_j/dx_i: (component-j, gradient-direction-i, x, y, z)
  r2tensor_grad_b = vfieldGradient(vfield_b, box_width, grad_order)
  ## f_i df_j/dx_i
  vbasis_normal_term1 = np.einsum("ixyz,jixyz->jxyz", vfield_b, r2tensor_grad_b)
  ## f_i f_j f_m df_m/dx_i
  vbasis_normal_term2 = np.einsum("ixyz,jxyz,mxyz,mixyz->jxyz", vfield_b, vfield_b, vfield_b, r2tensor_grad_b)
  ## (f_i df_j/dx_i) / (f_k f_k) - (f_i f_j f_m df_m/dx_i) / (f_k f_k)^2
  vfield_kappa = vbasis_normal_term1 * sfield_magn_b**(-2) - vbasis_normal_term2 * sfield_magn_b**(-4)
  ## clean up temporary quantities
  del vbasis_normal_term1, vbasis_normal_term2
  ## field curvature
  sfield_curvature = vfieldMagnitude(vfield_kappa)
  ## normal basis
  vbasis_normal = vfield_kappa / sfield_curvature
  ## ---- COMPUTE BINORMAL BASIS
  ## by definition it is orthogonal to both t- and n-basis
  vbasis_binormal = vfieldCrossProduct(vbasis_tangent, vbasis_normal)
  return vbasis_tangent, vbasis_normal, vbasis_binormal, sfield_curvature

@WWFuncs.time_function
def computeCurvatureTerms(vbasis_normal, vbasis_tangent, vfield_u, box_width=1.0, grad_order=2):
  ## du_j/dx_i: (component-j, gradient-direction-i, x, y, z)
  r2tensor_grad_u = vfieldGradient(vfield_u, box_width, grad_order)
  ## n_i n_j du_j/dx_i
  sfield_curvature = np.einsum("ixyz,jxyz,jixyz->xyz", vbasis_normal, vbasis_normal, r2tensor_grad_u)
  ## t_i t_j du_j/dx_i
  sfield_stretching = np.einsum("ixyz,jxyz,jixyz->xyz", vbasis_tangent, vbasis_tangent, r2tensor_grad_u)
  ## du_i/dx_i
  sfield_compression = np.einsum("iixyz->xyz", r2tensor_grad_u)
  return sfield_curvature, sfield_stretching, sfield_compression

@WWFuncs.time_function
def computeLorentzForce(vfield_b, box_width=1.0, grad_order=2):
  vfield_b = np.array(vfield_b)
  vbasis_tangent, vbasis_normal, _, sfield_kappa = vfieldTNB(vfield_b, box_width, grad_order)
  sfield_sq_magn_b           = vfieldMagnitude(vfield_b)**2
  vfield_tot_grad_pressure   = 0.5 * sfieldGradient(sfield_sq_magn_b, box_width, grad_order)
  vfield_align_grad_pressure = np.einsum("ixyz,jxyz,jxyz->ixyz", vbasis_tangent, vbasis_tangent, vfield_tot_grad_pressure)
  vfield_tension_force       = sfield_sq_magn_b * sfield_kappa * vbasis_normal
  vfield_ortho_grad_pressure = vfield_tot_grad_pressure - vfield_align_grad_pressure
  vfield_lorentz_force       = vfield_tension_force - vfield_ortho_grad_pressure
  del vbasis_tangent, vbasis_normal, sfield_kappa, sfield_sq_magn_b, vfield_tot_grad_pressure, vfield_align_grad_pressure
  return vfield_lorentz_force, vfield_tension_force, vfield_ortho_grad_pressure


## END OF LIBRARY