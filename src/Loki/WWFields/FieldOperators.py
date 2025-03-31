## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from Loki.WWFields import FieldGradients


## ###############################################################
## FUNCTIONS
## ###############################################################
def compute_sfield_rms(sfield_q):
  sfield_q = numpy.array(sfield_q)
  return numpy.sqrt(numpy.mean(numpy.square(sfield_q)))

def compute_vfield_cross_product(vfield_q1, vfield_q2):
  return numpy.array([
    vfield_q1[1] * vfield_q2[2] - vfield_q1[2] * vfield_q2[1],
    vfield_q1[2] * vfield_q2[0] - vfield_q1[0] * vfield_q2[2],
    vfield_q1[0] * vfield_q2[1] - vfield_q1[1] * vfield_q2[0]
  ])

def compute_vfield_dot_product(vfield_q1, vfield_q2):
  return numpy.einsum("ixyz,ixyz->xyz", vfield_q1, vfield_q2)

def compute_vfield_magnitude(vfield_q):
  vfield_q = numpy.array(vfield_q)
  return numpy.sqrt(numpy.sum(vfield_q*vfield_q, axis=0))

def get_gradient_function(
    grad_order: int,
    verbose: bool = False
  ):
  implemented_grad_funcs = {
    2: (FieldGradients.gradient_2ocd, "2nd order centered-difference"),
    4: (FieldGradients.gradient_4ocd, "4th order centered-difference"),
    6: (FieldGradients.gradient_6ocd, "6th order centered-difference"),
  }
  if grad_order not in implemented_grad_funcs: raise ValueError(f"Gradient order `{grad_order}` is invalid.")
  grad_func, grad_label = implemented_grad_funcs[grad_order]
  if verbose: print(f"Computing gradient using {grad_label}...")
  return grad_func

def compute_vfield_curl(vfield_q, box_width=1.0, grad_order=2, verbose=False):
  grad_func = get_gradient_function(grad_order, verbose)
  ## input format: (vector-component, x, y, z), assuming cubic domain with uniform grid
  ## output format: (curl-component, x, y, z)
  vfield_q   = numpy.array(vfield_q)
  cell_width = box_width / vfield_q.shape[1]
  ## curl components
  return numpy.array([
    grad_func(vfield_q[2], cell_width, 1) - grad_func(vfield_q[1], cell_width, 2),
    grad_func(vfield_q[0], cell_width, 2) - grad_func(vfield_q[2], cell_width, 0),
    grad_func(vfield_q[1], cell_width, 0) - grad_func(vfield_q[0], cell_width, 1)
  ])

def compute_sfield_gradient(sfield_q, box_width=1.0, grad_order=2, verbose=False):
  grad_func = get_gradient_function(grad_order, verbose)
  ## input format: (x, y, z), assuming cubic domain with uniform grid
  ## output format: (gradient-direction, x, y, z)
  sfield_q = numpy.array(sfield_q)
  cell_width = box_width / sfield_q.shape[0]
  return numpy.array([
    grad_func(sfield_q, cell_width, gradient_dir)
    for gradient_dir in [0, 1, 2]
  ])

def compute_vfield_gradient(vfield_q, box_width=1.0, grad_order=2):
  ## df_i/dx_j: (component-i, gradient-direction-j, x, y, z)
  return numpy.array([
    compute_sfield_gradient(sfield_qi, box_width, grad_order)
    for sfield_qi in vfield_q
  ])

def compute_vfield_divergence(vfield_q, box_width=1.0, grad_order=2):
  r2tensor_grad_q = compute_vfield_gradient(vfield_q, box_width, grad_order)
  # return numpy.sum(numpy.array([
  #   r2tensor_grad_q[index_i,index_i,:,:,:]
  #   for index_i in range(3)
  # ]), axis=0)
  return numpy.einsum("iixyz->xyz", r2tensor_grad_q)


## END OF MODULE