## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from Loki.WWFields import FieldGradients


## ###############################################################
## FUNCTIONS
## ###############################################################
def sfieldRMS(sfield_q):
  sfield_q = numpy.array(sfield_q)
  return numpy.sqrt(numpy.mean(sfield_q**2))

def vfieldCrossProduct(vfield_q1, vfield_q2):
  return numpy.array([
    vfield_q1[1] * vfield_q2[2] - vfield_q1[2] * vfield_q2[1],
    vfield_q1[2] * vfield_q2[0] - vfield_q1[0] * vfield_q2[2],
    vfield_q1[0] * vfield_q2[1] - vfield_q1[1] * vfield_q2[0]
  ])

def vfieldDotProduct(vfield_q1, vfield_q2):
  return numpy.einsum("ixyz,ixyz->xyz", vfield_q1, vfield_q2)

def vfieldMagnitude(vfield_q):
  vfield_q = numpy.array(vfield_q)
  return numpy.sqrt(numpy.sum(vfield_q*vfield_q, axis=0))

def vfieldCurl(vfield_q, box_width=1.0, grad_order=2, bool_verbose=False):
  if   int(grad_order) == int(2): func_gradient = FieldGradients.gradient_2ocd; str_order = "2nd"
  elif int(grad_order) == int(4): func_gradient = FieldGradients.gradient_4ocd; str_order = "4th"
  elif int(grad_order) == int(6): func_gradient = FieldGradients.gradient_6ocd; str_order = "6th"
  if bool_verbose: print(f"Computing {str_order} gradient of scalar field...")
  ## input format: (vector-component, x, y, z), assuming cubic domain with uniform grid
  ## output format: (curl-component, x, y, z)
  vfield_q   = numpy.array(vfield_q)
  cell_width = box_width / vfield_q.shape[1]
  ## curl components
  return numpy.array([
    func_gradient(vfield_q[2], cell_width, 1) - func_gradient(vfield_q[1], cell_width, 2),
    func_gradient(vfield_q[0], cell_width, 2) - func_gradient(vfield_q[2], cell_width, 0),
    func_gradient(vfield_q[1], cell_width, 0) - func_gradient(vfield_q[0], cell_width, 1)
  ])

def sfieldGradient(sfield_q, box_width=1.0, grad_order=2, bool_verbose=False):
  if   int(grad_order) == int(2): func_gradient = FieldGradients.gradient_2ocd; str_order = "2nd"
  elif int(grad_order) == int(4): func_gradient = FieldGradients.gradient_4ocd; str_order = "4th"
  elif int(grad_order) == int(6): func_gradient = FieldGradients.gradient_6ocd; str_order = "6th"
  if bool_verbose: print(f"Computing {str_order} gradient of scalar field...")
  ## input format: (x, y, z), assuming cubic domain with uniform grid
  ## output format: (gradient-direction, x, y, z)
  sfield_q = numpy.array(sfield_q)
  cell_width = box_width / sfield_q.shape[0]
  return numpy.array([
    func_gradient(sfield_q, cell_width, gradient_dir)
    for gradient_dir in [0, 1, 2]
  ])

def vfieldGradient(vfield_q, box_width=1.0, grad_order=2):
  ## df_i/dx_j: (component-i, gradient-direction-j, x, y, z)
  return numpy.array([
    sfieldGradient(sfield_qi, box_width, grad_order)
    for sfield_qi in vfield_q
  ])

def vfieldDivergence(vfield_q, box_width=1.0, grad_order=2):
  r2tensor_grad_q = vfieldGradient(vfield_q, box_width, grad_order)
  # return numpy.sum(numpy.array([
  #   r2tensor_grad_q[index_i,index_i,:,:,:]
  #   for index_i in range(3)
  # ]), axis=0)
  return numpy.einsum("iixyz->xyz", r2tensor_grad_q)


## END OF MODULE