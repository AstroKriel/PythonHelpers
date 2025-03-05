## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy


## ###############################################################
## FUNCTIONS
## ###############################################################
def gradient_2ocd(sfield_q, cell_width, gradient_dir):
  F = -1 # shift forwards
  B = +1 # shift backwards
  q_p = numpy.roll(sfield_q, int(1*F), axis=gradient_dir)
  q_m = numpy.roll(sfield_q, int(1*B), axis=gradient_dir)
  return (q_p - q_m) / (2*cell_width)

def gradient_4ocd(sfield_q, cell_width, gradient_dir):
  F = -1 # shift forwards
  B = +1 # shift backwards
  q_p1 = numpy.roll(sfield_q, int(1*F), axis=gradient_dir)
  q_p2 = numpy.roll(sfield_q, int(2*F), axis=gradient_dir)
  q_m1 = numpy.roll(sfield_q, int(1*B), axis=gradient_dir)
  q_m2 = numpy.roll(sfield_q, int(2*B), axis=gradient_dir)
  return (-1.0*q_p2 + 8.0*q_p1 - 8.0*q_m1 + 1.0*q_m2) / (12.0*cell_width)

def gradient_6ocd(sfield_q, cell_width, gradient_dir):
  F = -1 # shift forwards
  B = +1 # shift backwards
  q_p1 = numpy.roll(sfield_q, int(1*F), axis=gradient_dir)
  q_p2 = numpy.roll(sfield_q, int(2*F), axis=gradient_dir)
  q_p3 = numpy.roll(sfield_q, int(3*F), axis=gradient_dir)
  q_m1 = numpy.roll(sfield_q, int(1*B), axis=gradient_dir)
  q_m2 = numpy.roll(sfield_q, int(2*B), axis=gradient_dir)
  q_m3 = numpy.roll(sfield_q, int(3*B), axis=gradient_dir)
  return (1.0*q_p3 - 9.0*q_p2 + 45.0*q_p1 - 45.0*q_m1 + 9.0*q_m2 - 1.0*q_m3) / (60.0*cell_width)


## END OF MODULE