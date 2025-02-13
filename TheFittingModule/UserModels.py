## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import numpy as np


## ###############################################################
## CLASS OF USEFUL FUNCTIONS
## ###############################################################
class ListOfModels():
  def constant(x, a0):
    return a0 + 0.0*np.array(x)

  def linear(x, a0):
    """ linear function in linear-domain:
      y = a0 * x
    """
    return a0 * np.array(x)

  def linear_offset(x, a0, a1):
    """ linear function with offset in linear-domain:
      y = a1 + a0 * x
    """
    return a0 + a1 * np.array(x)

  def powerlaw_linear(x, a0, a1, a2):
    """ power-law in linear-domain:
      y = a0 + a1 * k^a2
    """
    return a0 + a1 * np.array(x)**a2

  # def powerlaw_log10(log10_x, log10_a0, a1):
  #   """ power-law in log10-domain:
  #     log10(y) = log10_a0 + a1 * log10(k)
  #   """
  #   return log10_a0 + a1 * np.array(log10_x)

  def exp_linear(x, a0, a1):
    """ exponential in linear-domain:
      y = a0 * exp(a1 * x)
    """
    return a0 * np.exp(a1 * np.array(x))

  def exp_loge(x, loge_a0, a1):
    """ exponential in log(e)-domain:
      l(y) = ln(a0) + a1 * x
    """
    return loge_a0 + a1 * np.array(x)

  def gaussian(x, a, mu, std):
    return a * np.exp( - (np.array(x) - mu)**2 / (2*std ** 2))

  def bimodal(x, a0, mu0, std0, a1, mu1, std1):
    return ListOfModels.gaussian(x, a0, mu0, std0) + ListOfModels.gaussian(x, a1, mu1, std1)

  def logistic_growth_increasing(x, a0, a1, a2):
    """ logistic model (increasing)
    """
    return a0 * (1 - np.exp( -(np.array(x) / a1)**a2 ))

  def logistic_growth_decreasing(x, a0, a1, a2):
    """ logistic model (decreasing)
    """
    return a0 / (1 - np.exp( -(np.array(x) / a1)**a2 ))


## END OF LIBRARY