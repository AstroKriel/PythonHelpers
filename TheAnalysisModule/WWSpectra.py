## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import numpy as np


## ###############################################################
## FUNCTIONS
## ###############################################################
def normSpectra(list_power):
  return np.array(list_power) / np.sum(list_power)

def normSpectra_grouped(spectra_group_t):
  return [
    normSpectra(list_power)
    for list_power in spectra_group_t
  ]

def aveSpectra(spectra_group_t, bool_norm=True):
  list_power_norm_group_t = [
    normSpectra(list_power) if bool_norm
    else list_power
    for list_power in spectra_group_t
  ]
  return np.mean(list_power_norm_group_t, axis=0)


## END OF LIBRARY