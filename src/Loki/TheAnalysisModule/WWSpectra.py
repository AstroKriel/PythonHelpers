## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import numpy as np


## ###############################################################
## FUNCTIONS
## ###############################################################
def normaliseSpectrum(spectrum):
  return np.array(spectrum) / np.sum(spectrum)

def getAverageNormalisedSpectrum(spectra_group_t):
  return np.mean([
    normaliseSpectrum(spectrum)
    for spectrum in spectra_group_t
  ], axis=0)


## END OF LIBRARY