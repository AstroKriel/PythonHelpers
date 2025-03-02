## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import numpy as np
import statsmodels.api as sm

## load user defined modules
from Loki.TheUsefulModule import WWFuncs


## ###############################################################
## FUNCTIONS
## ###############################################################
@WWFuncs.time_function
def fitLineToMasked2DJPDF(bedges_cols, bedges_rows, jpdf, level):
  jpdf = np.array(jpdf)
  mg_rows, mg_cols = np.meshgrid(bedges_rows, bedges_cols, indexing="ij")
  mask = jpdf > level
  fit_obj = sm.OLS(mg_cols[1:,1:][mask], sm.add_constant(mg_rows[1:,1:][mask]))
  results = fit_obj.fit()
  intercept, slope = results.params
  return intercept, slope


## END OF LIBRARY