## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import statsmodels.api
from Loki.Utils import Utils4Funcs


## ###############################################################
## FUNCTIONS
## ###############################################################
@Utils4Funcs.time_function
def fitLineToMasked2DJPDF(bedges_cols, bedges_rows, jpdf, level):
  jpdf = numpy.array(jpdf)
  mg_rows, mg_cols = numpy.meshgrid(bedges_rows, bedges_cols, indexing="ij")
  mask = jpdf > level
  fit_obj = statsmodels.api.OLS(mg_cols[1:,1:][mask], statsmodels.api.add_constant(mg_rows[1:,1:][mask]))
  results = fit_obj.fit()
  intercept, slope = results.params
  return intercept, slope


## END OF MODULE