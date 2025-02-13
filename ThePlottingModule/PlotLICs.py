## This file is part of the "line-integral-convolutions" project.
## Copyright (c) 2024 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.


## ###############################################################
## IMPORT MODULES
## ###############################################################
import numpy as np

from numba import njit, prange
from scipy import ndimage
from skimage.exposure import equalize_adapthist

from TheUsefulModule import WWFuncs


## ###############################################################
## LIC IMPLEMENTATION
## ###############################################################
def filterHighpass(
  sfield: np.ndarray,
  sigma:  float = 3.0,
):
  lowpass = ndimage.gaussian_filter(sfield, sigma)
  gauss_highpass = sfield - lowpass
  return gauss_highpass


def rescaledEqualize(
  sfield:                  np.ndarray,
  num_subregions_rows:     int   = 8,
  num_subregions_cols:     int   = 8,
  clip_intensity_gradient: float = 0.01,
  num_intensity_bins:      int   = 150,
):
  ## rescale values to enhance local contrast
  ## note, output values are bound by [0, 1]
  sfield = equalize_adapthist(
    image       = sfield,
    kernel_size = (num_subregions_rows, num_subregions_cols),
    clip_limit  = clip_intensity_gradient,
    nbins       = num_intensity_bins,
  )
  return sfield

@njit
def taperPixelContribution(
  streamlength: int,
  step_index:   int,
) -> float:
  return 0.5 * (1 + np.cos(np.pi * step_index / streamlength))

@njit
def interpolateBilinear(
  vfield: np.ndarray,
  row:    float,
  col:    float,
) -> tuple[float, float]:
  row_low  = int(np.floor(row))
  col_low  = int(np.floor(col))
  row_high = min(row_low + 1, vfield.shape[1] - 1)
  col_high = min(col_low + 1, vfield.shape[2] - 1)
  ## weight based on distance from pixel edge
  weight_row_high = row - row_low
  weight_col_high = col - col_low
  weight_row_low  = 1 - weight_row_high
  weight_col_low  = 1 - weight_col_high
  interpolated_vfield_comp_col = (
      vfield[0, row_low, col_low]   * weight_row_low  * weight_col_low
    + vfield[0, row_low, col_high]  * weight_row_low  * weight_col_high
    + vfield[0, row_high, col_low]  * weight_row_high * weight_col_low
    + vfield[0, row_high, col_high] * weight_row_high * weight_col_high
  )
  interpolated_vfield_comp_row = (
      vfield[1, row_low, col_low]   * weight_row_low  * weight_col_low
    + vfield[1, row_low, col_high]  * weight_row_low  * weight_col_high
    + vfield[1, row_high, col_low]  * weight_row_high * weight_col_low
    + vfield[1, row_high, col_high] * weight_row_high * weight_col_high
  )
  ## remember (x,y) -> (col, row)
  return interpolated_vfield_comp_col, interpolated_vfield_comp_row

@njit
def advectStreamline(
  vfield:            np.ndarray,
  sfield_in:         np.ndarray,
  start_row:         int,
  start_col:         int,
  dir_sgn:           int,
  streamlength:      int,
  bool_periodic_BCs: bool,
) -> tuple[float, float]:
  weighted_sum = 0.0
  total_weight = 0.0
  row_float, col_float = start_row, start_col
  num_rows, num_cols = vfield.shape[1], vfield.shape[2]
  for step in range(streamlength):
    row_int = int(np.floor(row_float))
    col_int = int(np.floor(col_float))
    # ## nearest neighbor interpolation
    # vfield_comp_col = dir_sgn * vfield[0, row_int, col_int]  # x
    # vfield_comp_row = dir_sgn * vfield[1, row_int, col_int]  # y
    ## bilinear interpolation (negligble performance hit compared to nearest neighbor)
    vfield_comp_col, vfield_comp_row = interpolateBilinear(
      vfield = vfield,
      row    = row_float,
      col    = col_float,
    )
    vfield_comp_col *= dir_sgn
    vfield_comp_row *= dir_sgn
    ## skip if the field magnitude is zero: advection has halted
    if abs(vfield_comp_row) == 0.0 and abs(vfield_comp_col) == 0.0: break
    ## compute how long the streamline advects before it leaves the current cell region (divided by cell-centers)
    if   vfield_comp_row > 0.0: delta_time_row = (np.floor(row_float) + 1 - row_float) / vfield_comp_row
    elif vfield_comp_row < 0.0: delta_time_row = (np.ceil(row_float) - 1 - row_float) / vfield_comp_row
    else:                       delta_time_row = np.inf
    if   vfield_comp_col > 0.0: delta_time_col = (np.floor(col_float) + 1 - col_float) / vfield_comp_col
    elif vfield_comp_col < 0.0: delta_time_col = (np.ceil(col_float) - 1 - col_float) / vfield_comp_col
    else:                       delta_time_col = np.inf
    ## equivelant to a CFL condition
    time_step = min(delta_time_col, delta_time_row)
    ## advect the streamline to the next cell region
    col_float += vfield_comp_col * time_step
    row_float += vfield_comp_row * time_step
    if bool_periodic_BCs:
      row_float = (row_float + num_rows) % num_rows
      col_float = (col_float + num_cols) % num_cols
    else:
      ## open boundaries: terminate if streamline leaves the domain
      if not ((0 <= row_float < num_rows) and (0 <= col_float < num_cols)):
        break
    ## weight the contribution of the current pixel based on its distance from the start of the streamline
    contribution_weight = taperPixelContribution(streamlength, step)
    weighted_sum += contribution_weight * sfield_in[row_int, col_int]
    total_weight += contribution_weight
  return weighted_sum, total_weight

@njit(parallel=True)
def _computeLIC(
  vfield:            np.ndarray,
  sfield_in:         np.ndarray,
  sfield_out:        np.ndarray,
  streamlength:      int,
  num_rows:          int,
  num_cols:          int,
  bool_periodic_BCs: bool,
) -> np.ndarray:
  for row in prange(num_rows):
    for col in range(num_cols):
      forward_sum, forward_total = advectStreamline(
        vfield            = vfield,
        sfield_in         = sfield_in,
        start_row         = row,
        start_col         = col,
        dir_sgn           = +1,
        streamlength      = streamlength,
        bool_periodic_BCs = bool_periodic_BCs,
      )
      backward_sum, backward_total = advectStreamline(
        vfield            = vfield,
        sfield_in         = sfield_in,
        start_row         = row,
        start_col         = col,
        dir_sgn           = -1,
        streamlength      = streamlength,
        bool_periodic_BCs = bool_periodic_BCs,
      )
      total_sum    = forward_sum   + backward_sum
      total_weight = forward_total + backward_total
      if total_weight > 0.0: sfield_out[row, col] = total_sum / total_weight
      else:                  sfield_out[row, col] = 0.0
  return sfield_out

@WWFuncs.timeFunc
def computeLIC(
    vfield:            np.ndarray,
    sfield_in:         np.ndarray = None,
    streamlength:      int        = None,
    seed_sfield:       int        = 42,
    bool_periodic_BCs: bool       = True,
) -> np.ndarray:
  assert vfield.ndim == 3, f"vfield must have 3 dimensions, but got {vfield.ndim}."
  num_vcomps, num_rows, num_cols = vfield.shape
  assert (
    num_vcomps == 2
  ), f"vfield must have 2 components (in the first dimension), but got {num_vcomps}."
  sfield_out = np.zeros((num_rows, num_cols), dtype=np.float32)
  if sfield_in is None:
    if seed_sfield is not None:
      np.random.seed(seed_sfield)
    sfield_in = np.random.rand(num_rows, num_cols).astype(np.float32)
  else:
    assert sfield_in.shape == (num_rows, num_cols), (
      f"sfield_in must have dimensions ({num_rows}, {num_cols}), "
      f"but received it with dimensions {sfield_in.shape}."
    )
  if streamlength is None:
    streamlength = min(num_rows, num_cols) // 4
  return _computeLIC(
    vfield            = vfield,
    sfield_in         = sfield_in,
    sfield_out        = sfield_out,
    streamlength      = streamlength,
    num_rows          = num_rows,
    num_cols          = num_cols,
    bool_periodic_BCs = bool_periodic_BCs,
  )

def computeLIC_postprocessing(
  vfield:            np.ndarray,
  sfield_in:         np.ndarray = None,
  streamlength:      int        = None,
  seed_sfield:       int        = 42,
  bool_periodic_BCs: bool       = True,
  num_iterations:    int        = 3,
  num_repetitions:   int        = 3,
  bool_filter:       bool       = True,
  filter_sigma:      float      = 3.0,
  bool_equalize:     bool       = True,
) -> np.ndarray:
  for _ in range(num_repetitions):
    for _ in range(num_iterations):
      sfield = computeLIC(
        vfield            = vfield,
        sfield_in         = sfield_in,
        streamlength      = streamlength,
        seed_sfield       = seed_sfield,
        bool_periodic_BCs = bool_periodic_BCs,
      )
      sfield_in = sfield
    if bool_filter: sfield = filterHighpass(sfield, sigma=filter_sigma)
  if bool_equalize: sfield = rescaledEqualize(sfield)
  return sfield


## END OF LIC IMPLEMENTATION
