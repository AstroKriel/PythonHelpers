## { MODULE

##
## === DEPENDENCIES
##

import numpy

from scipy.special import erfinv as scipy_erfinv

from jormi.ww_types import type_manager

##
## === FUNCTIONS
##


def sample_gaussian_distribution_from_quantiles(
    q1: float,
    q2: float,
    p1: float,
    p2: float,
    num_samples=10**3,
) -> numpy.ndarray:
    """
    Sample a normal distribution where the quantile-levels 0 < q1 < q2 < 1 corresponds with
    probability-values 0 < p1 < p2 < 1.
    """
    type_manager.ensure_type(
        param=q1,
        valid_types=(float, int),
    )
    type_manager.ensure_type(
        param=q2,
        valid_types=(float, int),
    )
    type_manager.ensure_type(
        param=p1,
        valid_types=(float, int),
    )
    type_manager.ensure_type(
        param=p2,
        valid_types=(float, int),
    )
    type_manager.ensure_type(
        param=num_samples,
        valid_types=int,
    )
    if not (0.0 < q1 < q2 < 1.0):
        raise ValueError("`q1` and `q2` must satisfy 0 < q1 < q2 < 1.")
    if not (p1 < p2):
        raise ValueError("`p1` must be strictly less than `p2`.")
    if num_samples <= 0:
        raise ValueError("`num_samples` must be a positive integer.")
    ## inverse CDF
    cdf_inv_p1 = numpy.sqrt(2) * scipy_erfinv(2 * q1 - 1)
    cdf_inv_p2 = numpy.sqrt(2) * scipy_erfinv(2 * q2 - 1)
    ## solve for the mean and standard deviation of the normal distribution
    mean_value = ((p1 * cdf_inv_p2) - (p2 * cdf_inv_p1)) / (cdf_inv_p2 - cdf_inv_p1)
    std_value = (p2 - p1) / (cdf_inv_p2 - cdf_inv_p1)
    ## generate sampled points from the normal distribution
    samples = mean_value + std_value * numpy.random.randn(num_samples)
    return samples


## } MODULE
