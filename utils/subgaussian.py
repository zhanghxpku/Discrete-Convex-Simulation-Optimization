# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:20:49 2020

@author: haixiang_zhang

Properties of sub-Gaussian RVs
"""

import math


def confidence_interval(alpha, params, num_samples=None):
    """
    Returns the confidence interval of a subGaussian RV.
    """

    # Retrieve parameters
    sigma = params["sigma"] if "sigma" in params else 1
    # Check number of samples
    if num_samples is None:
        num_samples = 1

    return math.sqrt(2 * sigma / num_samples * math.log(2 / alpha))


def required_samples(alpha, CI, params):
    """
    Returns the number of samples needed to achieve alpha and CI
    """

    # Retrieve parameters
    sigma = params["sigma"] if "sigma" in params else 1

    return math.ceil(2 * sigma * math.log(2 / alpha) / (CI ** 2))
