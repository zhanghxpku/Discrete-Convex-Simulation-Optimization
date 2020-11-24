# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:58:20 2020

@author: haixiang_zhang

Dimension reduction method
"""

import math
import numpy as np
import scipy as sp
import time
from utils.lovasz import Lovasz, Round, SO
from utils.lll import LLL
# from utils.subgaussian import RequiredSamples

def RandomWalkSolver(F,params):
    """
    Cutting-plane method via random walk for multi-dim problems.
    """

    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    # sigma = params["sigma"] if "sigma" in params else 1
    eps = params["eps"] if "eps" in params else 1
    delta = params["delta"] if "delta" in params else 1e-6
    L = params["L"] if "L" in params else 1
    
    
    return 0

