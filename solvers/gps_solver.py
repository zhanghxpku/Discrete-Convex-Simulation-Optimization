# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 20:43:56 2020

@author: haixiang_zhang

Vaidya's cutting-plane method
"""

import math
import numpy as np
import time
from utils.lovasz import Lovasz, Round, SO, LovaszCons, RoundCons, SOCons
from utils.subgaussian import RequiredSamples

def GPSSolver(F,params):
    """
    Gaussian process-based search method for multi-dim problems.
    """

    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    sigma = params["sigma"] if "sigma" in params else 1
    eps = params["eps"] if "eps" in params else 1
    delta = params["delta"] if "delta" in params else 1e-6
    
    # Parameters
    r = 70
    s = 10
    # a = 3
    # b = 1.5
    # sig = 2.5
    
    # Start timing
    start_time = time.time()
    # Count simulation runs
    total_samples = 0
    
    # Initial polytope Ax >= b
    A = np.concatenate( (np.eye(d),-np.eye(d)), axis=0 )
    b = np.concatenate((np.ones(d,),-N * np.ones((d,))))
    # print(A.shape,b.shape)
    # Initial volumetric center
    z = (N+1)/2 * np.ones((d,))

    
    # Round to an integral solution
    x_opt = Round(F,x_bar,params)["x_opt"]
    
    # Stop timing
    stop_time = time.time()
    
    return {"x_opt":x_opt, "time":stop_time-start_time, "total":total_samples}
    



















