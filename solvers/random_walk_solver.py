# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:59:11 2020

@author: haixiang_zhang

Cutting-plane method based on random walk
"""

import math
import numpy as np
import time
from utils.lovasz import Lovasz, Round, SO
from utils.subgaussian import RequiredSamples

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

    # Total number of iterations
    T = math.ceil(d * math.log(L*N/eps), 1.5)
    # Set of points where SO is called and their empirical means
    S = np.zeros((0,d+1))
    print(T)
    
    # Early stopping
    F_old = np.inf
    
    # Start timing
    start_time = time.time()
    # Count simulation runs
    total_samples = 0
    
    # Initial polytope Ax >= b
    A = np.concatenate( (np.eye(d),-np.eye(d)), axis=0 )
    b = np.concatenate((np.ones(d,),-N * np.ones((d,))))
    # Initial volumetric center
    z = (N+1)/2 * np.ones((d,))
    # Number of points to approximate covariance (N in the paper)
    M = 5 * 10 * d * math.log(d) * max( 10, math.log(d) )
    # Number of steps to approximate the uniform measure in P
    K = d**3 * 2e3
    
    for t in range(T):
        
        # Early stopping
        F_new = np.mean(S[-10:,-1])
        if F_new >= F_old:
            break
        else:
            F_old = F_new
    
    # Find the point with minimial empirical mean
    i = np.argmin(S[:,-1])
    x_bar = S[i,:d]
    
    # Round to an integral solution
    x_opt = Round(F,x_bar,params)["x_opt"]
    
    # Stop timing
    stop_time = time.time()
    
    return {"x_opt":x_opt, "time":stop_time-start_time, "total":total_samples}









