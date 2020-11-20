# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 14:30:51 2020

@author: haixiang_zhang

Uniform sampling algorithm for one-dim problems
"""

import math
import numpy as np
from utils.subgaussian import RequiredSamples, ConfidenceInterval

def UniformSolver(F,params):
    """
    The uniform sampling algorithm for one-dim problems.
    """
    
    # Retrieve parameters
    if "d" in params and params["d"] != 1:
        print("Uniform sampling only works for one-dim problems.")
        return None
    N = params["N"] if "N" in params else 2
    # sigma = params["sigma"] if "sigma" in params else 1
    eps = params["eps"] if "eps" in params else 1e-1
    delta = params["delta"] if "delta" in params else 1e-6

    # Initial active set
    S = np.linspace(1,N,N,dtype=np.int16)
    # Total iterations
    T_max = N
    # # Initial step size
    # step = 1
    
    while S.shape[0] > 2:
        
        # Upper bound on samples needed (devide 80 to ensure correctness)
        num_samples = RequiredSamples(delta/2/T_max,S.shape[0]*eps/8,
                                      params)
        # Empirical mean
        hat_F = np.zeros(S.shape)
        
        # Simulation
        for i in range(num_samples):
            for j in range(S.shape[0]):
                hat_F[j] = ( hat_F[j] * i + F([S[j]]) ) / (i + 1)
        
            # Check conditions
            CI = ConfidenceInterval(delta/2/T_max,params,i+1)
            
            # Condition (i)
            if np.max(hat_F) - np.min(hat_F) > 2 * CI:
                break
        
        # Condition (i)
        if np.max(hat_F) - np.min(hat_F) > 2 * CI:
            # The minimal index
            i_min = np.argmin(hat_F)
            # Left bound
            i_left = i_min
            while i_left > -1:
                if hat_F[i_left] - hat_F[i_min] > 2 * CI:
                    break
                else:
                    i_left -= 1
            # Right bound
            i_right = i_min
            while i_right < S.shape[0]:
                if hat_F[i_right] - hat_F[i_min] > 2 * CI:
                    break
                else:
                    i_right += 1
            
            # print(i_left,i_min,i_right,S.shape[0],CI)
            
            # Update S
            S = S[ i_left+1 : i_right ]
            
        # Condition (ii)
        else:
            S = np.array([ S[j] for j in range(0,S.shape[0],2) ])
        
        # print(S)
    
    # If S is a singleton
    if S.shape[0] == 1:
        return S[0]
    
    # Solve the sub-problem with 2 points
    hat_F = np.zeros((2,))
    # Upper bound on samples needed
    num_samples = RequiredSamples(delta/2/T_max,eps/4,params)
    
    # Simulation
    for i in range(num_samples):
        for j in range(2):
            hat_F[j] = ( hat_F[j] * i + F([S[j]]) ) / (i + 1)
        
        # Check confidence interval
        CI = ConfidenceInterval(delta/2/T_max,params,i+1)
        # Differentiable
        if np.max(hat_F) - np.min(hat_F) > 2 * CI:
            break

    # Return the point with the minimal empirical mean
    return S[np.argmin(hat_F)]

