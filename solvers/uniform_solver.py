# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 14:30:51 2020

@author: haixiang_zhang

Uniform sampling algorithm for one-dim problems
"""

import math
import numpy as np
import time
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
    # Initialize samples
    cur_samples = 0
    # Empirical mean
    hat_F = np.zeros(S.shape)
    
    # Start timing
    start_time = time.time()
    # Count simulation runs
    total_samples = 0
    
    while S.shape[0] > 2:
        
        # Upper bound on samples needed (devide 80 to ensure correctness)
        num_samples = RequiredSamples(delta/2/T_max,S.shape[0]*eps/20,
                                      params)
        print(cur_samples,num_samples)
        # Simulation
        for i in range(num_samples - cur_samples):
            for j in range(S.shape[0]):
                hat_F[j] = (hat_F[j] * (i + cur_samples) + F([S[j]]))\
                            / (cur_samples + i + 1)
        
            # Check conditions
            CI = ConfidenceInterval(delta/2/T_max,params,cur_samples+i+1)
            
            # Condition (i)
            if np.max(hat_F) - np.min(hat_F) > 2 * CI:
                cur_samples += (i + 1)
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
            
            # Update total samples
            total_samples += cur_samples * (S.shape[0] - i_right + i_left + 1)
            # Update S
            S = S[ i_left+1 : i_right ]
            # Update empirical mean
            hat_F = hat_F[ i_left+1 : i_right ]

        # Condition (ii)
        else:
            cur_samples = num_samples
            # Update total samples
            total_samples += cur_samples * math.floor(S.shape[0] / 2)
            # Update S
            S = np.array([ S[j] for j in range(0,S.shape[0],2) ])
            # Update empirical mean
            hat_F = np.array([ hat_F[j] for j in range(0,hat_F.shape[0],2) ])
        
        print(S,hat_F)
    
    # If S is a singleton
    if S.shape[0] == 1:
        x_opt = S[0]
    # Solve the sub-problem with 2 points
    else:
        # Upper bound on samples needed
        num_samples = RequiredSamples(delta/2/T_max,eps/4,params)
        
        # Simulation
        for i in range(num_samples - cur_samples):
            for j in range(2):
                hat_F[j] = ( hat_F[j] * (cur_samples+i) + F([S[j]]) )\
                            / (cur_samples + i + 1)
            
            # Check confidence interval
            CI = ConfidenceInterval(delta/2/T_max,params,cur_samples+i+1)
            # Differentiable
            if np.max(hat_F) - np.min(hat_F) > 2 * CI:
                break
        
        # Update total simulations
        total_samples += 2 * cur_samples
        # Return the point with the minimal empirical mean
        x_opt = S[np.argmin(hat_F)]
    
    # Stop timing
    stop_time = time.time()

    return {"x_opt":x_opt, "time":stop_time-start_time, "total":total_samples}

