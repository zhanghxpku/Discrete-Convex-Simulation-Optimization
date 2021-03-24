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

def LILUCBSolver(F,params):
    """
    The lil'UCB algorithm for one-dim problems.
    """
    
    # Retrieve parameters
    if "d" in params and params["d"] != 1:
        print("lil'UCB only works for one-dim problems.")
        return None
    N = params["N"] if "N" in params else 2
    sigma = params["sigma"] if "sigma" in params else 1
    eps0 = params["eps"] if "eps" in params else 1e-1
    nu = params["delta"] if "delta" in params else 1e-6
    
    # Parameters of the algorithm
    # beta = 1
    # a = 9
    # eps = 1e-2
    # delta = ( nu*eps / 5 / (2+eps) ) ** ( 1 / (1+eps) )
    # Heuristic parameters
    beta = 1/2
    a = 1 + 10 / N
    eps = 0
    delta = nu / 5

    # Number of samples on each arm
    cur_samples = np.zeros((N,))
    # Empirical mean
    hat_F = np.zeros((N,))
    
    # Start timing
    start_time = time.time()
    
    # Initialize with one sample on each arm
    for i in range(N):
        hat_F[i] = -F([i+1])
        cur_samples[i] = 1
    
    while (1+a)*np.max(cur_samples) < 1 + a * np.sum(cur_samples):
        
        # Find the arm with maximal UCB
        deviation = np.log( np.log((1+eps)*cur_samples+2)/delta ) / cur_samples
        deviation = (1+beta)*(1+np.sqrt(eps))*np.sqrt( 2*(1+eps)*deviation )
        I_t = np.argmax( hat_F + deviation * sigma )
        
        # Sample I_t
        hat_F[I_t] = ( hat_F[I_t] * cur_samples[I_t] - F([I_t+1]) )\
                                            / (cur_samples[I_t] + 1)
        cur_samples[I_t] += 1
        
        # Check the LS criterion
        bound = np.log( 2*N*np.log((1+eps)*cur_samples+2)/delta ) / cur_samples
        bound = (1+np.sqrt(eps)) * sigma * np.sqrt( 2*(1+eps)*bound )
        
        i_max = np.argmax(hat_F)
        lower = hat_F[i_max] - bound[i_max]
        upper = np.max( [np.max(hat_F[:i_max]+bound[:i_max]),
                       np.max(hat_F[i_max+1:]+bound[i_max+1:])] )
        
        if np.sum(cur_samples) % 10000 == 0:
            print(lower,upper,np.sum(cur_samples))
        if lower >= upper - eps0/2:
            break
    
    # while S.shape[0] > 5:
        
    #     # Upper bound on samples needed (devide 80 to ensure correctness)
    #     num_samples = RequiredSamples(delta/2/T_max,S.shape[0]*eps/20,
    #                                   params)
    #     # print(cur_samples,num_samples)
    #     # Simulation
    #     for i in range(num_samples - cur_samples):
    #         for j in range(S.shape[0]):
    #             hat_F[j] = (hat_F[j] * (i + cur_samples) + F([S[j]]))\
    #                         / (cur_samples + i + 1)

    #         # Check conditions
    #         CI = ConfidenceInterval(delta/2/T_max,params,cur_samples+i+1)
            
    #         # Condition (i)
    #         if np.max(hat_F) - np.min(hat_F) > 2 * CI:
    #             cur_samples += (i + 1)
    #             break
        
    #     # Condition (i)
    #     if np.max(hat_F) - np.min(hat_F) > 2 * CI:
    #         # The minimal index
    #         i_min = np.argmin(hat_F)
    #         # Left bound
    #         i_left = i_min
    #         while i_left > -1:
    #             if hat_F[i_left] - hat_F[i_min] > 2 * CI:
    #                 break
    #             else:
    #                 i_left -= 1
    #         # Right bound
    #         i_right = i_min
    #         while i_right < S.shape[0]:
    #             if hat_F[i_right] - hat_F[i_min] > 2 * CI:
    #                 break
    #             else:
    #                 i_right += 1
            
    #         # Update total samples
    #         total_samples += cur_samples * (S.shape[0] - i_right + i_left + 1)
    #         # Update S
    #         S = S[ i_left+1 : i_right ]
    #         # Update empirical mean
    #         hat_F = hat_F[ i_left+1 : i_right ]

    #     # Condition (ii)
    #     else:
    #         cur_samples = num_samples
    #         # Update total samples
    #         total_samples += cur_samples * math.floor(S.shape[0] / 2)
    #         # Update S
    #         S = np.array([ S[j] for j in range(0,S.shape[0],2) ])
    #         # Update empirical mean
    #         hat_F = np.array([ hat_F[j] for j in range(0,hat_F.shape[0],2) ])
        
    #     # print(S)
    
    # # Update total simulations
    # total_samples += (cur_samples * S.shape[0])
    
    # # If S is a singleton
    # if S.shape[0] == 1:
    #     x_opt = S[0]
    # # Solve the sub-problem
    # else:
    #     # Number of points
    #     num = S.shape[0]
    #     # Upper bound on samples needed
    #     num_samples = RequiredSamples(delta/2/T_max,eps/4,params)
    #     # Stop simumating if already too large
    #     blocked = np.zeros((num,))
        
    #     # Simulation
    #     for i in range(num_samples - cur_samples):
    #         for j in range(num):
    #             if blocked[j] == 0:
    #                 hat_F[j] = ( hat_F[j] * (cur_samples+i) + F([S[j]]) )\
    #                             / (cur_samples + i + 1)
    #         # Update total samples
    #         total_samples += np.sum(1 - blocked)
            
    #         # Check confidence interval
    #         CI = ConfidenceInterval(delta/2/T_max,params,cur_samples+i+1)
    #         # Block points with large empirical means
    #         blocked[ hat_F - np.min(hat_F) > 2 * CI ] = 1
    #         hat_F[ hat_F - np.min(hat_F) > 2 * CI ] = np.inf
    #         # print(hat_F)
    #         # Only one point left
    #         if np.sum(blocked) == (num - 1):
    #             break
    
    # Return the point with the minimal empirical mean
    x_opt = np.argmax(hat_F)
    # Count simulation runs
    total_samples = np.sum(cur_samples)
    
    # Stop timing
    stop_time = time.time()

    return {"x_opt":x_opt, "time":stop_time-start_time, "total":total_samples}

