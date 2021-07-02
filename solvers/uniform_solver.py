# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 14:30:51 2020

@author: haixiang_zhang

Uniform sampling algorithm for one-dim problems
"""

import math
import numpy as np
import time
import copy
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
    
    while S.shape[0] > 5:
        
        # Upper bound on samples needed (devide 80 to ensure correctness)
        num_samples = RequiredSamples(delta/2/T_max,S.shape[0]*eps/20,
                                      params)
        # print(cur_samples,num_samples)
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
        
        # print(S)
    
    # Update total simulations
    total_samples += (cur_samples * S.shape[0])
    
    # If S is a singleton
    if S.shape[0] == 1:
        x_opt = S[0]
    # Solve the sub-problem
    else:
        # Number of points
        num = S.shape[0]
        # Upper bound on samples needed
        num_samples = RequiredSamples(delta/2/T_max,eps/4,params)
        # Stop simumating if already too large
        blocked = np.zeros((num,))
        
        # Simulation
        for i in range(num_samples - cur_samples):
            for j in range(num):
                if blocked[j] == 0:
                    hat_F[j] = ( hat_F[j] * (cur_samples+i) + F([S[j]]) )\
                                / (cur_samples + i + 1)
            # Update total samples
            total_samples += np.sum(1 - blocked)
            
            # Check confidence interval
            CI = ConfidenceInterval(delta/2/T_max,params,cur_samples+i+1)
            # Block points with large empirical means
            blocked[ hat_F - np.min(hat_F) > 2 * CI ] = 1
            hat_F[ hat_F - np.min(hat_F) > 2 * CI ] = np.inf
            # print(hat_F)
            # Only one point left
            if np.sum(blocked) == (num - 1):
                break
    
        # Return the point with the minimal empirical mean
        x_opt = S[np.argmin(hat_F)]
        f_opt = np.min(hat_F)
    
    # Stop timing
    stop_time = time.time()

    return {"x_opt":x_opt, "time":stop_time-start_time, "total":total_samples,
            "f_opt":f_opt}

def UniformSolverMulti(F,params):
    """
    The uniform sampling algorithm for multi-dim problems.
    """
    
    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    # sigma = params["sigma"] if "sigma" in params else 1
    eps = params["eps"] if "eps" in params else 1e-1
    delta = params["delta"] if "delta" in params else 1e-6
    
    # The one-dim case
    if d == 1:
        return UniformSolver(F,params)
    
    # Initial active set
    S = np.linspace(1,N,N,dtype=np.int16)
    # Total iterations
    T_max = N
    # # Initialize samples
    # cur_samples = 0
    # Empirical mean
    hat_F = np.zeros(S.shape)
    # N_cur
    N_cur = np.inf
    
    # Store new parameters
    params_temp = copy.copy(params)
    params_temp["d"] = d - 1
    params_temp["delta"] = delta / 2 / T_max
    
    # Start timing
    start_time = time.time()
    # Count simulation runs
    total_samples = 0
    
    while S.shape[0] > 5:
        
        # # Upper bound on samples needed (devide 80 to ensure correctness)
        # num_samples = RequiredSamples(delta/2/T_max,S.shape[0]*eps/20,
        #                               params)
        # # print(cur_samples,num_samples)
        # # Simulation
        # for i in range(num_samples - cur_samples):
        
        if S.shape[0] < N_cur / 2:
            CI = S.shape[0] * eps / 10
            params_temp["eps"] = CI
            N_cur = S.shape[0]
            
            # Solve sub-problems
            # Record current minimal
            cur_min = np.inf
            for j in range(S.shape[0]):
                # print("here!" + str(j) )
                # The (d-1)-component function with the last element fixed
                F_temp = lambda x: F( np.concatenate((x,[S[j]])) )
                results = UniformSolverMulti(F_temp,params_temp)
                hat_F[j] = results["f_opt"]
                total_samples += results["total"]
                
                # Update the minimal value
                cur_min = hat_F[j] if hat_F[j] < cur_min else cur_min
                # Early stopping
                if hat_F[j] > cur_min + 1 * CI:
                    hat_F[j:] = np.inf
                    break
    
                # # Condition (i)
                # if np.max(hat_F) - np.min(hat_F) > 2 * CI:
                #     cur_samples += (i + 1)
                #     break
        
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
            # total_samples += cur_samples * (S.shape[0] - i_right + i_left + 1)
            # Update S
            S = S[ i_left+1 : i_right ]
            # Update empirical mean
            hat_F = hat_F[ i_left+1 : i_right ]

        # Condition (ii)
        else:
            # cur_samples = num_samples
            # Update total samples
            # total_samples += cur_samples * math.floor(S.shape[0] / 2)
            # Update S
            S = np.array([ S[j] for j in range(0,S.shape[0],2) ])
            # Update empirical mean
            hat_F = np.array([ hat_F[j] for j in range(0,hat_F.shape[0],2) ])
        
        if d == 3:
            print(S,total_samples)
    
    # # Update total simulations
    # total_samples += (cur_samples * S.shape[0])
    
    # If S is a singleton
    if S.shape[0] == 1:
        CI = eps / 2
        params_temp["eps"] = CI
        # The (d-1)-component function with the last element fixed
        F_temp = lambda x: F( np.concatenate((x,S)) )
        results = UniformSolverMulti(F_temp,params_temp)
        f_opt = results["f_opt"]
        total_samples += results["total"]
        x_opt = np.concatenate((results["x_opt"],S))
    # Solve the sub-problem
    else:
        CI = eps / 2
        params_temp["eps"] = CI
        # N_cur = S.shape[0]
        
        # Solve sub-problems
        # Record current minimal
        cur_min = np.inf
        for j in range(S.shape[0]):
            # The (d-1)-component function with the last element fixed
            F_temp = lambda x: F( np.concatenate((x,[S[j]])) )
            results = UniformSolverMulti(F_temp,params_temp)
            hat_F[j] = results["f_opt"]
            total_samples += results["total"]
            # print(total_samples)
            
            # Update the minimal value
            cur_min = hat_F[j] if hat_F[j] < cur_min else cur_min
            # Early stopping
            if hat_F[j] > cur_min + 1 * CI:
                hat_F[j:] = np.inf
                break
                
        
        # # Number of points
        # num = S.shape[0]
        
        # # Upper bound on samples needed
        # num_samples = RequiredSamples(delta/2/T_max,eps/4,params)
        # # Stop simumating if already too large
        # blocked = np.zeros((num,))
        
        # # Simulation
        # for i in range(num_samples - cur_samples):
        #     for j in range(num):
        #         if blocked[j] == 0:
        #             hat_F[j] = ( hat_F[j] * (cur_samples+i) + F([S[j]]) )\
        #                         / (cur_samples + i + 1)
        #     # Update total samples
        #     total_samples += np.sum(1 - blocked)
            
        #     # Check confidence interval
        #     CI = ConfidenceInterval(delta/2/T_max,params,cur_samples+i+1)
        #     # Block points with large empirical means
        #     blocked[ hat_F - np.min(hat_F) > 2 * CI ] = 1
        #     hat_F[ hat_F - np.min(hat_F) > 2 * CI ] = np.inf
        #     # print(hat_F)
        #     # Only one point left
        #     if np.sum(blocked) == (num - 1):
        #         break
    
        # Return the point with the minimal empirical mean
        x_opt = S[np.argmin(hat_F)]
        # print(x_opt)
        # f_opt = np.min(hat_F)
        
        # The (d-1)-component function with the last element fixed
        F_temp = lambda x: F( np.concatenate((x,[x_opt])) )
        results = UniformSolverMulti(F_temp,params_temp)
        # print(results)
        f_opt = results["f_opt"]
        total_samples += results["total"]
        if d == 2:
            x_opt = np.concatenate(([results["x_opt"]],[x_opt]))
        else:
            x_opt = np.concatenate((results["x_opt"],[x_opt]))
    
    # Stop timing
    stop_time = time.time()

    return {"x_opt":x_opt, "time":stop_time-start_time, "total":total_samples,
            "f_opt":f_opt}