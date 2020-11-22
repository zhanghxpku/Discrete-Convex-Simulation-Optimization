# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:57:17 2020

@author: haixiang_zhang

Truncated subgradient descent method.
"""

import math
import numpy as np
import time
from utils.lovasz import Lovasz, Round
from utils.subgaussian import RequiredSamples

def GradientSolver(F,params):
    """
    The truncated subgradient method for multi-dim problems.
    """
    
    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    sigma = params["sigma"] if "sigma" in params else 1
    eps = params["eps"] if "eps" in params else 1
    delta = params["delta"] if "delta" in params else 1e-6
    L = params["L"] if "L" in params else 1
    
    # Initial point
    x = np.floor(np.ones((d,)) * N / 2)
    # The moving average
    x_avg = np.copy(x)
    # Comparion of objective function values
    f_old = 0
    f_new = 0
    # Check stopping criterion every 1000 iterations
    interval = RequiredSamples(delta/2,eps/5,params)
    print(interval)
    
    # Iterate numbers and step size
    T = math.ceil( max( 64*d*(N**2)*sigma / (eps**2) * math.log(2/delta),
                       (d**2) * (L**2) / (eps**2),  
                       64*(d**2)*(N**2) / (eps**2) * math.log(sigma*d**2/N**3)
                       ) )
    M = max(2*sigma*math.sqrt(math.log( max(4*sigma*d*N*T / eps, 1) )), L) 
    print(T,M)
    eta =  N / M / np.sqrt( T )
    
    # Start timing
    start_time = time.time()
    # Count simulation runs
    total_samples = 0
    
    # Weighted average
    alpha = 0.75
    weight_cum = 0
    
    # Truncated subgradient descent
    for t in range(T):
        
        # Compute subgradient
        hat_F, sub_grad = Lovasz(F,x,params)
        total_samples += (2*d)
        
        # Truncate subgradient
        sub_grad = np.clip(sub_grad, -M, M)
        # if np.linalg.norm(sub_grad,np.inf) > M:
        #     print("Truncated!")
        
        # Update and project the current point
        x = x - eta * sub_grad
        x = np.clip(x,1,N)
        
        # Update the moving average
        new_weight = weight_cum * (1-alpha) + alpha
        x_avg = (x_avg * weight_cum * (1-alpha) + x*alpha) / new_weight
        # Update the function value
        f_new = (f_new * weight_cum * (1-alpha) + hat_F*alpha) / new_weight
        # Update the cumulative weight
        weight_cum = new_weight
        
        if t % interval == 0:
            f, _ = Lovasz(F,x_avg,params)
            print(f_new, hat_F, f)
        
        # Early stopping
        if t % interval == interval - 1 and t >= interval:
            # Decay is not sufficient
            if f_new - f_old >= 0:
                break

        if t % interval == interval - 1:
            # Update f_old and f_new
            f_old = f_new
    
    # Round to an integral point
    output_round = Round(F,x_avg,params)
    x_opt = output_round["x_opt"]
    total_samples += output_round["total"]
    # Stop timing
    stop_time = time.time()
    
    return {"x_opt":x_opt, "time":stop_time-start_time, "total":total_samples}
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
