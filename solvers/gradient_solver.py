# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:57:17 2020

@author: haixiang_zhang
"""

import math
import numpy as np
import time
from utils.lovasz import Lovasz, Round

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
    # Initial objective function value
    f_old = F(x)
    
    # Iterate numbers and step size
    T = math.ceil( max( 1*d*(N**2)*sigma / (eps**2) * math.log(2/delta),
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
    
    # Truncated subgradient descent
    for t in range(T):
        
        # Compute subgradient
        _, sub_grad = Lovasz(F,x,params)
        total_samples += 2 * d
        
        # Truncate subgradient
        sub_grad = np.clip(sub_grad, -M, M)
        # if np.linalg.norm(sub_grad,np.inf) > M:
        #     print("Truncated!")
        
        # Update and project the current point
        x = x - eta * sub_grad
        x = np.clip(x,1,N)
        
        # Update the moving average
        x_avg = (x_avg * t + x) / (t + 1)
        
        # if np.linalg.norm(sub_grad,np.inf) < tol / N:
        #     break
        
        # if t > 10000 and np.linalg.norm(ret - np.round(ret),np.inf) < 1e-1:
        #     ret = np.round(ret),np.inf
        #     break
        
        # if t % 10000 == 1000:
        #     f_new, _ = Lovasz(d,N,f,x_avg)
        #     print(f_new)
        #     # if f_new > f_old - 0e-6:
        #     #     break
        #     # else:
        #     #     f_old = f_new  
    
    # Round to an integral point
    output_round = Round(F,x_avg,params)
    x_opt = output_round["x_opt"]
    total_samples += output_round["total"]
    # Stop timing
    stop_time = time.time()
    
    return {"x_opt":x_opt, "time":stop_time-start_time, "total":total_samples}
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

