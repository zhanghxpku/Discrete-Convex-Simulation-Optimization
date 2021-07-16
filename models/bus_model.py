# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:35:29 2020

@author: haixiang_zhang
Generate news vendor with dynamic consumer substitution examples.
"""

import numpy as np
from scipy import stats
# import matplotlib.pyplot as plt
import time

def BusModel(params):
    """
    Generate the objective function
    """
    
    # Retrieve parameters: upper bound of time
    tau = params["N"] if "N" in params else 100
    # Retrieve parameters: number of buses
    d = params["d"] if "d" in params else 1
    
    # Parameters
        
    return {"F": lambda y: WaitingTime(y,params,False),
            "f": lambda y: WaitingTime(y,params,True)}


def WaitingTime(x,params,expectation=False):
    """
    Compute the waiting time
    """
    
    # Parameters: lambda
    lam = 10
    # Retrieve parameters: upper bound of time
    tau = params["N"] if "N" in params else 100
    # Retrieve parameters: number of buses
    d = params["d"] if "d" in params else 1
    
    # Length of intervals
    length = np.zeros((d+1,))
    length[0] = x[0] - 1
    length[2:-1] = np.diff(x)
    length[-1] = tau - x[-1] + 1
    
    if expectation:
        return lam / 2 * np.sum( length ** 2 )
    else:
        # Number of arrivals in each interval
        n = stats.poisson.rvs(length,size=(d+1,))
    
        # Waiting time
        wait_time = 0
        
        # Compute the waiting time
        for i in range(d+1):
            # # Number of arrivals
            # n = stats.poisson.rvs(length[i] * lam)
            # Waiting time
            wait_time += stats.uniform.rvs(0,n[i] * length[i] / 2)
        
        return wait_time
        