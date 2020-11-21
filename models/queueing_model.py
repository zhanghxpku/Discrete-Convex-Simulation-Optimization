# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:35:29 2020

@author: haixiang_zhang
Generate two queues under resource constraints examples.
"""

import numpy as np
from scipy import stats

def QueueModel(params):
    """
    Generate a queueing model with two queues
    """
    
    # Retrieve parameters
    N = params["N"] if "N" in params else 2
    
    # Generate intensity functions
    X = stats.uniform.rvs(0.75,1.25)
    Y = stats.uniform.rvs(0.75,1.25)
    Z = stats.uniform.rvs(-0.5,0.5)
    Gamma_1 = X + Z
    Gamma_2 = Y - Z
    lambda_1 = lambda t: Gamma_1 * ( 300 + 100 * np.sin(0.3*t) )
    lambda_2 = lambda t: Gamma_2 * ( 500 + 200 * np.sin(0.2*t) )
    
    # Maximal rates
    max_1 = 400 * Gamma_1
    max_2 = 700 * Gamma_2
    
    # Service time
    mean = (0.1,0.15)
    var = (0.1,0.1)
    # Compute the parameters of log-normal dist
    s = np.sqrt(np.log( (1 + np.sqrt(1 + 4*var[0])) / 2))
    loc = mean[0] - np.exp(s**2/2)
    service_1 = lambda: stats.lognorm.rvs(s,loc=loc)
    # Compute the parameters of Gamma dist   
    scale = var[1] / mean[1]
    a = mean[1] / scale
    service_2 = lambda: stats.gamma.rvs(a,scale=scale)
    
    # The total waiting time
    F = lambda x: SingleQueue(x-1,lambda_1,max_1,service_1,params)\
                  + SingleQueue(N-x,lambda_2,max_2,service_2,params)
    
    return {"F":F}

def SingleQueue(num_server,intensity,max_rate,service_t,params):
    """
     Compute the waiting time of a single queue
    """
    
    # Retrieve parameters
    T = params["T"] if "T" in params else 10
    
    # Number of arrivals
    n = stats.poisson.rvs(T * max_rate)
    # Arrival times
    t = stats.uniform.rvs(0,T,n)
    # Thining
    t = t[ stats.uniform.rvs(0,max_rate,n) < intensity(t) ]
    
    # The finishing time of each server
    finish_time = np.zeros((num_server,))
    # Total waiting time
    wait_time = 0
    
    for i in t:
        
    
    
    



















