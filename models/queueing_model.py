# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:35:29 2020

@author: haixiang_zhang
Generate two queues under resource constraints examples.
"""

import numpy as np
from scipy import stats
# import matplotlib.pyplot as plt
import time

def QueueModel(params):
    """
    Generate the objective function
    """
    
    # Service time
    mean = (0.25,0.15)
    var = (0.1,0.1)
    # Compute the parameters of log-normal dist
    s = np.sqrt(np.log( (1 + np.sqrt(1 + 4*var[0])) / 2))
    loc = mean[0] - np.exp(s**2/2)
    service_1 = lambda size: stats.lognorm.rvs(s,loc=loc,size=size)
    # Compute the parameters of Gamma dist   
    scale = var[1] / mean[1]
    a = mean[1] / scale
    service_2 = lambda size: stats.gamma.rvs(a,scale=scale,size=size)
    service = (service_1,service_2)
    
    return {"F": lambda x: WaitingTime(x,service,params)}

def WaitingTime(x,service,params):
    """
    Compute the waiting time of two queues
    """
    
    # Retrieve parameters
    N = params["N"] if "N" in params else 2
    
    # Parameters
    X = stats.uniform.rvs(0.75,0.5)
    Y = stats.uniform.rvs(0.75,0.5)
    Z = stats.uniform.rvs(-0.5,1)
    Gamma_1 = X + Z
    Gamma_2 = Y - Z
    
    # Generate intensity functions
    lambda_1 = lambda t: Gamma_1 * ( 300 + 100 * np.sin(0.3*t) )
    lambda_2 = lambda t: Gamma_2 * ( 500 + 200 * np.sin(0.2*t) )
    
    # Maximal rates
    max_1 = 400 * Gamma_1
    max_2 = 700 * Gamma_2
    
    # The total waiting time
    return SingleQueue(x[0],lambda_1,max_1,service[0],params)\
                   + SingleQueue(N+1-x[0],lambda_2,max_2,service[1],params)

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
    t = t[ stats.uniform.rvs(0,1,n) < intensity(t) / max_rate ]
    # Sorting
    t = np.sort(t)
    # print(t.shape)
    
    # The finishing time of each server
    finish_time = np.zeros((int(num_server),))
    # Total waiting time
    wait_time = 0
    # Service time
    service_time = service_t(t.shape[0])
    
    # p = np.zeros(t.shape)
    
    for j,i in enumerate(t):
        # Find the earliest finishing time
        next_server = np.argmin(finish_time)
        finish_min = finish_time[next_server]
        # Update waiting time
        wait_time += max(finish_min - i, 0)
        # Update finishing time
        finish_time[next_server] = max(finish_min,i) + service_time[j]
        # p[j] = max(finish_min - i, 0)
    # plt.figure()
    # plt.plot(t,p)
    # plt.savefig(str(t.shape[0]) + ".png")
    
    return wait_time / t.shape[0]
    

