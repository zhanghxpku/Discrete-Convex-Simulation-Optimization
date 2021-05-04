# -*- coding: utf-8 -*-
"""
Created on Sat May  1 21:07:45 2021

@author: haixiang_zhang
Generate one queue for 24-hours.
"""

import numpy as np
from scipy import stats
# import matplotlib.pyplot as plt
import time

def QueueORModel(params):
    """
    Generate the objective function
    """
    
    # Service time
    mean = 0.25
    var = 0.1
    # Compute the parameters of log-normal dist
    s = np.sqrt(np.log( (1 + np.sqrt(1 + 4*var)) / 2))
    loc = mean - np.exp(s**2/2)
    service = lambda size: stats.lognorm.rvs(s,loc=loc,size=size)
    
    return {"F": lambda x: WaitingTime(x,service,params)}

def QueueRegORModel(params):
    """
    Generate the objective function with a regularization
    """
    
    # Regularizer
    c = params["c"] if "c" in params else 0
    # Service time
    mean = 0.25
    var = 0.1
    # Compute the parameters of log-normal dist
    s = np.sqrt(np.log( (1 + np.sqrt(1 + 4*var)) / 2))
    loc = mean - np.exp(s**2/2)
    service = lambda size: stats.lognorm.rvs(s,loc=loc,size=size)
    
    return {"F": lambda x: (WaitingTime(x,service,params) + np.sum(c * x))}

def WaitingTime(x,service,params):
    """
    Compute the waiting time of the queue
    """
    
    # Retrieve parameters: max number of servers
    N = params["N"] if "N" in params else 2
    # Retrieve parameters: number of decisions in each period
    M = params["M"] if "M" in params else 24
    
    # Parameters
    # Gamma_1 = stats.uniform.rvs(0.75,0.5)
    
    # Generate intensity functions (i-th hour)
    lambda_1 = lambda t: 4 * N * ( 1 - np.abs( t - 12 ) / 12 )
    
    # Maximal rates
    max_1 = lambda_1(12)
    
    # The total waiting time
    t1, n1 = SingleQueue(x,lambda_1,max_1,service,params)
    
    return t1 / n1

def SingleQueue(x,intensity,max_rate,service_t,params):
    """
     Compute the waiting time of the queue
    """
    
    # Retrieve parameters: max number of servers
    N = params["N"] if "N" in params else 2
    # Retrieve parameters: number of decisions in each period
    M = params["M"] if "M" in params else 24
    # Retrieve parameters
    T = params["T"] if "T" in params else 24 / M
    
    # Number of arrivals
    n = stats.poisson.rvs(24 * max_rate)
    # Arrival times
    t = stats.uniform.rvs(0,24,n)
    # Thining
    t = t[ stats.uniform.rvs(0,1,n) < intensity(t) / max_rate ]
    # Sorting
    t = np.sort(t)
    # print(t.shape)
    # Service time
    service_time = service_t(t.shape[0])
    # Total number of customers
    customer_num = t.shape[0]
    # Total waiting time
    wait_time = 0
    # The finishing time of each server
    finish_time = np.zeros((N,))
    # Current slot
    k = 0
    # Randomly choose x[k] servers to work
    active_ind = np.random.choice(np.arange(N), int(x[k]), False)
    finish_time_ac = finish_time[active_ind]
    
    for j,i in enumerate(t):
        # Find the earliest finishing time
        next_server = np.argmin(finish_time_ac)
        finish_min = finish_time_ac[next_server]
        
        # Decide if we need to move to the next slot
        while finish_min >= (k+1) * T or i >= (k+1) * T:
            # Update the original array
            finish_time[active_ind] = finish_time_ac
            k += 1
            if k >= M:
                break
            # Randomly choose x[k] servers to work
            active_ind = np.random.choice(np.arange(N), int(x[k]), False)
            finish_time_ac = finish_time[active_ind]
        
            # Find the earliest finishing time
            next_server = np.argmin(finish_time_ac)
            finish_min = finish_time_ac[next_server]
        
        if k >= M:
            break
        # Update waiting time
        wait_time += max(finish_min - i, 0)
        # Update finishing time
        finish_time_ac[next_server] = max(finish_min,i) + service_time[j]

    return wait_time, customer_num
