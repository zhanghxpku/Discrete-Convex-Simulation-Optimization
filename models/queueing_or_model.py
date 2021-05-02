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
    mean = (0.75,0.65)
    var = (0.1,0.1)
    # Compute the parameters of log-normal dist
    s = np.sqrt(np.log( (1 + np.sqrt(1 + 4*var[0])) / 2))
    loc = mean[0] - np.exp(s**2/2)
    service = lambda size: stats.lognorm.rvs(s,loc=loc,size=size)
    
    return {"F": lambda x: WaitingTime(x,service,params)}

def WaitingTime(x,service,params):
    """
    Compute the waiting time of the queue
    """
    
    # Retrieve parameters: number of decisions in each period
    M = params["M"] if "M" in params else 24
    
    # Parameters
    Gamma_1 = stats.uniform.rvs(0.75,0.5)
    
    # Generate intensity functions (i-th hour)
    lambda_1 = lambda t, i: Gamma_1 * ( 65 + (20 + 15 * np.sin(2*np.pi/M*i)) * np.sin(0.3*t) )
    
    # Maximal rates
    max_1 = 100 * Gamma_1
    
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
    T = params["T"] if "T" in params else 2 / M
    
    # Total waiting time
    wait_time = 0
    # The finishing time of each server
    finish_time = np.zeros((N,))
    
    # For the k-th hour
    for k in range(M):
        # Randomly choose x[k] servers to work
        if abs(x[k] - int(x[k])) > 0.01:
            print(x[k])
        active_ind = np.random.choice(np.arange(N), int(x[k]), False)
        finish_time_ac = np.ones((int(x[k]),)) * k * T
        old_ind = finish_time[active_ind] < np.inf
        finish_time_ac[old_ind] = (finish_time[active_ind])[old_ind]
        # Update old finishing time to inf
        finish_time = np.ones((N,)) * np.inf

        # Number of arrivals
        n = stats.poisson.rvs(T * max_rate)
        # Arrival times
        t = stats.uniform.rvs(0,T,n)
        # Thining
        t = t[ stats.uniform.rvs(0,1,n) < intensity(t,k) / max_rate ]
        # Sorting
        t = np.sort(t) + k * T
        # print(t.shape)
        # Service time
        service_time = service_t(t.shape[0])
        # p = np.zeros(t.shape)
        
        for j,i in enumerate(t):
            # Find the earliest finishing time
            next_server = np.argmin(finish_time_ac)
            finish_min = finish_time_ac[next_server]
            # Update waiting time
            wait_time += max(finish_min - i, 0)
            # Update finishing time
            finish_time_ac[next_server] = max(finish_min,i) + service_time[j]
            # p[j] = max(finish_min - i, 0)
        
        finish_time[active_ind] = finish_time_ac
        
    # plt.figure()
    # plt.plot(t,p)
    # plt.savefig(str(t.shape[0]) + ".png")
    
    return wait_time, t.shape[0]
