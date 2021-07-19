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

def NewsModel(params):
    """
    Generate the objective function
    """
    
    # # Retrieve parameters: upper bound of time
    # tau = params["N"] - 1 if "N" in params else 100
    # Retrieve parameters: number of products
    d = params["d"] if "d" in params else 1
    
    # Net profit
    # profit = np.zeros((d+1,))
    profit = 0.5 * np.linspace(1,d,d)
    # Stocking price
    p = 0.3
    # Parameters
    a = 10
    b = 5
    
    # The optimal decision
    x_opt = np.ones((d,))
    x_opt[-1] = np.floor(b - (b-a+1) * p / profit[-1] ) + 1
        
    return {"F": lambda y: Profit(y,params,profit) + p*np.sum(y),
            "N":x_opt[-1],"x_opt":x_opt}


def Profit(x,params,profit):
    """
    Compute the total profit
    """
    
    # # Retrieve parameters: upper bound of time
    # tau = params["N"] + 1 if "N" in params else 100
    # Retrieve parameters: number of products
    d = params["d"] if "d" in params else 1
    # Parameters
    a = 10
    b = 5
    mu = 1.0
    
    # Number of customers
    N = np.random.randint(a,b*d+1)
    
    # Shift to the true stock value
    y = x - 1
    # Total number of iterations
    num_iter = min(N, int(np.sum(y)))
    
    # Iterate for each arrival
    for i in range(num_iter):
        # Compute the probability
        prob = np.exp( profit / mu ) * (y > 0)
        prob /= np.sum(prob)
        # Choose a random product
        ind = np.random.choice(np.arange(d), p=prob)
        y[ind] -= 1
    
    # Return the accumulated profit
    return -np.sum( profit * ( x - 1 - y ) )
        
    
    
    
        