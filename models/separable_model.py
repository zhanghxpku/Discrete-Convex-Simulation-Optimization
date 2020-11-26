# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:54:31 2020

@author: haixiang_zhang

Separable convex functions with flat landscape
"""

import numpy as np

def SeparableModel(params):
    
    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    sigma = params["sigma"] if "sigma" in params else 1
    
    # Optimal point
    x_opt = np.random.randint(1+int(0.2*(N-1)), 1+int(0.8*(N-1)), (d,))
    coef = np.random.uniform(0.75,1.25,(d,))

    f = lambda x: np.sum( coef * ( (np.array(x) < x_opt) / (np.array(x)) * (x_opt)\
                         + (np.array(x) >= x_opt) / (N + 1 - np.array(x)) * (N + 1 - x_opt) ) )
    F = lambda x: f(np.array(x)) + sigma * np.random.randn()
    L = max(np.linalg.norm( x_opt ,np.inf ), np.linalg.norm( N+1-x_opt, np.inf ))
    
    opt = x_opt
    
    # Return
    ret = {"F":F, "f":f, "L":L,"x_opt":opt}
    
    return ret