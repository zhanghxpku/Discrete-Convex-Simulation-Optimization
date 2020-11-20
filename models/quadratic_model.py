# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 22:38:26 2020

@author: haixiang_zhang

Generate qurdratic L-convex examples.
"""

import numpy as np

def QuadraticModel(params):
    
    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    sigma = params["sigma"] if "sigma" in params else 1
    
    # Generate a L-convex function
    A = - np.abs(np.random.randn(d,d))
    A = A + A.T
    
    for i in range(d):
        A[i,i] = - np.sum(A[i,i+1:i+d]) + np.abs(np.random.randn(1,1))
    
    b = A @ np.random.randint(2,2*N,(d,))
    
    # ell_inf Lipschitz constant
    L = 2 * np.sqrt(d) * N * np.linalg.norm(A,2) + np.linalg.norm(b)
    
    # Objective function
    f = lambda x: np.sum( x * (A @ x) ) - np.sum(b * x)
    F = lambda x: f(x) + sigma * np.random.randn()
    
    # Return
    ret = {"F":F, "f":f, "L":L}
    
    return ret





