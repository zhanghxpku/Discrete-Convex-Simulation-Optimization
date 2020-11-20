# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 22:27:43 2020

@author: haixiang_zhang

The Lovasz extension
"""

import numpy as np

# Compute the Lovasz extension and its subgradient at point x
def Lovasz(F,x,params):
    
    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    
    # Round x to an integral point
    x_int = np.floor(x)
    x_int = np.clip(x_int,-np.inf,N-1)
    # Local coordinate
    x_loc = x - x_int
    
    # Compute a consistent permuation
    temp = sorted(enumerate(-x_loc), key=lambda x:x[1])
    alpha = np.array([ ind[0] for ind in temp ])
    
    # Compute the Lovasz extension and its subgradient
    lov = F(x_int)
    sub_grad = np.zeros((d,))
    
    # The 0-th neighboring point
    x_old = np.copy(x_int)
    for i in range(d):
        # The i-th neighboring point
        x_new = np.copy(x_old)
        x_new[alpha[i]] += 1
        
        sub_grad[alpha[i]] = F(x_new) - F(x_old)
        lov += (F(x_new) - F(x_old)) * x_loc[alpha[i]]
        
        # Update neighboring point
        x_old = x_new
    
    return lov, sub_grad

