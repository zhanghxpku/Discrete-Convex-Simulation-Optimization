# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 22:27:43 2020

@author: haixiang_zhang

The Lovasz extension
"""

import numpy as np
from .subgaussian import RequiredSamples, ConfidenceInterval

def Lovasz(F,x,params):
    """
    Compute the Lovasz extension and its subgradient at point x.
    """
    
    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    
    # Round x to an integral point
    x_int = np.floor(x)
    x_int = np.clip(x_int,-np.inf,N-1)
    # Local coordinate
    x_loc = x - x_int
    # Compute a consistent permuation
    alpha = np.argsort(-x_loc)
    
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
        lov += sub_grad[alpha[i]] * x_loc[alpha[i]]
        
        # Update neighboring point
        x_old = x_new
    
    return lov, sub_grad

def Round(F,x,params):
    """
    Round (eps/2,delta/2) solution x to integral (eps,delta) solution.
    """
    
    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    # sigma = params["sigma"] if "sigma" in params else 1
    eps = params["eps"] if "eps" in params else 1
    delta = params["delta"] if "delta" in params else 1e-6
    
    # Round x to an integral point
    x_int = np.floor(x)
    x_int = np.clip(x_int,-np.inf,N-1)
    # Local coordinate
    x_loc = x - x_int
    # Compute a consistent permuation
    alpha = np.argsort(-x_loc)
    
    # Neighboring points
    x_nei = np.zeros((d+1,d))
    x_nei[0,:] = x_int
    for i in range(1,d+1):
        x_nei[i,:] = np.copy(x_nei[i-1,:])
        x_nei[i,alpha[i-1]] += 1
    
    # Number of samples needed
    num_samples = RequiredSamples(delta/4,eps/4,params)
    # Record empirical mean
    hat_F = np.zeros((d+1,))
    # Record number of samples
    total_samples = 0
    # Stop simumating if already too large
    blocked = np.zeros((d+1,))
    
    # Simulation
    for i in range(num_samples):
        for j in range(d+1):
            if blocked[j] == 0:
                hat_F[j] = ( hat_F[j] * i + F(x_nei[j,:]) ) / (i + 1)
        # Update total samples
        total_samples += (d + 1 - np.sum(blocked))
        
        # Check confidence interval
        CI = ConfidenceInterval(delta/4,params,i+1)
        # Block points with large empirical means
        blocked[ hat_F - np.min(hat_F) > 2 * CI ] = 1
        # print(hat_F)
        # Only one point left
        if np.sum(blocked) == d:
            break
    
    # Return the point with the minimal empirical mean
    x_opt = x_nei[np.argmin(hat_F),:]
    
    return {"x_opt":x_opt, "total":total_samples}
    

def SO(F,x,eps,delta,params):
    """
    Compute an (eps,delta)-SO oracle and empirical obj value at x.
    """
    
    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    # sigma = params["sigma"] if "sigma" in params else 1
    
    # Number of samples needed
    num_samples = RequiredSamples(delta,eps/2/N/np.sqrt(d),params)
    
    # Record empirical mean and empirical subgradient
    hat_F = 0
    hat_grad = np.zeros((d,))
    
    # Simulate
    for t in range(num_samples):
        lov, grad = Lovasz(F, x, params)
        
        # Update
        hat_F = ( hat_F * t + lov ) / (t + 1)
        hat_grad = ( hat_grad * t + grad ) / (t + 1)
        
    # Return
    return { "hat_F":hat_F, "hat_grad":hat_grad, "total":num_samples*d }
    
    
    
    
    
    
    
    
    
    
    