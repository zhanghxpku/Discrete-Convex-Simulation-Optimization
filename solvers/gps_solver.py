# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 20:43:56 2020

@author: haixiang_zhang

Vaidya's cutting-plane method
"""

import math
import numpy as np
from scipy import stats
import time
from utils.lovasz import Lovasz, Round, SO, LovaszCons, RoundCons, SOCons
from utils.subgaussian import RequiredSamples

def GPSSolver(F,params):
    """
    Gaussian process-based search method for multi-dim problems.
    """

    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    sigma = params["sigma"] if "sigma" in params else 1
    eps = params["eps"] if "eps" in params else 1
    delta = params["delta"] if "delta" in params else 1e-6
    
    # Parameters
    r = 70
    s = 10
    a = 3
    b = 1.5
    sig = 2.5
    
    # Start timing
    start_time = time.time()
    # Count simulation runs
    total_samples = 0
    
    # Basic block
    E = np.eye(d)
    E_inv = np.eye(d)
    for i in range(d-1):
        E[i+1,i] = -1
        for j in range(i+1):
            E_inv[i+1,j] = 1

    # Initial points
    x_0 = E_inv @ np.random.uniform(1,N,(d,s))
    # Store existing points
    S = {}
    for i in range(s):        
        S[array2str(x_0[:,i])] = i
        
    # Step 0
    # Store number of evaluations and emporical mean and variance
    records = np.zeros((s,3))
    samples = np.zeros((r,))
    for i in range(s):
        records[i,0] = r
        # Simulation
        for j in range(r):
            samples[j] = F(x_0[:,i])
        records[i,1] = np.mean(samples)
        records[i,1] = np.var(samples)
    
    # Sample-optimal solution
    x_hat_opt = x_0[:, np.argmin(records[:,1])]
    g_hat_opt = np.min(records[:,1])
    
    # Iterate until stopping criterion is satisfised
    while True:
        break

    
    # Round to an integral solution
    # x_opt = Round(F,x_bar,params)["x_opt"]
    x_opt = 0
    
    # Stop timing
    stop_time = time.time()
    
    return {"x_opt":x_opt, "time":stop_time-start_time, "total":total_samples}
    




def ARS(x_0,records,a,b,sig,params):
    """
    The Acceptance-Rejection Algorithm.
    """
    
    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    
    # Current sample-minimum obj value
    min_val = np.min(records[:,1])
    
    # Keep sampling until acceptance
    while True:    
        y = np.cumsum(np.random.uniform(1,N,(d,)))
        # Compute the conditional expectation and variance
        cond_exp, cond_var = CondProb(y,x_0,records,a,b,sig)
        prob = 1 - stats.norm.cdf(min_val, loc=cond_exp, scale=np.sqrt(cond_var))
        if stats.uniform.rvs() <= 2 * prob:
            break
        
    return y


def MCCS(x_0,records,a,b,sig,params):
    """
    The Markov Chain Coordinate Sampling Algorithm.
    """
    
    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    
    # Number of samplings
    T = 200
    
    # Current sample-minimum obj value
    min_val = np.min(records[:,1])
    # Current sample-minimum as the starting point
    y = x_0[:,np.argmin(records[:,1])]
    # Compute the conditional expectation and variance
    cond_exp, cond_var = CondProb(y,x_0,records,a,b,sig)
    prob = 1 - stats.norm.cdf(min_val, loc=cond_exp, scale=np.sqrt(cond_var))
    
    # Keep sampling until acceptance
    for t in range(T):
        # Choose a coordinate
        ind = np.random.randint(0,d)
        # Current coordinate
        if ind == 0:
            cur_val = y[0]
        else:
            cur_val = y[ind] - y[ind - 1]
        # New coordinate
        new_val = np.random.randint(1,d)
        new_val += int(new_val >= cur_val)
        
        # The new point
        z = np.copy(y)
        z[ind:] += (new_val - cur_val)
        # Compute the new conditional expectation and variance
        cond_exp_new, cond_var_new = CondProb(z,x_0,records,a,b,sig)
        prob_new = 1 - stats.norm.cdf(min_val, loc=cond_exp, scale=np.sqrt(cond_var))
        if stats.uniform.rvs() <= prob_new / prob:
            y, prob = z, prob_new
        
    return y


def CondProb(x,x_0,records,a,b,sig):
    """
    The conditional expectation and variance.
    """
    
    # Compute weights
    lamb = Lambda(x,x_0,b)
    gamma = GammaVec(x,x_0,a)
    Gamma = GammaMat(x_0,a)
    Sigma = np.diag( records[:,2] / records[:,0] )
    
    # Conditional expectation
    cond_exp = np.sum(lamb * records[:,1])
    # Conditional variance
    cond_var = (sig**2) * (1 - 2*np.sum(lamb*gamma) + lamb.T @ (Gamma @ lamb))\
                + lamb.T @ (Sigma @ lamb)
    
    return cond_exp, cond_var


def GammaVec(x,x_0,a):
    """
    The gamma vector.
    """
    
    return np.exp(-a * np.linalg.norm(x-x_0.T, axis = -1))
    
    
def GammaMat(x_0,a):
    """
    The gamma matrix.
    """
    
    # The diagonal
    diag = np.diag(x_0.T @ x_0).reshape((1,-1))
    
    return np.exp(-a * np.sqrt( diag + diag.T - 2 * x_0.T @ x_0 ) )


def Lambda(x,x_0,b):
    """
    The lambda function.

    """
    
    diff = x - x_0
    if np.sum(diff == 0) > 0:
        ind = (diff == 0)
        ret = np.zeros(x.shape)
        ret[ind] = 1
    else:
        ret = np.linalg.norm(diff) ** (-b)
        ret = ret / np.sum(ret)

    return ret


def array2str(y):
    """
    Transform a numpy array to a string

    """
    return ",".join([str(z) for z in y])
