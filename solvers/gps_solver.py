# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 20:43:56 2020

@author: haixiang_zhang

Gaussian Process-based Search (GPS) method
"""

import math
import numpy as np
from scipy import stats
import time
# from utils.lovasz import Lovasz, Round, SO, LovaszCons, RoundCons, SOCons
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
    r = 1000
    # r = RequiredSamples(delta/2,eps/4,params)
    s = 100
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
        records[i,2] = np.var(samples)
    # Update total number of samples
    total_samples += r * s

    # Sample-optimal solution
    # x_hat_opt = x_0[:, np.argmin(records[:,1])]
    g_hat_opt = np.min(records[:,1])
    count = 0
    
    # Iterate until stopping criterion is satisfised
    while True:
        # Steps 1&2
        samples_new = np.zeros((r,))
        for i in range(s):
            # Generate new samples
            x_new = MCCS(x_0,records,a,b,sig,params)
            # Simulate on new samples
            for j in range(r):
                samples_new[j] = F(x_new)
        
            # Step 3: Update the mean and variance
            if array2str(x_new) in S:
                ind = S[array2str(x_new)]
                # Update the variance
                records[ind,2] = (records[ind,2] + records[ind,1]**2) * records[ind,0]
                records[ind,2] = (records[ind,2] + np.sum(samples_new**2))\
                                 / (records[ind,0] + r)
                # Update the mean
                records[ind,1] = (records[ind,1]*records[ind,0] + np.sum(samples_new))\
                                 / (records[ind,0] + r)
                # Update the variance
                records[ind,2] -= records[ind,1] ** 2
                # Update the number of samples
                records[ind,0] += r
                
            else:
                S[array2str(x_new)] = x_0.shape[1]
                x_0 = np.concatenate((x_0,x_new.reshape((d,1))),axis=1)
                records = np.concatenate((records,np.zeros((1,3))),axis=0)
                records[-1,0] = r
                records[-1,1] = np.mean(samples_new)
                records[-1,2] = np.var(samples_new)
        
        # Update total number of samples
        total_samples += r * s
        
        # Check the stopping criterion
        g_hat_opt_new = np.min(records[:,1])
        print(g_hat_opt_new,len(S))
        if g_hat_opt_new > g_hat_opt - 2*eps / d / np.sqrt(N):
            count += 1
        if count > 10:
            break
        g_hat_opt = g_hat_opt_new
    
    # Round to an integral solution
    x_opt = x_0[:, np.argmin(records[:,1])]
    
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
        prob = stats.norm.cdf(min_val, loc=cond_exp, scale=np.sqrt(cond_var))
        if stats.uniform.rvs() <= 2 * prob:
            break
        
    return y


def MCCS(x_0,records,a,b,sig,params):
    """
    The Markov Chain Coordinate Sampling Algorithm.
    """
    
    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    # Number of steps
    T = 200
    
    # Current sample-minimum obj value
    min_val = np.min(records[:,1])
    # Current sample-minimum as the starting point
    y = x_0[:,np.argmin(records[:,1])]
    # Compute the conditional expectation and variance
    cond_exp, cond_var = CondProb(y,x_0,records,a,b,sig)
    prob = stats.norm.cdf(min_val, loc=cond_exp, scale=np.sqrt(cond_var))
    
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
        prob_new = stats.norm.cdf(min_val, loc=cond_exp, scale=np.sqrt(cond_var))
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
    
    if cond_var < 0:
        print(lamb, records)
        # print(np.sum(lamb*gamma), lamb.T @ (Gamma @ lamb), lamb.T @ (Sigma @ lamb) )
    
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
    dist = diag + diag.T - 2 * x_0.T @ x_0
    dist -= np.min(diag + diag.T - 2 * x_0.T @ x_0)
    # if np.min(diag + diag.T - 2 * x_0.T @ x_0) < 0:
    #     print(np.min(diag + diag.T - 2 * x_0.T @ x_0))
    
    return np.exp(-a * np.sqrt( dist ) )


def Lambda(x,x_0,b):
    """
    The lambda function.

    """
    
    diff = np.linalg.norm(x - x_0.T, axis=-1)
    if np.sum(diff == 0) > 0:
        ind = (diff == 0)
        ret = np.zeros((x_0.shape[1],))
        ret[ind] = 1
    else:
        ret = diff ** (-b)
        ret = ret / np.sum(ret)

    return ret


def array2str(y):
    """
    Transform a numpy array to a string

    """
    return ",".join([str(z) for z in y])
