# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:59:11 2020

@author: haixiang_zhang

Cutting-plane method based on random walk
"""

import math
import numpy as np
import scipy as sp
import time
from utils.lovasz import Lovasz, Round, SO
# from utils.subgaussian import RequiredSamples

def RandomWalkSolver(F,params):
    """
    Cutting-plane method via random walk for multi-dim problems.
    """

    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    # sigma = params["sigma"] if "sigma" in params else 1
    eps = params["eps"] if "eps" in params else 1
    delta = params["delta"] if "delta" in params else 1e-6
    L = params["L"] if "L" in params else 1

    # Total number of iterations
    T = math.ceil(d * math.log(L*N/eps, 1.5))
    # Set of points where SO is called and their empirical means
    S = np.zeros((0,d+1))
    print(T)
    
    # Early stopping
    F_old = np.inf
    
    # Start timing
    start_time = time.time()
    # Count simulation runs
    total_samples = 0
    
    # Number of points to approximate covariance (N in the paper)
    M = math.ceil(5 * 10 * d * math.log(d) * max( 10, math.log(d) ))
    # # Number of steps to approximate the uniform measure in P
    # K = math.ceil(d**3 * 2e3)
    
    # Record polytope Ax >= b
    A = np.zeros((0,d))
    b = np.zeros((0,1))
    # Initial analytical center
    z = (N+1)/2 * np.ones((d,))
    # Initial uniform distribution
    y_set = np.random.uniform(1,N,(d,M))
    
    for t in range(T):
        
        # Separation oracle
        so = SO(F,z,eps/4*min(N,2**t+N),delta/4,params)
        c = -so["hat_grad"]
        hat_F = so["hat_F"]
        # Update total samples
        total_samples += so["total"]
        
        # Update A and b
        c = np.reshape(c,(1,d))
        A = np.concatenate((A,c), axis=0)
        b = np.concatenate((b,[[np.sum(c*z)]]),axis=0)
        # print(A)
        
        # Update S
        temp = np.concatenate((z,[hat_F]),axis=0) # (d+1) vector
        temp = np.reshape(temp,(1,d+1)) # 1*(d+1) vector
        S = np.concatenate((S,temp),axis=0)
        print(hat_F)
        
        # Warm-start distribution
        # print(A.shape,y_set.shape,b.shape)
        violation = np.min(A @ y_set - b, axis=0)
        y_set = y_set[:,violation >= 0]
        # Estimate the covarance matrix
        y_bar = np.mean(y_set,axis=1,keepdims=True)
        temp = y_set - y_bar
        Y = np.zeros((d,d))
        for i in range(y_set.shape[1]):
            Y += ( temp[:,i:i+1] @ temp[:,i:i+1].T )
        Y /= y_set.shape[1]
        # print(Y)
        
        # Approximate uniform distribution
        y_set = RandomWalk(y_set,Y,A,b,params)
        
        # Update analytical center
        z = np.mean(y_set[:,:M],axis=1)
        y_set = y_set[:,M:]
        
        # Early stopping
        F_new = np.mean(S[-10:,-1])
        if F_new >= F_old:
            break
        else:
            F_old = F_new
    
    # Find the point with minimial empirical mean
    i = np.argmin(S[:,-1])
    x_bar = S[i,:d]
    
    # Round to an integral solution
    x_opt = Round(F,x_bar,params)["x_opt"]
    # Stop timing
    stop_time = time.time()
    
    return {"x_opt":x_opt, "time":stop_time-start_time, "total":total_samples}

def RandomWalk(y_set,Y,A,b,params,M=None):
    """
    Generate approximate uniform distribution in Ax >= b.
    """
    
    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    
    # Number of points to approximate covariance (N in the paper)
    if M is None:
        M = math.ceil(5 * 10 * d * math.log(d) * max( 10, math.log(d) ))
    # M = 800
    # Number of steps to approximate the uniform measure in P
    K = math.ceil(d**3 * 5e2)
    # K = 4000
    
    # Square root of covariance matrix
    U = sp.linalg.sqrtm(Y)
    
    while y_set.shape[1] < 2*M:        
        # Num of points to be updated
        n = min( 2*M - y_set.shape[1], y_set.shape[1] )
        # Initial points
        y_update = np.copy(y_set[:,np.random.randint(0,y_set.shape[1],(n,))])
        # Ball walk step size
        eta = 1 / math.sqrt(d)
        # Stopping steps
        stop_time = np.random.randint(0,K,(n,))
        # print(M,stop_time.max())
        
        # Each update
        for j in range(np.max(stop_time)+1):
            # Count outside points
            block = np.zeros((n,),dtype=np.int16)
            # Block indices that are larger than stop_time
            block[ stop_time < j ] = 1
            # print(n - block.sum())
            
            while np.sum(block) < n:
                # Indices to be re-selected
                ind = np.where(block == 0)[0]
                num = ind.shape[0]
                # Generate uniform distribution in ball
                u = np.random.randn(d,num)
                norm = np.linalg.norm(u,axis=0,keepdims=True)
                r = eta * np.random.uniform(0,1,(1,num)) ** (1/d)
                # Update step
                y_delta = (U @ u) * r / norm
                # For checking the constraints
                temp = y_update[:,ind] + y_delta
                
                # Block indices with new points inside P
                y_min = np.min(temp,axis=0) - 1
                y_max = N - np.max(temp,axis=0)
                if A.shape[0] > 0:
                    violation = np.min(A @ temp - b, axis=0)
                    check = np.minimum( np.minimum(violation, y_min), y_max )
                else:
                    check = np.minimum(y_min, y_max)
                valid = np.where(check >= 0)[0]
                
                # Update
                block[ind[ valid ]] = 1
                y_update[:,ind[valid]] = temp[:,valid]
            break
                                    
        # Update the set of points
        y_set = np.concatenate((y_set,y_update),axis=1)
            
    return y_set







