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
    # print(T)
    
    # Early stopping
    F_old = np.inf
    
    # Start timing
    start_time = time.time()
    # Count simulation runs
    total_samples = 0
    
    # Number of points to approximate covariance (N in the paper)
    M = math.ceil(50 * 20 * d * math.log(d+1) * max( 20, math.log(d) ))
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
        # print(z)
        so = SO(F,z,eps/8*min(N,N),delta/4,params)
        c = -so["hat_grad"]
        # print(c)
        # c = - np.ones((d,))
        # c[0] = -1
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
        # print(hat_F)
        # print(np.sum(c*x_opt),np.sum(c*z))
        
        # Warm-start distribution
        # print(A.shape,y_set.shape,b.shape)
        # print(y_set.shape)
        violation = np.min(A[-1:,:] @ y_set - b[-1:,:], axis=0)
        y_set = y_set[:,violation >= 0]
        # print(y_set.shape)
        # Estimate the covarance matrix
        y_bar = np.mean(y_set,axis=1,keepdims=True)
        temp = y_set - y_bar
        Y = np.zeros((d,d))
        for i in range(y_set.shape[1]):
            Y += ( temp[:,i:i+1] @ temp[:,i:i+1].T )
        Y /= y_set.shape[1]
        # print(Y)
        
        # Approximate uniform distribution
        y_set = RandomWalk(y_set,Y,A,b,params,M)
        
        # Update analytical center
        y_set = y_set[:,np.random.permutation(np.arange(2*M))]
        z = np.mean(y_set[:,:M],axis=1)
        y_set = y_set[:,M:]
        
        # Early stopping
        F_new = np.mean(S[-3:,-1])
        if F_new >= F_old - 2*eps / d / np.sqrt(N):
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
        M = math.ceil(50 * 20 * d * math.log(d+1) * max( 20, math.log(d) ))
    # M = 800
    # Number of steps to approximate the uniform measure in P
    K = math.ceil(d**3 * 2e3)
    # print(K)
    # K = 4000
    m = y_set.shape[1]
    # Square root of covariance matrix
    U = np.real(sp.linalg.sqrtm(Y))
    # print(np.linalg.eigvalsh(Y))
    # print(U[:3,:3])
    
    while y_set.shape[1] < 2*M + m:
        # print(y_set.shape[1],2*M+m)
        # Num of points to be updated
        n = min( 2*M + m - y_set.shape[1], y_set.shape[1] )
        # Initial points
        y_update = np.copy(y_set[:,np.random.randint(0,y_set.shape[1],(n,))])
        # Ball walk step size
        eta = 1 / math.sqrt(d)
        # Stopping steps
        stop_time = np.random.randint(0,K,(n,))
        stop_time = K * np.ones((n,), dtype=np.int32)
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
                u *= (r / norm)
                # Update step
                y_delta = U @ u
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
        # # Update covariance matrix
        # Y = np.zeros((d,d))
        # # Estimate the covarance matrix
        # y_bar = np.mean(y_set,axis=1,keepdims=True)
        # temp = y_set - y_bar
        # for i in range(y_set.shape[1]):
        #     Y += ( temp[:,i:i+1] @ temp[:,i:i+1].T )
        # Y /= y_set.shape[1]
        # # print(Y)
            
    return y_set[:,m:]

def RandomWalkProjSolver(F,params):
    """
    Cutting-plane method via random walk for multi-dim problems with capacity constraint.
    """

    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    # sigma = params["sigma"] if "sigma" in params else 1
    eps = params["eps"] if "eps" in params else 1
    delta = params["delta"] if "delta" in params else 1e-6
    L = params["L"] if "L" in params else 1
    # The capacity constraint
    K = params["K"] if "K" in params else N * d

    # Total number of iterations
    T = math.ceil(d * math.log(L*N/eps, 1.5))
    # Set of points where SO is called and their empirical means
    S = np.zeros((0,d+1))
    # print(T)
    
    # Early stopping
    F_old = np.inf
    
    # Start timing
    start_time = time.time()
    # Count simulation runs
    total_samples = 0
    
    # Number of points to approximate covariance (N in the paper)
    M = math.ceil(50 * 20 * d * math.log(d+1) * max( 20, math.log(d) ))
    # # Number of steps to approximate the uniform measure in P
    # K = math.ceil(d**3 * 2e3)
    
    # Record polytope Ax >= b
    # Basic block
    E = np.eye(d)
    for i in range(d-1):
        E[i+1,i] = -1
    A = np.zeros((2*d+1,d))
    A[:d,:] = E
    A[d:2*d,:] = -E
    A[-1,-1] = -1
    # b = np.zeros((0,1))
    b = np.concatenate((np.ones((d,1))*(1+1e-7),
                        np.ones((d,1))*(-N+1e-7),
                        [[-K]]
                        ))
    # Initial analytical center
    x = np.ones((d,)) * (N+1) / 4
    z = np.cumsum(x)
    # Initial uniform distribution
    y_set = np.random.uniform(1,N,(d,M))
    
    for t in range(T):
        
        # Separation oracle
        print(z)
        so = SO(F,z,eps/8*min(N,N)*800,delta/4,params)
        c = -so["hat_grad"]
        # print(c)
        # c = - np.ones((d,))
        # c[0] = -1
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
        # print(np.sum(c*x_opt),np.sum(c*z))
        
        # Warm-start distribution
        # print(A.shape,y_set.shape,b.shape)
        # print(y_set.shape)
        violation = np.min(A[-1:,:] @ y_set - b[-1:,:], axis=0)
        y_set = y_set[:,violation >= 0]
        # print(y_set.shape)
        # Estimate the covarance matrix
        y_bar = np.mean(y_set,axis=1,keepdims=True)
        temp = y_set - y_bar
        Y = np.zeros((d,d))
        for i in range(y_set.shape[1]):
            Y += ( temp[:,i:i+1] @ temp[:,i:i+1].T )
        Y /= y_set.shape[1]
        # print(Y)
        
        print("here!")
        # Approximate uniform distribution
        y_set = RandomProjWalk(y_set,Y,A,b,params,M)
        print("here!!")
        
        # Update analytical center
        y_set = y_set[:,np.random.permutation(np.arange(2*M))]
        z = np.mean(y_set[:,:M],axis=1)
        y_set = y_set[:,M:]
        
        # Early stopping
        F_new = np.mean(S[-3:,-1])
        if F_new >= F_old - 2*eps / d / np.sqrt(N):
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

def RandomProjWalk(y_set,Y,A,b,params,M=None):
    """
    Generate approximate uniform distribution in Ax >= b.
    """
    
    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    # The capacity constraint
    K = params["K"] if "K" in params else N * d
    
    # Number of points to approximate covariance (N in the paper)
    if M is None:
        M = math.ceil(50 * 20 * d * math.log(d+1) * max( 20, math.log(d) ))
    # M = 800
    # Number of steps to approximate the uniform measure in P
    K = math.ceil(d**3 * 2e3)
    # print(K)
    # K = 4000
    m = y_set.shape[1]
    # Square root of covariance matrix
    U = np.real(sp.linalg.sqrtm(Y))
    # print(np.linalg.eigvalsh(Y))
    # print(U[:3,:3])
    
    while y_set.shape[1] < 2*M + m:
        # print(y_set.shape[1],2*M+m)
        # Num of points to be updated
        n = min( 2*M + m - y_set.shape[1], y_set.shape[1] )
        # Initial points
        y_update = np.copy(y_set[:,np.random.randint(0,y_set.shape[1],(n,))])
        # Ball walk step size
        eta = 1 / math.sqrt(d)
        # Stopping steps
        stop_time = np.random.randint(0,K,(n,))
        stop_time = K * np.ones((n,), dtype=np.int32)
        print(M,stop_time.max())
        
        # Each update
        for j in range(np.max(stop_time)+1):
            # Count outside points
            block = np.zeros((n,),dtype=np.int16)
            # Block indices that are larger than stop_time
            block[ stop_time < j ] = 1
            # print(n - block.sum())
            
            while np.sum(block) < n:
                print(np.sum(block),n)
                # Indices to be re-selected
                ind = np.where(block == 0)[0]
                num = ind.shape[0]
                # Generate uniform distribution in ball
                u = np.random.randn(d,num)
                norm = np.linalg.norm(u,axis=0,keepdims=True)
                r = eta * np.random.uniform(0,1,(1,num)) ** (1/d)
                u *= (r / norm)
                # Update step
                y_delta = U @ u
                # For checking the constraints
                temp = y_update[:,ind] + y_delta
                
                # Block indices with new points inside P
                y_min = np.min(temp,axis=0) - 1
                y_max = K - np.max(temp,axis=0)
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
        # # Update covariance matrix
        # Y = np.zeros((d,d))
        # # Estimate the covarance matrix
        # y_bar = np.mean(y_set,axis=1,keepdims=True)
        # temp = y_set - y_bar
        # for i in range(y_set.shape[1]):
        #     Y += ( temp[:,i:i+1] @ temp[:,i:i+1].T )
        # Y /= y_set.shape[1]
        # # print(Y)
            
    return y_set[:,m:]
