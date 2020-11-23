# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 20:43:56 2020

@author: haixiang_zhang

Vaidya's cutting-plane method
"""

import math
import numpy as np
import time
from utils.lovasz import Lovasz, Round, SO
from utils.subgaussian import RequiredSamples

def VaidyaSolver(F,params):
    """
    Vaidya's cutting-plane method for multi-dim problems.
    """

    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    # sigma = params["sigma"] if "sigma" in params else 1
    eps = params["eps"] if "eps" in params else 1
    delta = params["delta"] if "delta" in params else 1e-6
    L = params["L"] if "L" in params else 1
    
    # Set parameters
    p = 1e-4 # eps in paper
    q = 1e-3 * p # delta in paper
    
    # Total number of iterations
    T = d * math.log(d*L*N/eps) / q * 2
    # Set of points where SO is called and their empirical means
    S = np.zeros((0,d+1))
    print(T)
    
    # Start timing
    start_time = time.time()
    # Count simulation runs
    total_samples = 0
    
    # Initial polytope Ax >= b
    A = np.diag(np.concatenate([np.ones(d,),-np.ones((d,))]))
    b = np.concatenate([np.ones(d,),-N * np.ones((d,))])
    # Initial volumetric center
    z = (N+1)/2 * np.ones((d,))
    # Initial Hessian
    H_inv,alpha,_,_ = Auxiliary(z,A,b)
    
    for t in range(T):
        
        # Case I
        if np.min(alpha) >= q:
            # Separation oracle
            so = SO(eps/8,delta/4,params)
            c = so["hat_grad"]
            hat_F = so["hat_F"]
            beta = np.sum(c*z) + math.sqrt( 2*(c.T @ H_inv) @ c\
                                           / math.sqrt(p*q) )
            
            # Update total samples
            total_samples += so["total"]
            
            # Update A and b
            c = np.reshape(c,(1,d))
            A = np.concatenate((A,c), axis=0)
            b = np.concatenate((b,[beta]))
            
            # Update matrices
            H_inv,alpha,nabla,Q = Auxiliary(z,A,b)
            
            # Update S
            temp = np.concatenate((c,[[hat_F]])) # 1*(d+1) vector
            S = np.concatenate((S,temp),axis=0)
            
            # Update volumetric center
            # Number of Newton steps
            num_newton = math.ceil( 30 * math.log( 2 / (q**4.5) ) )
            for _ in range(num_newton):
                z -= (0.18 * np.linalg.solve(Q,nabla))
        # Case II
        else:
            # Find the cutting-plane to be removed
            i = np.argmin(alpha)
            
            # Update A, b, S
            A = np.delete(A,i,axis=0)
            b = np.delete(b,i,axis=0)
            S = np.delete(S,i,axis=0)
            
            # Update volumetric center
            # Number of Newton steps
            num_newton = math.ceil( 30 * math.log( 4 / (q**3) ) )
            for _ in range(num_newton):
                z -= (0.18 * np.linalg.solve(Q,nabla))
    
    # Find the point with minimial empirical mean
    i = np.argmin(S[:,-1])
    x_bar = S[i,:d]
    
    # Round to an integral solution
    x_opt = Round(F,x_bar)
    
    # Stop timing
    stop_time = time.time()
    
    return {"x_opt":x_opt, "time":stop_time-start_time, "total":total_samples}
    
def Auxiliary(x,A,b):
    """
    Returns H^-1(x), sigma(x), nabla F(x), Q(x).
    """
    
    # Get the shape of A
    m, d = A.shape
    
    H_x = np.zeros((d,d))
    alpha_x = np.zeros((m,))
    nabla_x = np.zeros((d,))
    Q_x = np.zeros((d,d))
    
    v_x = 1 / ( A@x - b )
    v_x = v_x.reshape((m,1))
    u_x = v_x * A
    
    # Compute H(x)
    for i in range(m):
        H_x += u_x[i,:].T @ u_x[i,:]
    # Inverse of H(x)
    H_inv = np.linalg.inv(H_x)
    
    # Compute other functions
    for i in range(m):
        alpha_x[i] = u_x[i,:] @ (H_inv @ u_x[i,:].T)
        temp = alpha_x[i] * u_x[i,:].T
        nabla_x -= temp
        Q_x += temp @ u_x[i,:]
    
    return H_inv, alpha_x, nabla_x, Q_x

    
    



















