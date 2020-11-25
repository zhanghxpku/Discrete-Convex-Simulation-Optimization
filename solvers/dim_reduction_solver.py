# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:58:20 2020

@author: haixiang_zhang

Dimension reduction method
"""

import math
import numpy as np
import scipy as sp
import time
from utils.lovasz import Lovasz, Round, SO
from utils.lll import LLL, Projection
from utils.subgaussian import RequiredSamples
from .random_walk_solver import RandomWalk

# import gurobipy as gp
# from gurobipy import GRB

def DimensionReductionSolver(F,params):
    """
    Cutting-plane method via random walk for multi-dim problems.
    """

    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    # sigma = params["sigma"] if "sigma" in params else 1
    eps = params["eps"] if "eps" in params else 1
    delta = params["delta"] if "delta" in params else 1e-6
    
    # # Parameters of the algorithm
    # eta = 1e-2 # eps in paper
    
    # Start timing
    start_time = time.time()
    # Count simulation runs
    total_samples = 0
    
    # The current dimension
    d_cur = d
    # The basis of integral lattice
    L = np.eye(d)
    
    # Record polytope Ax >= b
    A = np.zeros((0,d))
    b = np.zeros((0,1))
    # The initial uniform distribution in P
    y_set = np.random.uniform(1,N,(d,400))
    # The initial ellipsoid
    C_inv = (N-1)**(-2) * 12 * np.eye(d) # A in paper
    C = (N-1)**2 / 12 * np.eye(d) # A_inv in paper
    # Initial centroid and inner center
    z_k = (N+1)/2 * np.ones((d,))
    z_in = (N+1)/2 * np.ones((d,))
    
    # Iteratively solve d_cur-dimensional problems
    while d_cur > 1:
        
        # Simulation cost of each separation oracle
        so_samples = RequiredSamples(delta,eps/4/np.sqrt(d_cur),params)
        # The current basis
        L_cur = L[d-d_cur:,:]
        
        # Number of samples in the initialization of random walk method
        total_samples += so_samples
        # Iterate until we find a short basis vector
        # Use random walk method
        for K,z_k,A_new,b_new, y_new in RandomWalkApproximator(F,L_cur,C,y_set,
                                                               A,b,params):
            # Number of samples
            total_samples += so_samples

            # The LLL algorithm
            basis = LLL(L_cur,K)
            # Choose the shortest vector
            norm = np.diag( (basis @ K) @ basis.T )
            print(np.min(norm))

            # Stopping criterion
            if np.min(norm) < 1e-2 / d**2 / 2**(2*d):
                i_short = np.argmin( norm )
                basis[[0,i_short],:] = basis[[i_short,0],:]
                # Update the basis and point set
                y_set = y_new
                L_cur = basis
                # Update A and b
                A = A_new
                b = b_new
                # Update inner center
                z_in = z_k
                break
            
        # Dimension reduction
        z = L[d-d_cur,:]
        # Find the pre-image
        for j in range(d-d_cur-1,-1,-1):
            # Projection direction
            v = L[j,:]
            
            # Solve for alpha
            # Create a new model
            model = gp.Model("alpha")
            model.Params.OutputFlag = 0 # Controls output
            # model.Params.MIPGap = 1e-9
            # Variables
            x = model.addVars(range(d-j+1), vtype=GRB.INTEGER, name="x")
            alpha = model.addVar(vtype=GRB.CONTINUOUS, name="alpha")
            # Add constraints
            model.addConstrs(
                (z[i] + alpha*v[i] == 
                 gp.quicksum(L[k,i] * x[k-j] for k in range(j,d) )
                 for i in range(d)),
                name="c1")
            model.addConstr( alpha <=  0.5, name='c2')
            model.addConstr( alpha >= -0.5, name='c3')
            # Set the objective function as constant
            model.setObjective(0, GRB.MAXIMIZE)
            # Solve the feasibility problem
            model.optimize()
            alpha = alpha.X
            
            z += alpha * v
                
                
        
        
        
        
        
        
        
        
        
        
    
    # Solve the one-dimensional problem
    x_opt = 0
    
    # Stop timing
    stop_time = time.time()
    
    return {"x_opt":x_opt, "time":stop_time-start_time, "total":total_samples}


def RandomWalkApproximator(F,L,Y,y_in,A_in,b_in,params):
    """
    Output the new polytope, its approximate centeriod and approximate covariance matrix.
    
    L: lattice basis
    Y: approximate covariance matrix
    y_set: warm-up distribution in P
    A_in, b_in: polytope
    """
    
    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    eps = params["eps"] if "eps" in params else 1
    delta = params["delta"] if "delta" in params else 1e-6
    A = np.copy(A_in)
    b = np.copy(b_in)
    y_set = np.copy(y_in)
        
    # # Generate a warm-start distribution with 2M points
    # # M = math.ceil(5 * 10 * d * math.log(d) * max( 10, math.log(d) ))
    # M = 256
    # # Initial point
    # y_set = np.reshape(x,(d,1))
    # y_set = np.random.uniform(1,N,(d,400))
    # # Warm-start distribution
    # y_set = RandomWalk(y_set,Y,A,b,params,M=int(1e5))
    # y_set = y_set[ :,-int(2e3): ]
    # # Refine centroid and covariance
    # y_bar = np.mean(y_set,axis=1,keepdims=True)
    # temp = y_set - y_bar
    # Y = np.zeros((d,d))
    # for i in range(y_set.shape[1]):
    #     Y += ( temp[:,i:i+1] @ temp[:,i:i+1].T )
    # Y /= y_set.shape[1]
    # print(y_bar,Y)
    
    # Generate the uniform distribution in P
    y_set = RandomWalk(y_set,Y,A,b,params)    
    # Approximate centroid
    y_bar = np.mean(y_set,axis=1,keepdims=True)
        
    # Constantly generate polytopes
    while True:
        
        # Separation oracle
        so = SO(F,y_bar[:,0],eps/4*min(N,N),delta/4,params)
        c = -so["hat_grad"]
        # hat_F = so["hat_F"]
        
        # Update A and b
        c = np.reshape(c,(1,d))
        A = np.concatenate((A,c), axis=0)
        b = np.concatenate((b,c@y_bar),axis=0)
        
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
        
        # Update uniform distribution in P
        y_set = RandomWalk(y_set,Y,A,b,params)
        
        # Approximate centroid and covariance
        M = int(y_set.shape[1] / 2)
        y_bar = np.mean(y_set[:,:M],axis=1,keepdims=True)
        temp = y_set[:,:M] - y_bar
        Y = np.zeros((d,d))
        for i in range(M):
            Y += ( temp[:,i:i+1] @ temp[:,i:i+1].T )
        Y /= M
        
        # Update point set
        y_set = y_set[:,M:]
        # print(y_bar, Y)
        
        # Output
        yield Y, y_bar[:,0], A, b, y_set
        
        
        


