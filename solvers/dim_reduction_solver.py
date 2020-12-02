# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:58:20 2020

@author: haixiang_zhang

Dimension reduction method
"""

import math
import numpy as np
import time
from utils.lovasz import Round, SO
from utils.lll import LLL
from utils.subgaussian import RequiredSamples
from .random_walk_solver import RandomWalk
from .uniform_solver import UniformSolver
from hsnf import column_style_hermite_normal_form

import gurobipy as gp
from gurobipy import GRB

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
    # Record points where SO is called and their empirical means
    S = np.zeros((d+1,0))
    # Simulation cost of each separation oracle
    so_samples = RequiredSamples(delta/4,eps/8/np.sqrt(d),params)
    
    # The current dimension
    d_cur = d
    # The basis of integral lattice
    L = np.eye(d)
    # The pre-images
    Z = np.zeros((d,d))
    
    # Record polytope Ax >= b
    # A = np.concatenate((np.eye(d),-np.eye(d)),axis=0)
    # b = np.concatenate((np.ones((d,1),-N*np.ones(d,1))),axis=0)
    A = np.zeros((0,d))
    b = np.zeros((0,1))
    # The initial uniform distribution in P
    y_set = np.random.uniform(1,N,(d,400))
    # The initial ellipsoid
    # C_inv = (N-1)**(-2) * 12 * np.eye(d) # A in paper
    C = (N-1)**2 / 12 * np.eye(d) # A_inv in paper
    # Initial centroid
    z_k = (N+1)/2 * np.ones((d,))
    
    # Early stopping
    early_stop = False
    # Iteratively solve d_cur-dimensional problems
    while d_cur > 1:
        
        # print(d_cur)
        # The current basis
        L_cur = np.copy(L[d-d_cur:,:])
        
        # Iterate until we find a short basis vector
        # Use random walk method
        for K,z_new,A_new,b_new,y_new,s in RandomWalkApproximator(F,C,y_set,A,
                                                                  b,params):
            # Check early stopping
            if K is None:
                early_stop = True
                break
            
            u = np.linalg.eigvalsh(K)
            # print(u)
            if np.sum(u > 1e-10) < d_cur:
                break
            
            # Number of samples
            total_samples += so_samples*(2*d)
            # Update set S
            S = np.concatenate((S,s),axis=1)
            # print(s[-1,0])
            # print(K)

            # The LLL algorithm
            basis = LLL(L_cur,K)
            # print(basis)
            # print(K)
            # Choose the shortest vector
            norm = np.diag( (basis @ K) @ basis.T )
            # print(np.min(norm))
            # print(norm)

            # Stopping criterion
            if np.min(norm) < 1e0 / d**2:
            # if np.min(norm) < 300:
                i_short = np.argmin( norm )
                L_cur[0,:] = basis[i_short,:]
                # basis[[0,i_short],:] = basis[[i_short,0],:]
                # Update the basis and point set
                y_set = y_new
                # Update A and b
                A, b = A_new, b_new
                # Update centroid
                z_k = z_new
                break
        
        # # Check if intersection is empty
        # if early_stop:
        #     break
        
        # Dimension reduction
        # L[d-d_cur:,:] = L_cur
        v = L_cur[0,:]
        # L[d-d_cur+1:,:] = (L[d-d_cur+1:,:].T\
        #                    / np.linalg.norm(L[d-d_cur+1:,:],axis=1)).T
        # print(L[d-d_cur+1:,:] @ v.reshape((d,1)))
        # # Normalize the basis
        # min_val = np.zeros((d_cur,))
        # for i in range(d_cur):
        #     min_val[i] = np.min(np.abs(L_cur[i,np.abs(L_cur[i,:])>1e-8]))
        # L[d-d_cur:,:] = (L[d-d_cur:,:].T / min_val).T
        
        # print(v,L)
        # Find the pre-image
        if d_cur == d:
            z = v
            Z[0,:] = v
        else:
            # Solve for z
            # Create a new model
            model = gp.Model("pre-image")
            model.Params.OutputFlag = 0 # Controls output
            # model.Params.MIPGap = 1e-9
            # Variables
            x = model.addVars(range(2*d), vtype=GRB.INTEGER, name="x")
            # alpha = model.addVar( vtype = GRB.CONTINUOUS, name="alpha" )
            # Add constraints
            # model.addConstrs(
            #     (gp.quicksum( (x[k]-x[d+k])*L[j,k] for k in range(d) ) == 0
            #       for j in range(d-d_cur)),
            #     name="c1")
            model.addConstrs(
                (gp.quicksum( (x[k]-x[d+k])*L[j,k] for k in range(d) )\
                 == gp.quicksum( v[k]*L[j,k] for k in range(d) )
                for j in range(d-d_cur,d)),
                name="c3")
            # model.addConstr(alpha <= 1)
            # # Set initial point
            # for i in range(d):
            #     x[i].start = max(v[i],0)
            #     x[i+d].start = -min(v[i],0)
            # model.update()
            # Set the objective function as constant
            model.setObjective(0, GRB.MAXIMIZE)
            # model.setObjective(gp.quicksum((x[k]-x[d+k]-v[k])*(x[k]-x[d+k]-v[k]) for k in range(d)),
            #                    GRB.MINIMIZE)
            # Solve the feasibility problem
            model.optimize()
            
            z = np.zeros((d,))
            for i in range(d):
                z[i] = x[i].X - x[i+d].X
            # print(z)
            Z[d-d_cur,:] = z
            # print(Z)
            # try:
            #     for i in range(d):
            #         z[i] = x[i].X
            # except AttributeError:
            #     early_stop = True
            #     break
        
        # Construct the hyperplane v^T y = v_y
        v_y = np.sum( (v-z)*z_k ) + round( np.sum( z * z_k ) )
        # print(v,v_y)
        
        # Update the point set
        y_set = y_set - v.reshape((d,1)) @ ((v.reshape((d,1)).T @ y_set - v_y)\
                                            / np.sum(v*v))
        # Remove outside points
        y_min = np.min(y_set,axis=0) - 1
        y_max = N - np.max(y_set,axis=0)
        violation = np.min(A @ y_set - b, axis=0)
        check = np.minimum( np.minimum(violation, y_min), y_max )
        y_set = y_set[:,check >= 0]
        # print(y_set.shape)
        
        # Check if intersection is empty
        if y_set.shape[1] == 0:
            early_stop = True
            break
        
        # Estimate the centroid covarance matrix
        y_bar = np.mean(y_set,axis=1,keepdims=True)
        temp = y_set - y_bar
        Y = np.zeros((d,d))
        for i in range(y_set.shape[1]):
            Y += ( temp[:,i:i+1] @ temp[:,i:i+1].T )
        Y /= y_set.shape[1]
        # print(Y)
        # Remove negative eigenvalues
        u = min( np.min(np.linalg.eigvalsh(Y)), 0)
        Y -= u * np.eye(d)
                
        # Update the uniform distribution in P
        C,z_k,y_set = next(RandomWalkApproximator(F,Y,y_set,A,b,params,True))
        # print(C,z_k,y_set.shape)
        # Remove negative eigenvalues
        u = min( np.min(np.linalg.eigvalsh(C)), 0)
        C -= u * np.eye(d)
        
        # Project the lattice basis onto the subspace
        # L[d-d_cur+1:,:] = L[d-d_cur+1:,:] - L[d-d_cur+1:,:] @ v.reshape((d,1))\
        #                                 @ v.reshape((d,1)).T / np.sum(v*v)
        # Compute a basis by Hermite normal form
        _, R = column_style_hermite_normal_form(Z[:d-d_cur+1,:])
        V = R[:,d-d_cur+1:]
        L[d-d_cur+1:,:] = (V @ np.linalg.inv(V.T @ V)).T
        # print(L[d-d_cur+1:,:])
        d_cur -= 1
    
    # If no early stopping
    if not early_stop:
        # Solve the one-dimensional problem
        v = L[-1,:] # Direction of the line
        y_bar = np.mean(y_set,axis=1) # Point on the line
        # print(v,y_bar)
        
        # Find an integral point on the line
        # Create a new model
        model = gp.Model("search")
        model.Params.OutputFlag = 0 # Controls output
        # model.Params.MIPGap = 1e-9
        # Variables
        x = model.addVars(range(d), vtype=GRB.INTEGER, ub=N, lb=1, name="x")
        alpha = model.addVars(range(2), vtype = GRB.CONTINUOUS, name="alpha" )
        # Add constraints
        model.addConstrs(
            ( y_bar[k] + (alpha[0]-alpha[1]) * v[k] <= x[k] + 1e-3 for k in range(d) ),
            name="c1")
        model.addConstrs(
            ( y_bar[k] + (alpha[0]-alpha[1]) * v[k] >= x[k] - 1e-3 for k in range(d) ),
            name="c2")
        # # Set initial point
        # for i in range(d):
        #     x[i].start = y_bar[i]
        # model.update()
        # Set the objective function as constant
        model.setObjective(0, GRB.MAXIMIZE)
        # Solve the feasibility problem
        model.optimize()
        
        z = np.zeros((d,))
        try:
            for i in range(d):
                z[i] = x[i].X
            # print(z)
            
            # Find the upper and lower bound of one-dim problem
            bound = [0,0]
            for i in range(N):
                flag = False
                if np.max( z + i * v ) < N + 1 and np.min( z + i * v ) > 0:
                    bound[1] = i
                    flag = True
                if np.max( z - i * v ) < N + 1 and np.min( z - i * v ) > 0:
                    bound[0] = -i
                    flag = True
                if not flag:
                    break
            
            # Shift to a problem with leftmost point 1
            z = z + (bound[0] - 1) * v
            M = bound[1] - bound[0] + 1
            # Define a one-dimensional problem
            G = lambda alpha: float(F( z + alpha[0] * v ))
            params_new = params.copy()
            params_new["N"] = M
            params_new["d"] = 1
            # params_new["eps"] = eps / 4
            params_new["eps"] = eps
            params_new["delta"] = delta / 4
            # print(params_new)
            # Use the uniform solver to solve the one-dim problem
            output_uniform = UniformSolver(G, params_new)
            # Update the total number of points
            total_samples += output_uniform["total"]
            # print(params_new)
            # Optimal point
            x_uni = z + output_uniform["x_opt"] * v
            # print(x_uni)
            
            # Estimate the empirical mean of x_opt
            num_samples = RequiredSamples(delta/2,eps/4,params)
            hat_F = 0
            for i in range(num_samples):
                hat_F = hat_F + float(F(x_uni))
            hat_F /= num_samples
            
            s = np.concatenate(( x_uni.reshape((d,1)) ,[[hat_F]]),axis=0)
            S = np.concatenate((S,s),axis=1)
            # print(S)
        
        except AttributeError:
            print("One-dim problem failed.")
            print(y_bar,L[-1,:])
    
    # Find the point with minimal empirical mean in S
    i_min = np.argmin(S[-1,:])
    x_bar = S[:-1,i_min]
    # Round to an integral solution
    x_opt = Round(F,x_bar,params)["x_opt"]
    
    # Stop timing
    stop_time = time.time()
    
    return {"x_opt":x_opt, "time":stop_time-start_time, "total":total_samples}


def RandomWalkApproximator(F,Y,y_in,A_in,b_in,params,centroid=False):
    """
    Output the new polytope, its approximate centroid and approximate covariance matrix.
    
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
    M = math.ceil(5 * 20 * d * math.log(d+1) * max( 20, math.log(d) ))
    y_set = RandomWalk(y_set,Y,A,b,params,M)
    # Approximate centroid
    y_bar = np.mean(y_set,axis=1,keepdims=True)
    
    # Only need to update the uniform distribution
    if centroid:
        # print(y_bar)
        temp = y_set - y_bar
        Y = np.zeros((d,d))
        for i in range(y_set.shape[1]):
            Y += ( temp[:,i:i+1] @ temp[:,i:i+1].T )
        Y /= y_set.shape[1]
        # print(Y, y_bar[:,0], y_set.shape)
        # Remove negative eigenvalues
        u = min(np.min(np.linalg.eigvalsh(Y)),0)
        Y -= u * np.eye(d)
        yield Y, y_bar[:,0], y_set
        return
    
    # Constantly generate polytopes
    while True:

        # Separation oracle
        so = SO(F,y_bar[:,0],eps/4*min(N,N/4),delta/4,params)
        c = -so["hat_grad"]
        hat_F = so["hat_F"]
        s = np.concatenate((y_bar,[[hat_F]]),axis=0)

        # Update A and b
        c = np.reshape(c,(1,d))
        A = np.concatenate((A,c), axis=0)
        b = np.concatenate((b,c@y_bar),axis=0)
        
        # Warm-start distribution
        # print(A.shape,y_set.shape,b.shape)
        violation = np.min(A @ y_set - b, axis=0)
        y_set = y_set[:,violation >= 0]
        
        # Infeasible
        if y_set.shape[1] == 0:
            yield None, None, None, None, None, None
        
        # Estimate the covarance matrix
        y_bar = np.mean(y_set,axis=1,keepdims=True)
        temp = y_set - y_bar
        Y = np.zeros((d,d))
        for i in range(y_set.shape[1]):
            Y += ( temp[:,i:i+1] @ temp[:,i:i+1].T )
        Y /= y_set.shape[1]
        # print(Y)
        # Remove negative eigenvalues
        u = min(np.min(np.linalg.eigvalsh(Y)),0)
        # print(np.min(np.linalg.eigvalsh(Y)),u)
        Y -= u * np.eye(d)
        
        # Update uniform distribution in P
        # print("start")
        M = math.ceil(5 * 20 * d * math.log(d+1) * max( 20, math.log(d) ))
        y_set = RandomWalk(y_set,Y,A,b,params,M)
        # print("end")

        # Approximate centroid and covariance
        M = int(y_set.shape[1] / 2)
        y_bar = np.mean(y_set[:,:M],axis=1,keepdims=True)
        temp = y_set[:,:M] - y_bar
        Y = np.zeros((d,d))
        for i in range(M):
            Y += ( temp[:,i:i+1] @ temp[:,i:i+1].T )
        Y /= M
        # Remove negative eigenvalues
        u = min( np.min(np.linalg.eigvalsh(Y)), 0)
        # print(np.min(np.linalg.eigvalsh(Y)), u)
        Y -= u * np.eye(d)
        
        # Update point set
        y_set = y_set[:,M:]
        # print(y_bar, Y)
        
        # Output
        yield Y, y_bar[:,0], A, b, y_set, s



