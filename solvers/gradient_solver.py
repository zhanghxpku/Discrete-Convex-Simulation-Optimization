# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:57:17 2020

@author: haixiang_zhang

Truncated subgradient descent method.
"""

import math
import numpy as np
import time
from utils.lovasz import Lovasz, Round, RoundCons, LovaszCons
from utils.subgaussian import RequiredSamples

import gurobipy as gp
from gurobipy import GRB

def GradientSolver(F,params,truncated=True):
    """
    The truncated subgradient method for multi-dim problems.
    """
    
    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    sigma = params["sigma"] if "sigma" in params else 1
    eps = params["eps"] if "eps" in params else 1
    delta = params["delta"] if "delta" in params else 1e-6
    L = params["L"] if "L" in params else 1
    
    # Initial point
    x = np.ones((d,)) * (N+1) / 2
    # x = np.random.uniform(1,N,(d,))
    # The moving average
    x_avg = np.copy(x)
    # Comparion of objective function values
    f_old = 0
    f_new = 0
    # Check stopping criterion every 1000 iterations
    interval = RequiredSamples(delta/4,eps/16/np.sqrt(d),params)
    # print(interval)
    
    # Iterate numbers and step size
    if truncated:
        T = math.ceil( max( 64*d*(N**2)*sigma / (eps**2) * math.log(2/delta),
                        (d**2) * (L**2) / (eps**2),  
                        64*(d**2)*(N**2) / (eps**2) * math.log(sigma*d**2/N**3)
                        ) )
        M = max(sigma*math.sqrt(math.log( max(4*sigma*d*N*T / eps, 1) )), L) 
        eta =  N / M / np.sqrt( T )
    else:
        T = math.ceil( max( 64*d*(N**2)*sigma / (eps**2) * math.log(2/delta),
                        (d**2) * (L**2) / (eps**2),  
                        64*(d**2)*(N**2) / (eps**2) * math.log(sigma*d**2/N**3)
                        ) )
        M = max(sigma*math.sqrt(math.log( max(4*sigma*d*N*T / eps, 1) )), L) 
        eta =  N / M / np.sqrt( T )
        # T = math.ceil( 64*(d + (N**2))*sigma / (eps**2) * math.log(2/delta))
        # M = np.inf
        # eta = N / sigma * np.sqrt(d / T)
    # print(T,eta,M)
    
    # Start timing
    start_time = time.time()
    # Count simulation runs
    total_samples = 0
    
    # Weighted average
    alpha = 0.5
    weight_cum = 0
    # Early stopping
    cnt = 0
    f_old = np.inf
    f_new = 1
    # interval = int(interval/10)
    
    # Truncated subgradient descent
    for t in range(T):
        
        # Compute subgradient
        hat_F, sub_grad = Lovasz(F,x,params)
        
        total_samples += (2*d)
        
        # Truncate subgradient
        sub_grad = np.clip(sub_grad, -M, M)
        # if np.linalg.norm(sub_grad,np.inf) > M:
        #     print("Truncated!")
        
        # Update and project the current point
        x = x - 150 * d / int(t/interval+1) * eta * sub_grad
        x = np.clip(x,1,N)
        
        # Update the moving average
        new_weight = weight_cum * (1-alpha) + alpha
        x_avg = ( x_avg * t + x ) / ( t + 1 )
        # x_avg = (x_avg * weight_cum * (1-alpha) + x*alpha) / new_weight
        # Update the function value
        f_new = ( f_new * t + hat_F ) / ( t + 1 )
        # f_new = (f_new * weight_cum * (1-alpha) + hat_F*alpha) / new_weight
        # Update the cumulative weight
        weight_cum = new_weight
        
        if t % (interval * 1) == 0:
            f, _ = Lovasz(F,x_avg,params)
            print(f_new, hat_F, f)
            # print(sub_grad)
            print(x_avg)
        
        # Early stopping
        if t % interval == interval - 1 and t >= 0 * interval:
            print(cnt,f_new,f_old,total_samples)
            # Decay is not sufficient
            if f_new - f_old >= -eps / np.sqrt(N):
                cnt += 1
            else:
                cnt = 0
            if f_new < f_old:
                f_old = f_new
            if cnt > 2:
                break

        # if t % interval == interval - 1 and f_new < f_old:
        #     # Update f_old and f_new
        #     f_old = f_new
    
    # Round to an integral point
    output_round = Round(F,x_avg,params)
    x_opt = output_round["x_opt"]
    total_samples += output_round["total"]
    # Stop timing
    stop_time = time.time()
    
    return {"x_opt":x_opt, "time":stop_time-start_time, "total":total_samples}
    

def GradientProjSolver(F,params,truncated=True):
    """
    The truncated subgradient method for multi-dim problems with capacity constraints.
    """
    
    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    sigma = params["sigma"] if "sigma" in params else 1
    eps = params["eps"] if "eps" in params else 1
    delta = params["delta"] if "delta" in params else 1e-6
    L = params["L"] if "L" in params else 1
    
    # The capacity constraint
    K = params["K"] if "K" in params else N * d
    
    # Initial point
    x = np.ones((d,)) * (N+1) / 4
    x = np.cumsum(x)
    # x = np.random.uniform(1,N,(d,))
    # The moving average
    x_avg = np.copy(x)
    # Comparion of objective function values
    f_old = 0
    f_new = 0
    # Check stopping criterion every 1000 iterations
    interval = int(RequiredSamples(delta/4,eps/16/np.sqrt(d),params) * 1e-1)
    # print(interval)
    
    if truncated:
        T = math.ceil( max( 64*d*(N**2)*sigma / (eps**2) * math.log(2/delta),
                        (d**2) * (L**2) / (eps**2),  
                        64*(d**2)*(N**2) / (eps**2) * math.log(sigma*d**2/N**3)
                       ) )
        M = max(sigma*math.sqrt(math.log( max(4*sigma*d*N*T / eps, 1) )), L) 
        eta =  N / M / np.sqrt( T ) * params["eta"]
    else:
        T = math.ceil( 64*(d + (N**2))*sigma / (eps**2) * math.log(2/delta))
        M = np.inf
        eta = N / sigma * np.sqrt(d / T) * params["eta"]
    # # Iterate numbers and step size
    # if truncated:
    #     T = math.ceil( max( 64*d*(K**2)*sigma / (eps**2) * math.log(2/delta),
    #                     (d**2) * (L**2) / (eps**2),  
    #                     64*(d**2)*(K**2) / (eps**2) * math.log(sigma*d**2/K**3)
    #                    ) )
    #     M = max(sigma*math.sqrt(math.log( max(4*sigma*d*N*T / eps, 1) )), L) 
    #     eta =  N / M / np.sqrt( T )
    # else:
    #     # T = math.ceil( max( 64*d*(K**2)*sigma / (eps**2) * math.log(2/delta),
    #     #                 (d**2) * (L**2) / (eps**2),  
    #     #                 64*(d**2)*(K**2) / (eps**2) * math.log(sigma*d**2/K**3)
    #     #                 ) )
    #     # M = max(sigma*math.sqrt(math.log( max(4*sigma*d*K*T / eps, 1) )), L) 
    #     # eta =  K / M / np.sqrt( T )
    #     T = math.ceil( 64*(d + (K**2))*sigma / (eps**2) * math.log(2/delta))
    #     M = np.inf
    #     eta = K / sigma * np.sqrt(d / T)
    # # print(T,eta,M)
    
    # Start timing
    start_time = time.time()
    # Count simulation runs
    total_samples = 0
    
    # Weighted average
    alpha = 0.1
    weight_cum = 0
    # Early stopping
    cnt = 0
    f_old = np.inf
    f_new = 1
    # interval = int(interval/10)
    
    # Truncated subgradient descent
    for t in range(T):
                
        # Compute subgradient
        hat_F, sub_grad = LovaszCons(F,x,params)
        
        total_samples += (2*d)
        
        # Truncate subgradient
        sub_grad = np.clip(sub_grad, -M, M)
        # if np.linalg.norm(sub_grad,np.inf) > M:
        #     print("Truncated!")
        
        # Update and project the current point
        x = x - 5 * np.sqrt(N) * d / int(t/interval+1) * eta * sub_grad
        
        # Projection under the capacity constraint
        model = gp.Model("projection")
        model.Params.OutputFlag = 0 # Controls output
        # Variables
        y = model.addVars(range(d), vtype=GRB.CONTINUOUS, name="y")
        # Add constraints
        model.addConstrs(
            (y[j+1] >= y[j] + 1 + 1e-7 for j in range(d-1)),
            name="c1")
        model.addConstrs(
            (y[j+1] <= y[j] + N - 1e-7 for j in range(d-1)),
            name="c2")
        model.addConstr(
            y[0] >= 1 + 1e-7,
            name="c3")
        model.addConstr(
            y[d-1] <= K,
            name="c4")
        model.addConstr(
            y[0] <= N - 1e-7,
            name="c5")
        # model.addConstr(
        #     gp.quicksum( y[k] for k in range(d) ) <= K,
        #     name="c5")
        # Set the objective function
        model.setObjective(gp.quicksum( (x[k]-y[k])*(x[k]-y[k]) for k in range(d) ), GRB.MINIMIZE)
        # model.setObjective(gp.quicksum((x[k]-x[d+k]-v[k])*(x[k]-x[d+k]-v[k]) for k in range(d)),
        #                    GRB.MINIMIZE)
        # Solve the projection problem
        model.optimize()
        
        x = np.zeros((d,))
        for i in range(d):
            x[i] = y[i].X
        
        # # if np.min(np.diff(x)) < 1:
        # print("pts:", x)

        # Update the moving average
        new_weight = weight_cum * (1-alpha) + alpha
        x_avg = ( x_avg * t + x ) / ( t + 1 )
        # x_avg = (x_avg * weight_cum * (1-alpha) + x*alpha) / new_weight
        # Update the function value
        f_new = ( f_new * t + hat_F ) / ( t + 1 )
        # f_new = (f_new * weight_cum * (1-alpha) + hat_F*alpha) / new_weight
        # Update the cumulative weight
        weight_cum = new_weight
        
        if t % (int(interval / 3)) == 0:
            f, _ = LovaszCons(F,x_avg,params)
            # print(f_new, hat_F, f)
            # # print(np.linalg.norm(sub_grad))
            # print(x_avg)
        
        # Early stopping
        if t % int(interval) == int(interval) - 1 and t >= 0 * interval:
            # print(cnt,f_new,f_old,total_samples)
            # Decay is not sufficient
            if truncated:
                if f_new - f_old >= -eps / np.sqrt(N):
                    cnt += 1
                else:
                    cnt = 0
            else:
                if f_new - f_old >= -eps / np.sqrt(N) / 8:
                    cnt += 1
                else:
                    cnt = 0
            if f_new < f_old:
                f_old = f_new
            if cnt > 1:
                break

        # if t % interval == interval - 1 and f_new < f_old:
        #     # Update f_old and f_new
        #     f_old = f_new
    
    # Round to an integral point
    output_round = RoundCons(F,x_avg,params)
    x_opt = output_round["x_opt"]
    total_samples += output_round["total"]
    # Stop timing
    stop_time = time.time()
    
    return {"x_opt":x_opt, "time":stop_time-start_time, "total":total_samples}
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

