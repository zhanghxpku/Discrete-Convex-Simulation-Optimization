# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 19:36:45 2020

@author: haixiang_zhang
"""

import math
import numpy as np

# Compute the Lovasz extension and its subgradient at point x
def lovasz(d,N,F,x):
    
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

# Use the subgradient descent method to find the exact minimizer
def sgd_solver(d,N,f,L):
    
    # Initial point
    x = np.ones((d,)) * N / 2
    x_avg = np.zeros((d,))
    
    # Initial objective function value
    f_old, _ = lovasz(d,N,f,x)
    
    # Iterate numbers and step size
    tol = 1e0
    T = math.ceil(1 * d * (N**2) * (L**2) / (tol**2))
    print(T)
    eta = np.sqrt( d / T ) * N / 1 / L
    
    # Subgradient descent method
    for t in range(T):
        
        # Update cumulative point
        x_avg = (x_avg * t + x) / (t + 1)
        
        # Update the current iteration
        val, sub_grad = lovasz(d,N,f,x)
        x = x - eta * sub_grad
        x = np.clip(x,1,N)
        
        # if np.linalg.norm(sub_grad,np.inf) < tol / N:
        #     break
        
        # if t > 10000 and np.linalg.norm(ret - np.round(ret),np.inf) < 1e-1:
        #     ret = np.round(ret),np.inf
        #     break
        
        if t % 10000 == 1000:
            f_new, _ = lovasz(d,N,f,x_avg)
            print(f_new)
            if f_new > f_old - tol:
                break
            else:
                f_old = f_new

    # Round x_avg to an integral point
    x_int = np.floor(x_avg)
    x_int = np.clip(x_int,-np.inf,N-1)
    # Local coordinate
    x_loc = x_avg - x_int
    
    # Compute a consistent permuation
    temp = sorted(enumerate(-x_loc), key=lambda x:x[1])
    alpha = np.array([ ind[0] for ind in temp ])
    
    x_min = np.copy(x_int)
    f_min = f(x_int)
    
    for i in range(d):
        x_int[alpha[i]] += 1
        f_new = f(x_int)
        if f_new < f_min:
            f_min = f_new
            x_min = x_int

    return x_min


# Use the stochastic subgradient descent method to find PGS solution
def ssgd_solver(d,N,f,F,L,eps,delta):
    
    # Initial point
    x = np.ones((d,)) * N / 2
    x_avg = np.zeros((d,))
    
    # Initial objective function value
    f_old, _ = lovasz(d,N,F,x)
    
    # Iterate numbers and step size
    T = math.ceil( max( 1 * d * (N**2) / (eps**2) * math.log(2/delta),
                       (d**2) * (L**2) / (eps**2),  
                       64 * (d**2) * (N**2) / (eps**2) * math.log(d**2/N**3)
                       ) )
    M = max(2 * math.sqrt( math.log(4*d*N*T / eps) ), L) 
    print(T,M)
    eta = np.sqrt( 1 / T ) * N / M
    
    # Subgradient descent method
    for t in range(T):
        
        # Update average point
        x_avg = (x_avg * t + x) / (t + 1)
        
        # Update the current iteration
        _, sub_grad = lovasz(d,N,F,x)
        # if np.linalg.norm(sub_grad,np.inf) > M:
        #     print("Truncated!")
        sub_grad = np.clip(sub_grad,-M,M)
        x = x - eta * sub_grad
        x = np.clip(x,1,N)
        
        # if np.linalg.norm(sub_grad,np.inf) < tol / N:
        #     break
        
        # if t > 10000 and np.linalg.norm(ret - np.round(ret),np.inf) < 1e-1:
        #     ret = np.round(ret),np.inf
        #     break
        
        if t % 10000 == 1000:
            f_new, _ = lovasz(d,N,f,x_avg)
            print(f_new)
            # if f_new > f_old - 0e-6:
            #     break
            # else:
            #     f_old = f_new        
        
    # Round x_avg to an integral point
    x_int = np.floor(x_avg)
    x_int = np.clip(x_int,-np.inf,N-1)
    # Local coordinate
    x_loc = x_avg - x_int
    
    # Compute a consistent permuation
    temp = sorted(enumerate(-x_loc), key=lambda x:x[1])
    alpha = np.array([ ind[0] for ind in temp ])
    
    # Compute the empirical mean
    f_emp = np.zeros((d+1,))
    n = math.ceil(32 / (eps**2) * math.log(8 / delta))
    
    x_min = np.copy(x_int)
    f_min = np.inf
    
    for i in range(d+1):
        for _ in range(n):
            f_emp[i] += F(x_int) / n
        
        if f_emp[i] < f_min:
            x_min = np.copy(x_int)
            f_min = f_emp[i]
        
        if i < d:
            x_int[alpha[i]] += 1

    return x_min














