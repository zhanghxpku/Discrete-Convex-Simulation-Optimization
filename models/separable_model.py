# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:54:31 2020

@author: haixiang_zhang

Separable convex functions with flat landscape
"""

import math
import numpy as np

def OneDimFunction(x_0,x_opt,x_1,x_2,params):
    
    x = np.array(x_0)
    if len(x.shape) == 1:
        # return np.sum((x-x_opt)**4 * ( (x < x_opt)*x_1 + (x >= x_opt)*x_2 ))
        return np.sum(( (x < x_opt)*x_1/np.sqrt(x)\
                       + (x >= x_opt)*x_2/np.sqrt(params["N"]+1-x) ))
    else:
        opt = x_opt.reshape((x_opt.shape[0],1))
        return np.sum( ((x < opt).T * x_1).T/np.sqrt(x)\
                        + (((x >= opt).T * x_2).T )/np.sqrt(params["N"]+1-x)
                      ,axis=0)

def SeparableModel(params):
    
    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    sigma = params["sigma"] if "sigma" in params else 1
    
    # Optimal point
    x_opt = np.random.randint(1+int(0.0*(N-1)), 1+int(0.3*(N-1)), (d,))
    coef = np.random.uniform(0.75,1.25,(d,))

    # f = lambda x: np.sum( coef * ( (np.array(x) < x_opt) * np.sqrt((x_opt) / (np.array(x))) \
    #                       + (np.array(x) >= x_opt) * np.sqrt( (N + 1 - x_opt) / (N + 1 - np.array(x))) ))\
    #                       - np.sum(coef)
    # f = lambda x: np.sum(coef*( (np.array(x) < x_opt) * ((x_opt-1)**0.25-(np.array(x)-1)**0.25)\
    #             + (np.array(x) >= x_opt) *( (N-x_opt)**0.25-(N-np.array(x))**0.25 ) ))
    x_1 = coef * (x_opt**0.5)
    x_2 = coef * ((N+1-x_opt)**0.5)
    f = lambda x: OneDimFunction(x,x_opt,x_1,x_2,params) - np.sum(coef)
    # f = lambda x: np.sum((np.array(x)-x_opt)**4 * ( (np.array(x) < x_opt)*x_1\
    #             + (np.array(x) >= x_opt)*x_2 ))
    F = lambda x: f(x) + sigma * np.random.randn( (np.array(x).shape[-1])\
                                             ** (len(np.array(x).shape)-1) )
    F_hat = lambda x, n=1: f(x) + sigma / math.sqrt(n)\
    * np.random.randn( (np.array(x).shape[-1]) ** (len(np.array(x).shape)-1))
    L = np.sqrt(np.sum( np.maximum( x_opt, N+1-x_opt ) ))
    
    opt = x_opt
    
    # Return
    ret = {"F":F_hat, "f":f, "F_hat":F_hat, "L":L, "x_opt":opt}
    
    return ret

def SeparableORModel(params):
    
    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    sigma = params["sigma"] if "sigma" in params else 1
    
    # Optimal point
    x_opt = np.random.randint(1+int(0.0*(N-1)), 1+int(0.3*(N-1)), (d,))
    coef = np.random.uniform(0.75,1.25,(d,))

    # f = lambda x: np.sum( coef * ( (np.array(x) < x_opt) * np.sqrt((x_opt) / (np.array(x))) \
    #                       + (np.array(x) >= x_opt) * np.sqrt( (N + 1 - x_opt) / (N + 1 - np.array(x))) ))\
    #                       - np.sum(coef)
    # f = lambda x: np.sum(coef*( (np.array(x) < x_opt) * ((x_opt-1)**0.25-(np.array(x)-1)**0.25)\
    #             + (np.array(x) >= x_opt) *( (N-x_opt)**0.25-(N-np.array(x))**0.25 ) ))
    x_1 = coef * (x_opt**0.5)
    x_2 = coef * ((N+1-x_opt)**0.5)
    f = lambda x: OneDimFunction(x,x_opt,x_1,x_2,params) / d - np.mean(coef)
    # f = lambda x: np.sum((np.array(x)-x_opt)**4 * ( (np.array(x) < x_opt)*x_1\
    #             + (np.array(x) >= x_opt)*x_2 ))
    F = lambda x: f(x) + sigma * np.random.randn( (np.array(x).shape[-1])\
                                             ** (len(np.array(x).shape)-1) )
    F_hat = lambda x, n=1: f(x) + sigma / math.sqrt(n)\
    * np.random.randn( (np.array(x).shape[-1]) ** (len(np.array(x).shape)-1))
    L = np.sqrt(np.sum( np.maximum( x_opt, N+1-x_opt ) )) / d
    
    opt = x_opt
    
    # Return
    ret = {"F":F_hat, "f":f, "F_hat":F_hat, "L":L, "x_opt":opt}
    
    return ret