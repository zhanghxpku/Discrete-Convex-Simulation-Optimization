# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 22:02:33 2020

@author: haixiang_zhang

The LLL algorithm.
"""

import numpy as np

def LLL(basis):
    """
    Compute an LLL-reduced basis.
    """
    
    # Copy the parameter
    b = np.copy(basis).astype('float64') 
    # Get the dimension
    d = basis.shape[0]
    
    # Algorithm parameter
    delta = 0.75
    # Gram-Schmidt orthogonalization
    u = GramSchmidt(b)
    
    k = 1
    while k < d:
        for j in range(k-1,-1,-1):
            mu_kj = np.sum(b[k,:]*u[j,:]) / np.sum(u[j,:]*u[j,:])
            
            if abs(mu_kj) > 0.5:
                b[k,:] -= ( np.round(mu_kj) * b[j,:] )
                # Gram-Schmidt orthogonalization
                u = GramSchmidt(b)
        
        mu_kk_1 = np.sum(b[k,:]*u[k-1,:]) / np.sum(u[k-1,:]*u[k-1,:])
        if np.sum(u[k,:]*u[k,:]) >= \
            ( delta - mu_kk_1**2 ) * np.sum(u[k-1,:]*u[k-1,:]):
            k += 1
        else:
            # Exchange basis
            b[[k-1,k],:] = b[[k,k-1],:]
            # Gram-Schmidt orthogonalization
            u = GramSchmidt(b)
            k = max(k-1, 1)
    
    return b
    
def GramSchmidt(basis):
    """
    Gram-Schmidt orthogonalization.
    """
    
    # Get the dimension
    d = basis.shape[0]
    
    # The orthogonal basis
    u = np.zeros(basis.shape)
    
    for i in range(d):
        u[i,:] = basis[i,:]
        for j in range(i):
            u[i,:] -= ( (np.sum(u[i,:]*u[j,:])/np.sum(u[j,:]*u[j,:])) * u[j,:] )
    
    return u
