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
    a = 1e-4 # eps in paper
    b = 1e-3 * a # delta in paper
    
    # Total number of iterations
    T = d * math.log(d*L*N/eps) / a * 2
    # Set of points where SO is called and their empirical means
    S = np.zeros((T,d+1))
    print(T)
    
    # Initial polytope Ax >= b
    A = np.diag(np.concatenate([np.ones(d,),-np.ones((d,))]))
    b = np.concatenate([np.ones(d,),-N * np.ones((d,))])
    # Initial volumetric center
    z = N/2 * np.ones((d,))
    
        
    
    