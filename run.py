# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 19:26:17 2020

@author: haixiang_zhang
"""

import numpy as np
np.random.seed(11)
import matplotlib.pyplot as plt

import models
import utils

# Set parameters
params = {}

# Dimension and scale
params["d"] = 1
params["N"] = 10

# Optimality criteria
params["eps"] = 1e0
params["delta"] = 1e-6

# Generate the model
params["sigma"] = 1 # sub-Gaussian parameter
model = models.quadratic_model.QuadraticModel(params)




# The optimal solution
# x_opt = sgd_solver(d,N,f,L)

# Optimization ##########################################################


# x_ssgd = ssgd_solver(d,N,f,F,L,eps,delta)
























