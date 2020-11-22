# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 19:26:17 2020

@author: haixiang_zhang
"""

import numpy as np
np.random.seed(101)
import matplotlib.pyplot as plt
import sys

import models
import utils
import solvers

# print(sys.argv[1])

# Set parameters
params = {}

# Optimality criteria
params["eps"] = 3e-1
params["delta"] = 1e-6

# Generate the model
params["sigma"] = 1e0 # sub-Gaussian parameter

# Dimension and scale
params["d"] = 1
# params["N"] = int(sys.argv[1])
for N in range(10,110,10):

    params["N"] = N
    
    # Record average simulation runs and optimality gaps
    total_samples = np.zeros((2,))
    gaps = np.zeros((2,))
    rate = np.zeros((2,))
    
    # Open the output file
    f_out = open("quad_" + str(params["N"]) + ".txt", "w")
    
    for t in range(100):
        print(t)
        model = models.quadratic_model.QuadraticModel(params)
        
        f_out.write(str(t))
        f_out.write("\n")
        f_out.flush()
        
        if "f" in model:
            # Plot the function
            x = np.linspace(1,params["N"],params["N"])
            y = np.zeros((params["N"],))
            
            for i,z in enumerate(x):
                y[i] = model["f"]([z])
            
            # plt.plot(x,y)
        else:
            # Plot the function
            x = np.linspace(1,params["N"],params["N"])
            y = np.zeros((params["N"],))
            
            for i,z in enumerate(x):
                for _ in range(50):
                    y[i] += model["F"]([z])
            y /= 50
            # plt.plot(x,y)
        
        # # Write the obj values
        # f_out.write(" ".join([ str(z) for z in y ]))
        # f_out.write("\n")
        
        # Get the optimal objective value
        f_opt = np.min(y)
        
        # # Set precision
        # params["eps"] = max(abs(f_opt) * 1e-1, 1e-1)
        # print(params)
        
        # Use adaptive sampling algorithm
        output_ada = solvers.adaptive_solver.AdaptiveSolver(model["F"],params)
        # print(output_ada)
        # Use uniform sampling algorithm
        output_uni = solvers.uniform_solver.UniformSolver(model["F"],params)
        # print(output_uni)
        
        # Update records
        total_samples[0] = ( total_samples[0] * t + output_ada["total"] ) / (t+1)
        total_samples[1] = ( total_samples[1] * t + output_uni["total"] ) / (t+1)
        
        gaps[0] = ( gaps[0] * t + y[output_ada["x_opt"]-1] - f_opt ) / (t+1)
        gaps[1] = ( gaps[1] * t + y[output_uni["x_opt"]-1] - f_opt ) / (t+1)
        
        if y[output_ada["x_opt"]-1] - f_opt <= params["eps"]:
            rate[0] = ( rate[0] * t + 1 ) / (t+1)
        else:
            rate[0] = ( rate[0] * t + 0 ) / (t+1)
        
        if y[output_uni["x_opt"]-1] - f_opt <= params["eps"]:
            rate[1] = ( rate[1] * t + 1 ) / (t+1)
        else:
            rate[1] = ( rate[1] * t + 0 ) / (t+1)
        
        # # Use truncated subgradient descent method
        # output_grad = solvers.gradient_solver.GradientSolver(model["F"],params)
        # print(output_grad)
        
        f_out.write(" ".join( [str(output_ada["total"]),str(output_uni["total"]),\
                           str(y[output_ada["x_opt"]-1] - f_opt),\
                           str(y[output_uni["x_opt"]-1] - f_opt)] ))
        f_out.write("\n")
        f_out.flush()
    
    f_out.write("\n")
    f_out.write( " ".join([ str(total_samples[0]),str(gaps[0]),str(rate[0]) ]) )
    f_out.write("\n")
    
    f_out.write( " ".join([ str(total_samples[1]),str(gaps[1]),str(rate[1]) ]) )
    f_out.write("\n")  
    
    f_out.close()














