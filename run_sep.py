# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:03:48 2020

@author: haixiang_zhang
"""

import math
import numpy as np
np.random.seed(101)
import matplotlib.pyplot as plt

import models
import utils
import solvers

# print(sys.argv[1])

# Set parameters
params = {}

# Generate the model
# Dimension and scale
params["d"] = 1
# params["N"] = int(sys.argv[1])
params["N"] = 150
# sub-Gaussian parameter
params["sigma"] = 1e0

# Optimality criteria
params["eps"] = 2e-1
params["delta"] = 1e-6

# Record average simulation runs and optimality gaps
total_samples = np.zeros((2,))
gaps = np.zeros((2,))
rate = np.zeros((2,))

# Open the output file
f_out = open("./results/sep_" + str(params["N"]) + ".txt", "w")

for t in range(1):
    print(t)
    model = models.separable_model.separablemodel(params)
    
    f_out.write(str(t))
    f_out.write("\n")
    f_out.flush()
    
    # if "f" in model:
    #     # Plot the function
    #     x = np.linspace(1,params["N"],params["N"])
    #     y = np.zeros((params["N"],))
        
    #     for i,z in enumerate(x):
    #         y[i] = model["f"]([z])
        
    #     plt.plot(x,y)
    # else:
    #     # Plot the function
    #     x = np.linspace(1,params["N"],params["N"])
    #     y = np.zeros((params["N"],))
        
    #     for i,z in enumerate(x):
    #         for _ in range(50):
    #             y[i] += model["F"]([z])
    #     y /= 50
    #     # plt.plot(x,y)
    
    # # Write the obj values
    # f_out.write(" ".join([ str(z) for z in y ]))
    # f_out.write("\n")
    
    # Get the optimal objective value
    f_opt = model["f"](model["x_opt"])
    
    # Use adaptive sampling algorithm
    output_ada = solvers.adaptive_solver.AdaptiveSolver(model["F"],params)
    print(output_ada)
    # Use uniform sampling algorithm
    output_uni = solvers.uniform_solver.UniformSolver(model["F"],params)
    print(output_uni)
    
    # Update records
    total_samples[0] = ( total_samples[0] * t + output_ada["total"] ) / (t+1)
    total_samples[1] = ( total_samples[1] * t + output_uni["total"] ) / (t+1)
    
    gaps[0] = ( gaps[0] * t + model["f"]([output_ada["x_opt"]]) - f_opt ) / (t+1)
    gaps[1] = ( gaps[1] * t + model["f"]([output_uni["x_opt"]]) - f_opt ) / (t+1)
    
    if model["f"]([output_ada["x_opt"]]) - f_opt <= params["eps"]:
        rate[0] = ( rate[0] * t + 1 ) / (t+1)
    else:
        rate[0] = ( rate[0] * t + 0 ) / (t+1)
    
    if model["f"]([output_uni["x_opt"]]) - f_opt <= params["eps"]:
        rate[1] = ( rate[1] * t + 1 ) / (t+1)
    else:
        rate[1] = ( rate[1] * t + 0 ) / (t+1)
    
    # # Use truncated subgradient descent method
    # output_grad = solvers.gradient_solver.GradientSolver(model["F"],params)
    # print(output_grad)
    
    f_out.write(" ".join( [str(output_ada["total"]),str(output_uni["total"]),\
                        str(model["f"]([output_ada["x_opt"]]) - f_opt),\
                        str(model["f"]([output_uni["x_opt"]]) - f_opt)] ))
    f_out.write("\n")
    f_out.flush()

f_out.write("\n")
f_out.write( " ".join([ str(total_samples[0]),str(gaps[0]),str(rate[0]) ]) )
f_out.write("\n")
f_out.write( " ".join([ str(total_samples[1]),str(gaps[1]),str(rate[1]) ]) )

f_out.close()




