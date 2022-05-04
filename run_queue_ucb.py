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

# Dimension and scale
params["d"] = 1
params["N"] = int(sys.argv[1])
# params["N"] = 100

# Optimality criteria
params["eps"] = 1e0
params["delta"] = 1e-6

# Generate the model
params["sigma"] = 1e1 # sub-Gaussian parameter

# Record average simulation runs and optimality gaps
total_samples = np.zeros((2,))
gaps = np.zeros((2,))
rate = np.zeros((2,))

# Open the output file
f_out = open("./results/queue_ucb_" + str(params["N"]) + ".txt", "w")

for t in range(100):
    print(t)
    model = models.queueing_model.queuemodel(params)
    
    f_out.write(str(t))
    f_out.write("\n")
    f_out.flush()
    
    # if "f" in model:
    #     # Plot the function
    #     x = np.linspace(1,params["N"],params["N"])
    #     y = np.zeros((params["N"],))
        
    #     for i,z in enumerate(x):
    #         y[i], _ = model["f"]([z])
        
    #     plt.plot(x,y)
    # else:
    #     # Plot the function
    #     x = np.linspace(1,params["N"],params["N"])
    #     y = np.zeros((params["N"],))
        
    #     for i,z in enumerate(x):
    #         for _ in range(20):
    #             y[i] += model["F"]([z])
    #     y /= 20
    #     plt.plot(x,y)
    
    # # # Write the obj values
    # # f_out.write(" ".join([ str(z) for z in y ]))
    # # f_out.write("\n")
    
    # # Get the optimal objective value
    # f_opt = np.min(y)
    
    # # Use adaptive sampling algorithm
    # output_ada = solvers.adaptive_solver.AdaptiveSolver(model["F"],params)
    # # print(output_ada)
    # # Use uniform sampling algorithm
    # output_uni = solvers.uniform_solver.UniformSolver(model["F"],params)
    # # print(output_uni)
    # Use lil'UCB algorithm
    output_ucb = solvers.ucb_solver.LILUCBSolver(model["F"],params)
    # print(output_ucb)
    
    # Update records
    total_samples[0] = ( total_samples[0] * t + output_ucb["total"] ) / (t+1)
    
    # Get the optimal value
    num_samples = utils.subgaussian.RequiredSamples(params["delta"],
                                                    params["eps"],
                                                    params)
    y = np.zeros((2,))
    for i in range(int(num_samples/100)):
        y[0] = ( y[0] * i + model["F"]([output_ucb["x_opt"]]) ) / (i+1)
    
    gaps[0] = ( gaps[0] * t + y[0] ) / (t+1)
    
    # if y[output_ada["x_opt"]-1] - f_opt <= params["eps"]:
    #     rate[0] = ( rate[0] * t + 1 ) / (t+1)
    # else:
    #     rate[0] = ( rate[0] * t + 0 ) / (t+1)
    
    # if y[output_uni["x_opt"]-1] - f_opt <= params["eps"]:
    #     rate[1] = ( rate[1] * t + 1 ) / (t+1)
    # else:
    #     rate[1] = ( rate[1] * t + 0 ) / (t+1)
    
    # # Use truncated subgradient descent method
    # output_grad = solvers.gradient_solver.GradientSolver(model["F"],params)
    # print(output_grad)
    
    f_out.write(" ".join( [str(output_ucb["total"]),str(y[0])] ))
    f_out.write("\n")
    f_out.flush()

f_out.write("\n")
f_out.write( " ".join([ str(total_samples[0]),str(gaps[0]) ]) )

f_out.close()


stat = np.zeros((15,8))
min_val = np.zeros((15,))

for N in range(10,160,10):
    i = 0
    results = np.zeros((102,4))
        
    with open("./results/queue_"+str(N)+".txt") as f_in:
        for line in f_in:
            line = line.split(" ")
            if len(line) == 4:
                for j in range(4):
                    results[i,j] = float(line[j])
                i += 1
            elif len(line) == 3:
                for j in range(3):
                    results[i,j] = float(line[j])
                i += 1
    
    # Average simulation runs
    stat[int(N/10)-1,0] = np.mean(results[:-2,0])
    stat[int(N/10)-1,1] = np.std(results[:-2,0])
    stat[int(N/10)-1,2] = np.mean(results[:-2,1])
    stat[int(N/10)-1,3] = np.std(results[:-2,1])
    
    # Average optimality gaps
    stat[int(N/10)-1,4] = np.mean(results[:-2,2])
    stat[int(N/10)-1,5] = np.std(results[:-2,2])
    stat[int(N/10)-1,6] = np.mean(results[:-2,3])
    stat[int(N/10)-1,7] = np.std(results[:-2,3])
    
    # Minimal value
    min_val[int(N/10)-1] = np.min( results[:-2,[2,3]] )

plt.figure()
plt.errorbar([N for N in range(10,160,10)], stat[:,0],
              yerr=stat[:,1])
plt.errorbar([N for N in range(10,160,10)], stat[:,2],
              yerr=stat[:,3])
# N_set = np.array([N for N in range(10,110,10)])
# plt.plot( N_set, (np.log(N_set) **1) * 1e4 + stat[0,0] - 1e3*math.log(10)**2 )
# plt.plot( N_set, np.log(N_set) * 6e2 + stat[1,2] - 6e2*math.log(20) )

plt.figure()
plt.errorbar([N for N in range(10,160,10)], stat[:,4],
              yerr=stat[:,5])
plt.errorbar([N for N in range(10,160,10)], stat[:,6],
              yerr=stat[:,7])






