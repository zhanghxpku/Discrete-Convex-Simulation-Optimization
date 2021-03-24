# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 10:21:31 2020

@author: haixiang_zhang
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
# import matplotlib as mpl
# plt.rcParams.update({
#     "text.usetex": True,
#     # "font.family": "sans-serif",
#     # "font.sans-serif": ["Helvetica"],
#     "text.latex.preamble": [r'\usepackage{amsfonts}']})

stat = np.zeros((15,8))
min_val = np.zeros((15,))

for N in range(10,160,10):
    i = 0
    results = np.zeros((102,4))
        
    # with open("./results/sqrt_sep/sep_"+str(N)+".txt") as f_in:
    with open("./results/queue/queue_"+str(N)+".txt") as f_in:
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

# Plot the results
N_set = np.array([N for N in range(10,160,10)])
plt.figure()
curve1, = plt.plot(N_set, stat[:,0], color="navy")
              # yerr=stat[:,1])
curve2, = plt.plot(N_set, stat[:,2], color="purple")
              # yerr=stat[:,3])
              
# Plot the variance
cs = CubicSpline(N_set, stat[:,0])
xfit = np.linspace(10, 150, 1400)
yfit = cs(xfit)
cs1 = CubicSpline(N_set, stat[:,1])
yfit1 = cs1(xfit)
plt.fill_between(xfit, yfit-yfit1, yfit+yfit1,color='b', alpha=0.1)

# Plot the variance
cs = CubicSpline(N_set, stat[:,2])
xfit = np.linspace(10, 150, 1400)
yfit = cs(xfit)
cs1 = CubicSpline(N_set, stat[:,3])
yfit1 = cs1(xfit)
plt.fill_between(xfit, yfit-yfit1, yfit+yfit1,color='r', alpha=0.1)

# Plot the trend
curve3, = plt.plot( N_set, 
                    np.ceil(np.log(0.5*N_set+11))*4.9e4+stat[0,0]-4.6e4*math.log(10),
                    alpha=0.2, color="b" )
curve4, = plt.plot( N_set, 4.6e4 * np.ones(N_set.shape), alpha=0.2, color="r" )
# curve3, = plt.plot( N_set, 
#                     np.log(N_set)*1.4e5+stat[0,0]-1.4e5*math.log(10),
#                     alpha=0.2, color="b" )
# curve4, = plt.plot( N_set, 1.1e5 * np.ones(N_set.shape), alpha=0.2, color="r" )

plt.legend([curve1,curve2,curve3,curve4],
           ["AS","US","y = O([log N])","y = Const"],
           loc='center right')
plt.xlabel("N")
plt.ylabel("Sample complexity")
# plt.title("Queueing Model")
plt.savefig("./results/queue/queue.png",bbox_inches='tight', dpi=300)
# plt.title("Artificial Convex Model")
# plt.savefig("./results/sqrt_sep/sep.png",bbox_inches='tight', dpi=300)


# plt.figure()
# plt.errorbar(N_set, stat[:,4],
#               yerr=stat[:,5])
# plt.errorbar(N_set, stat[:,6],
#               yerr=stat[:,7])