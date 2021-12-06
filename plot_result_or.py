# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 10:21:31 2020

@author: haixiang_zhang
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from matplotlib.pyplot import MultipleLocator
# import matplotlib as mpl
# plt.rcParams.update({
#     "text.usetex": True,
#     # "font.family": "sans-serif",
#     # "font.sans-serif": ["Helvetica"],
#     "text.latex.preamble": [r'\usepackage{amsfonts}']})



eps = lambda d: (math.factorial(d))**(1/d) / 5 if d < 60 else d / math.exp(1) / 5
delta = 1e-6
T = lambda d,N: d**2 * N**2 / (eps(d))**2 * math.log(1/delta)


# N = 30
# results = [1291437.8699999996, 1572420.83, 1693635.96, 1785155.97, 1826923.65]
# bounds = [ 0.87 * T(d, N) for d in range(10, 60, 10) ]
# x = [d for d in range(10, 60, 10)]

# x_major_locator = MultipleLocator(10)
# ax = plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# plt.ylim([1.2e6, 2.1e6])
# plt.plot(x,results)
# plt.plot(x,bounds)
# plt.xlabel("d")
# plt.ylabel("Simulation cost")
# plt.legend(["Empirical", "Theory"])
# # plt.show()
# plt.savefig("./results/multi_sep_or_1.png",bbox_inches='tight', dpi=300)




d = 10
results = [1291437.8699999996, 5060472.949999997, 11415249.43, 20237026.589999992, 31199991.4]
bounds = [ 0.87 * T(d, N) for N in range(30, 180, 30) ]
x = [d for d in range(30, 180, 30)]

x_major_locator = MultipleLocator(30)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.ylim([0e6, 3.6e7])
plt.plot(x,results)
plt.plot(x,bounds)
plt.xlabel("N")
plt.ylabel("Simulation cost")
plt.legend(["Empirical", "Theory"])
# plt.show()
plt.savefig("./results/multi_sep_or_2.png",bbox_inches='tight', dpi=300)



