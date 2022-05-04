# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:54:31 2020

@author: haixiang_zhang

Separable convex functions with flat landscape
"""

import math
import numpy as np


def one_dim_function(x_0, x_opt, x_1, x_2, params, deg=0.5):
    x = np.array(x_0)
    if len(x.shape) == 1:
        return np.sum(((x < x_opt) * x_1 / x ** deg + (x >= x_opt) * x_2 / (params["N"] + 1 - x) ** deg))
    else:
        opt = x_opt.reshape((x_opt.shape[0], 1))
        return np.sum(((x < opt).T * x_1).T / x ** deg
                      + ((x >= opt).T * x_2).T / (params["N"] + 1 - x) ** deg
                      , axis=0)


def separable_model(params):
    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    sigma = params["sigma"] if "sigma" in params else 1

    deg = 0.5

    # Optimal point
    x_opt = np.random.randint(1 + int(0.0 * (N - 1)), 1 + int(0.3 * (N - 1)), (d,))
    coef = np.random.uniform(0.75, 1.25, (d,))

    x_1 = coef * (x_opt ** deg)
    x_2 = coef * ((N + 1 - x_opt) ** deg)
    f = lambda x: one_dim_function(x, x_opt, x_1, x_2, params, deg) - np.sum(coef)

    # F = lambda x: f(x) + sigma * np.random.randn((np.array(x).shape[-1]) \
    #                                              ** (len(np.array(x).shape) - 1))
    F_hat = lambda x, n=1: f(x) + sigma / math.sqrt(n) \
        * np.random.randn((np.array(x).shape[-1]) ** (len(np.array(x).shape) - 1))
    L = np.sum((np.maximum(x_opt, N + 1 - x_opt)) ** deg * coef)
    opt = x_opt

    # Return
    ret = {"F": F_hat, "f": f, "F_hat": F_hat, "L": L, "x_opt": opt}

    return ret


def one_dim_function_new(x_0, x_opt, x_1, x_2, params, deg=1.0):
    x = np.array(x_0)
    if len(x.shape) == 1:
        return np.sum(((x < x_opt) * x_1 * np.abs(x_opt - x) ** deg
                       + (x >= x_opt) * x_2 * np.abs(x - x_opt) ** deg))
    else:
        opt = x_opt.reshape((x_opt.shape[0], 1))
        return np.sum(((x < opt).T * x_1).T * np.abs(opt - x) ** deg \
                      + ((x >= opt).T * x_2).T * np.abs(x - opt) ** deg
                      , axis=0)


def separable_model_new(params):
    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    sigma = params["sigma"] if "sigma" in params else 1

    deg = 1.2

    # Optimal point
    x_opt = np.random.randint(1 + int(0.0 * (N - 1)), 1 + int(0.3 * (N - 1)), (d,))
    coef = np.random.uniform(0.75, 1.25, (d,)) * 1

    x_1 = coef
    x_2 = coef
    f = lambda x: one_dim_function_new(x, x_opt, x_1, x_2, params, deg)

    F = lambda x: f(x) + sigma * np.random.randn((np.array(x).shape[-1]) \
                                                 ** (len(np.array(x).shape) - 1))
    F_hat = lambda x, n=1: f(x) + sigma / math.sqrt(n) \
                           * np.random.randn((np.array(x).shape[-1]) ** (len(np.array(x).shape) - 1))
    L = deg * np.sum((np.maximum(x_opt, N + 1 - x_opt)) ** (deg - 1) * coef)
    opt = x_opt

    # Return
    ret = {"F": F_hat, "f": f, "F_hat": F_hat, "L": L, "x_opt": opt}

    return ret


def separable_model_or(params):
    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    sigma = params["sigma"] if "sigma" in params else 1

    # Optimal point
    x_opt = np.random.randint(1 + int(0.0 * (N - 1)), 1 + int(0.3 * (N - 1)), (d,))
    coef = np.random.uniform(0.75, 1.25, (d,))

    x_1 = coef * (x_opt ** 0.5)
    x_2 = coef * ((N + 1 - x_opt) ** 0.5)
    f = lambda x: one_dim_function(x, x_opt, x_1, x_2, params) / d - np.mean(coef)
    F = lambda x: f(x) + sigma * np.random.randn((np.array(x).shape[-1]) \
                                                 ** (len(np.array(x).shape) - 1))
    F_hat = lambda x, n=1: f(x) + sigma / math.sqrt(n) \
                           * np.random.randn((np.array(x).shape[-1]) ** (len(np.array(x).shape) - 1))
    L = np.sqrt(np.sum(np.maximum(x_opt, N + 1 - x_opt))) / d
    opt = x_opt

    # Return
    ret = {"F": F_hat, "f": f, "F_hat": F_hat, "L": L, "x_opt": opt}

    return ret
