# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:58:20 2020

@author: haixiang_zhang

Dimension reduction method
"""

import math
import numpy as np
import time
import utils
from utils.lovasz import Round, SO, RoundCons, SOCons
from utils.lll import LLL
from utils.subgaussian import required_samples
from .random_walk_solver import random_walk, random_proj_walk
from .uniform_solver import uniform_solver
from .adaptive_solver import adaptive_solver
from hsnf import column_style_hermite_normal_form

import importlib
importlib.reload(utils)

import gurobipy as gp
from gurobipy import GRB

cst = 16


def dimension_reduction_solver(F, params):
    """
    Dimension reduction method for multi-dim problems.
    """

    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    # sigma = params["sigma"] if "sigma" in params else 1
    eps = params["eps"] if "eps" in params else 1
    delta = params["delta"] if "delta" in params else 1e-6

    # # Parameters of the algorithm
    # eta = 1e-2 # eps in paper

    # Start timing
    start_time = time.time()
    # Count simulation runs
    total_samples = 0
    # Record points where SO is called and their empirical means
    S = np.zeros((d + 1, 0))
    # Simulation cost of each separation oracle
    so_samples = required_samples(delta / 4, eps * cst / 2 / np.sqrt(d), params)
    print(so_samples)

    # The current dimension
    d_cur = d
    # The basis of integral lattice
    L = np.eye(d)
    # The pre-images
    Z = np.zeros((d, d))

    # Record polytope Ax >= b
    A = np.zeros((0, d))
    b = np.zeros((0, 1))
    # The initial uniform distribution in P
    y_set = np.random.uniform(1, N, (d, math.ceil(200 * 20 * d * math.log(d + 1) * max(20.0, math.log(d)))))
    # The initial ellipsoid
    # C_inv = (N-1)**(-2) * 12 * np.eye(d) # A in paper
    C = (N - 1) ** 2 / 12 * np.eye(d)  # A_inv in paper
    # Initial centroid
    z_k = (N + 1) / 2 * np.ones((d,))

    # Early stopping
    early_stop = False
    # Iteratively solve d_cur-dimensional problems
    while d_cur > 1:

        print(d_cur)
        # The current basis
        L_cur = np.copy(L[d - d_cur:, :])

        # Iterate until we find a short basis vector
        # Use random walk method
        for K, z_new, A_new, b_new, y_new, s in randomwalk_approximator(F, C, y_set, A,
                                                                        b, params):
            # Check early stopping
            if K is None:
                early_stop = True
                break

            u = np.linalg.eigvalsh(K)
            if np.sum(u > 1e-10) < d_cur:
                break

            # Number of samples
            total_samples += so_samples * (2 * d)
            # Update set S
            S = np.concatenate((S, s), axis=1)

            # The LLL algorithm
            basis = LLL(L_cur, K)
            # Choose the shortest vector
            norm = np.diag((basis @ K) @ basis.T)

            # Stopping criterion
            # if np.min(norm) < 10e1 / d**2:
            if np.min(norm) < 1000:
                i_short = np.argmin(norm)
                L_cur[0, :] = basis[i_short, :]
                # Update the basis and point set
                y_set = y_new
                # Update A and b
                A, b = A_new, b_new
                # Update centroid
                z_k = z_new
                break

        # Check if intersection is empty
        if early_stop:
            break

        # Dimension reduction
        v = L_cur[0, :]

        # Find the pre-image
        if d_cur == d:
            z = v
            Z[0, :] = v
        else:
            # Solve for z
            # Create a new model
            model = gp.Model("pre-image")
            model.Params.OutputFlag = 0  # Controls output
            # model.Params.MIPGap = 1e-9
            # Variables
            x = model.addVars(range(2 * d), vtype=GRB.INTEGER, name="x")
            # alpha = model.addVar( vtype = GRB.CONTINUOUS, name="alpha" )
            # Add constraints
            model.addConstrs(
                (gp.quicksum((x[k] - x[d + k]) * L[j, k] for k in range(d)) \
                 == gp.quicksum(v[k] * L[j, k] for k in range(d))
                 for j in range(d - d_cur, d)),
                name="c3")
            # # Set initial point
            # for i in range(d):
            #     x[i].start = max(v[i],0)
            #     x[i+d].start = -min(v[i],0)
            # model.update()
            # Set the objective function as constant
            model.setObjective(0, GRB.MAXIMIZE)
            # Solve the feasibility problem
            model.optimize()

            z = np.zeros((d,))
            for i in range(d):
                z[i] = x[i].X - x[i + d].X
            Z[d - d_cur, :] = z

        # Construct the hyperplane v^T y = v_y
        v_y = np.sum((v - z) * z_k) + round(np.sum(z * z_k))

        # Update the point set
        y_set = y_set - v.reshape((d, 1)) @ ((v.reshape((d, 1)).T @ y_set - v_y) / np.sum(v * v))
        # Remove outside points
        y_min = np.min(y_set, axis=0) - 1
        y_max = N - np.max(y_set, axis=0)
        violation = np.min(A @ y_set - b, axis=0)
        check = np.minimum(np.minimum(violation, y_min), y_max)
        y_set = y_set[:, check >= 0]

        # Check if intersection is empty
        if y_set.shape[1] == 0:
            early_stop = True
            break

        # Estimate the centroid covarance matrix
        y_bar = np.mean(y_set, axis=1, keepdims=True)
        temp = y_set - y_bar
        Y = np.zeros((d, d))
        for i in range(y_set.shape[1]):
            Y += (temp[:, i:i + 1] @ temp[:, i:i + 1].T)
        Y /= y_set.shape[1]
        # Remove negative eigenvalues
        u = min(np.min(np.linalg.eigvalsh(Y)), 0)
        Y -= u * np.eye(d)

        # Update the uniform distribution in P
        C, z_k, y_set = next(randomwalk_approximator(F, Y, y_set, A, b, params, True))
        # Remove negative eigenvalues
        u = min(np.min(np.linalg.eigvalsh(C)), 0)
        C -= u * np.eye(d)

        # Project the lattice basis onto the subspace
        # L[d-d_cur+1:,:] = L[d-d_cur+1:,:] - L[d-d_cur+1:,:] @ v.reshape((d,1))\
        #                                 @ v.reshape((d,1)).T / np.sum(v*v)
        # Compute a basis by Hermite normal form
        _, R = column_style_hermite_normal_form(Z[:d - d_cur + 1, :])
        V = R[:, d - d_cur + 1:]
        L[d - d_cur + 1:, :] = (V @ np.linalg.inv(V.T @ V)).T
        d_cur -= 1

    # If no early stopping
    if not early_stop:
        # Solve the one-dimensional problem
        v = L[-1, :]  # Direction of the line
        y_bar = np.mean(y_set, axis=1)  # Point on the line
        # print(v,y_bar)

        # Find an integral point on the line
        # Create a new model
        model = gp.Model("search")
        model.Params.OutputFlag = 0  # Controls output
        # model.Params.MIPGap = 1e-9
        # Variables
        x = model.addVars(range(d), vtype=GRB.INTEGER, ub=N, lb=1, name="x")
        alpha = model.addVars(range(2), vtype=GRB.CONTINUOUS, name="alpha")
        # Add constraints
        model.addConstrs(
            (y_bar[k] + (alpha[0] - alpha[1]) * v[k] <= x[k] + 1e-3 for k in range(d)),
            name="c1")
        model.addConstrs(
            (y_bar[k] + (alpha[0] - alpha[1]) * v[k] >= x[k] - 1e-3 for k in range(d)),
            name="c2")
        # # Set initial point
        # for i in range(d):
        #     x[i].start = y_bar[i]
        # model.update()
        # Set the objective function as constant
        model.setObjective(0, GRB.MAXIMIZE)
        # Solve the feasibility problem
        model.optimize()

        z = np.zeros((d,))
        try:
            for i in range(d):
                z[i] = x[i].X

            # Find the upper and lower bound of one-dim problem
            bound = [0, 0]
            for i in range(N):
                flag = False
                if np.max(z + i * v) < N + 1 and np.min(z + i * v) > 0:
                    bound[1] = i
                    flag = True
                if np.max(z - i * v) < N + 1 and np.min(z - i * v) > 0:
                    bound[0] = -i
                    flag = True
                if not flag:
                    break

            # Shift to a problem with leftmost point 1
            z = z + (bound[0] - 1) * v
            M = bound[1] - bound[0] + 1
            # Define a one-dimensional problem
            G = lambda alpha: float(F(z + alpha[0] * v))
            params_new = params.copy()
            params_new["N"] = M
            params_new["d"] = 1
            params_new["eps"] = eps * 4
            params_new["delta"] = delta / 4
            # Use the uniform solver to solve the one-dim problem
            # output_uniform = UniformSolver(G, params_new)
            output_uniform = adaptive_solver(G, params_new)
            # Update the total number of points
            total_samples += output_uniform["total"]
            # Optimal point
            x_uni = z + output_uniform["x_opt"] * v

            # Estimate the empirical mean of x_opt
            num_samples = required_samples(delta / 2, eps / 4, params)
            hat_F = 0
            for i in range(num_samples):
                hat_F = hat_F + float(F(x_uni))
            hat_F /= num_samples

            s = np.concatenate((x_uni.reshape((d, 1)), [[hat_F]]), axis=0)
            S = np.concatenate((S, s), axis=1)

        except AttributeError:
            print("One-dim problem failed.")
            print(y_bar, L[-1, :])

    # Find the point with minimal empirical mean in S
    i_min = np.argmin(S[-1, :])
    x_bar = S[:-1, i_min]
    # Round to an integral solution
    x_opt = Round(F, x_bar, params)["x_opt"]

    # Stop timing
    stop_time = time.time()

    return {"x_opt": x_opt, "time": stop_time - start_time, "total": total_samples}


def randomwalk_approximator(F, Y, y_in, A_in, b_in, params, centroid=False):
    """
    Output the new polytope, its approximate centroid and approximate covariance matrix.
    
    Y: approximate covariance matrix
    y_set: warm-up distribution in P
    A_in, b_in: polytope
    """

    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    eps = params["eps"] if "eps" in params else 1
    delta = params["delta"] if "delta" in params else 1e-6
    A = np.copy(A_in)
    b = np.copy(b_in)
    y_set = np.copy(y_in)

    # Generate the uniform distribution in P
    M = math.ceil(2 * 20 * d * math.log(d + 1) * max(20.0, math.log(d)))
    y_set = random_walk(y_set, Y, A, b, params, M)
    # Approximate centroid
    y_bar = np.mean(y_set, axis=1, keepdims=True)

    # Only need to update the uniform distribution
    if centroid:
        # print(y_bar)
        temp = y_set - y_bar
        Y = np.zeros((d, d))
        for i in range(y_set.shape[1]):
            Y += (temp[:, i:i + 1] @ temp[:, i:i + 1].T)
        Y /= y_set.shape[1]
        # print(Y, y_bar[:,0], y_set.shape)
        # Remove negative eigenvalues
        u = min(np.min(np.linalg.eigvalsh(Y)), 0)
        Y -= u * np.eye(d)
        yield Y, y_bar[:, 0], y_set
        return

    # Constantly generate polytopes
    while True:
        # Separation oracle
        so = SO(F, y_bar[:, 0], eps * cst * min(N, N), delta / 4, params)
        c = -so["hat_grad"]
        hat_F = so["hat_F"]
        s = np.concatenate((y_bar, [[hat_F]]), axis=0)

        # Update A and b
        c = np.reshape(c, (1, d))
        A = np.concatenate((A, c), axis=0)
        b = np.concatenate((b, c @ y_bar), axis=0)

        # Warm-start distribution
        violation = np.min(A[-1:, :] @ y_set - b[-1:, :], axis=0)
        y_set = y_set[:, violation >= 0]

        # Infeasible
        if y_set.shape[1] == 0:
            yield None, None, None, None, None, None

        # Estimate the covarance matrix
        y_bar = np.mean(y_set, axis=1, keepdims=True)
        temp = y_set - y_bar
        Y = np.zeros((d, d))
        for i in range(y_set.shape[1]):
            Y += (temp[:, i:i + 1] @ temp[:, i:i + 1].T)
        Y /= y_set.shape[1]
        # Remove negative eigenvalues
        u = min(np.min(np.linalg.eigvalsh(Y)), 0)
        Y -= u * np.eye(d)

        # Update uniform distribution in P
        y_set = random_walk(y_set, Y, A, b, params, M)

        # Approximate centroid and covariance
        M_new = int(y_set.shape[1] / 2)
        y_set = y_set[:, np.random.permutation(np.arange(y_set.shape[1]))]
        # M_new = M
        y_bar = np.mean(y_set[:, :M_new], axis=1, keepdims=True)
        temp = y_set[:, :M_new] - y_bar
        Y = np.zeros((d, d))
        for i in range(M_new):
            Y += (temp[:, i:i + 1] @ temp[:, i:i + 1].T)
        Y /= M_new
        # Remove negative eigenvalues
        u = min(np.min(np.linalg.eigvalsh(Y)), 0)
        Y -= u * np.eye(d)

        # Update point set
        y_set = y_set[:, M_new:]

        # Output
        yield Y, y_bar[:, 0], A, b, y_set, s


def dimension_reduction_proj_solver(F, params):
    """
    Dimension reduction method for multi-dim problems with capacity constraint.
    """

    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    # sigma = params["sigma"] if "sigma" in params else 1
    eps = params["eps"] if "eps" in params else 1
    delta = params["delta"] if "delta" in params else 1e-6
    # The capacity constraint
    K_val = params["K"] if "K" in params else N * d

    # # Parameters of the algorithm
    # eta = 1e-2 # eps in paper

    # Start timing
    start_time = time.time()
    # Count simulation runs
    total_samples = 0
    # Record points where SO is called and their empirical means
    S = np.zeros((d + 1, 0))
    # Simulation cost of each separation oracle
    so_samples = required_samples(delta / 4, eps / 8 / np.sqrt(d) * params["eta"], params)

    # The current dimension
    d_cur = d
    # The basis of integral lattice
    L = np.eye(d)
    # The pre-images
    Z = np.zeros((d, d))

    # Record polytope Ax >= b
    # Basic block
    E = np.eye(d)
    E_inv = np.eye(d)
    for i in range(d - 1):
        E[i + 1, i] = -1
        for j in range(i + 1):
            E_inv[i + 1, j] = 1
    A = np.zeros((2 * d + 1, d))
    A[:d, :] = E
    A[d:2 * d, :] = -E
    A[-1, -1] = -1
    b = np.concatenate((np.ones((d, 1)) * (1 + 1e-7),
                        np.ones((d, 1)) * (-N + 1e-7),
                        [[-K_val]]
                        ))
    # The initial uniform distribution in P
    y_set = E_inv @ np.random.uniform(1, N, (d, math.ceil(50 * 20 * d * math.log(d + 1) * max(20.0, math.log(d)))))
    # The initial ellipsoid
    # C_inv = (N-1)**(-2) * 12 * np.eye(d) # A in paper
    C = (N - 1) ** 2 / 12 * (E_inv @ E_inv.T)  # A_inv in paper

    # Initial centroid
    x = np.ones((d,)) * (N + 1) / 2
    z = np.cumsum(x)

    # Early stopping
    early_stop = False
    # Iteratively solve d_cur-dimensional problems
    while d_cur > 1:

        print(d_cur)
        # The current basis
        L_cur = np.copy(L[d - d_cur:, :])

        # Iterate until we find a short basis vector
        # Use random walk method
        for K, z_new, A_new, b_new, y_new, s, total in random_walk_proj_approximator(F,
                                                                              C, y_set, A,
                                                                              b, params):
            # Check early stopping
            if K is None:
                early_stop = True
                break

            u = np.linalg.eigvalsh(K)
            if np.sum(u > 1e-10) < d_cur:
                break

            # Number of samples
            # total_samples += so_samples * (2 * d)
            total_samples += total
            print(total, so_samples * 2 * d)
            # Update set S
            S = np.concatenate((S, s), axis=1)

            # The LLL algorithm
            basis = LLL(L_cur, K)
            # Choose the shortest vector
            norm = np.diag((basis @ K) @ basis.T)
            print(np.min(norm))

            # Stopping criterion
            if np.min(norm) < 1e1 / d_cur ** 2:
                i_short = np.argmin(norm)
                L_cur[0, :] = basis[i_short, :]
                # Update the basis and point set
                y_set = y_new
                # Update A and b
                A, b = A_new, b_new
                # Update centroid
                z_k = z_new
                break

        # Check if intersection is empty
        if early_stop:
            break

        # Dimension reduction
        v = L_cur[0, :]

        # Find the pre-image
        if d_cur == d:
            z = v
            Z[0, :] = v
        else:
            # Solve for z
            # Create a new model
            model = gp.Model("pre-image")
            model.Params.OutputFlag = 0  # Controls output
            # model.Params.MIPGap = 1e-9
            # Variables
            x = model.addVars(range(2 * d), vtype=GRB.INTEGER, name="x")
            # Add constraints
            model.addConstrs(
                (gp.quicksum((x[k] - x[d + k]) * L[j, k] for k in range(d)) \
                 == gp.quicksum(v[k] * L[j, k] for k in range(d))
                 for j in range(d - d_cur, d)),
                name="c3")
            # # Set initial point
            # for i in range(d):
            #     x[i].start = max(v[i],0)
            #     x[i+d].start = -min(v[i],0)
            # model.update()
            # Set the objective function as constant
            model.setObjective(0, GRB.MAXIMIZE)
            # Solve the feasibility problem
            model.optimize()

            z = np.zeros((d,))
            for i in range(d):
                z[i] = x[i].X - x[i + d].X
            Z[d - d_cur, :] = z

        # Construct the hyperplane v^T y = v_y
        v_y = np.sum((v - z) * z_k) + round(np.sum(z * z_k))

        # Update the point set
        y_set = y_set - v.reshape((d, 1)) @ ((v.reshape((d, 1)).T @ y_set - v_y)
                                             / np.sum(v * v))
        check = np.min(A @ y_set - b, axis=0)
        y_set = y_set[:, check >= 0]

        # Check if intersection is empty
        if y_set.shape[1] == 0:
            early_stop = True
            break

        # Estimate the centroid covarance matrix
        y_bar = np.mean(y_set, axis=1, keepdims=True)
        temp = y_set - y_bar
        Y = np.zeros((d, d))
        for i in range(y_set.shape[1]):
            Y += (temp[:, i:i + 1] @ temp[:, i:i + 1].T)
        Y /= y_set.shape[1]
        # Remove negative eigenvalues
        u = min(np.min(np.linalg.eigvalsh(Y)), 0)
        Y -= u * np.eye(d)

        # Update the uniform distribution in P
        C, z_k, y_set = next(random_walk_proj_approximator(F, Y, y_set, A, b, params, True))
        # Remove negative eigenvalues
        u = min(np.min(np.linalg.eigvalsh(C)), 0)
        C -= u * np.eye(d)

        # Project the lattice basis onto the subspace
        # L[d-d_cur+1:,:] = L[d-d_cur+1:,:] - L[d-d_cur+1:,:] @ v.reshape((d,1))\
        #                                 @ v.reshape((d,1)).T / np.sum(v*v)
        # Compute a basis by Hermite normal form
        _, R = column_style_hermite_normal_form(Z[:d - d_cur + 1, :])
        V = R[:, d - d_cur + 1:]
        L[d - d_cur + 1:, :] = (V @ np.linalg.inv(V.T @ V)).T
        d_cur -= 1

    # If no early stopping
    if not early_stop:
        # Solve the one-dimensional problem
        v = L[-1, :]  # Direction of the line
        v /= np.min(np.abs(v[v != 0]))
        y_bar = np.mean(y_set, axis=1)  # Point on the line

        # Find an integral point on the line
        # Create a new model
        model = gp.Model("search")
        model.Params.OutputFlag = 0  # Controls output
        # model.Params.MIPGap = 1e-9
        # Variables
        x = model.addVars(range(d), vtype=GRB.INTEGER, ub=K_val, lb=1, name="x")
        alpha = model.addVars(range(2), vtype=GRB.CONTINUOUS, name="alpha")
        # Add constraints
        model.addConstrs(
            (y_bar[k] + (alpha[0] - alpha[1]) * v[k] <= x[k] + 1e-7 for k in range(d)),
            name="c1")
        model.addConstrs(
            (y_bar[k] + (alpha[0] - alpha[1]) * v[k] >= x[k] - 1e-7 for k in range(d)),
            name="c2")
        model.addConstrs(
            (x[k + 1] - x[k] >= 1 for k in range(d - 1)),
            name="c3")
        model.addConstrs(
            (x[k + 1] - x[k] <= N for k in range(d - 1)),
            name="c4")
        model.addConstr(
            x[0] >= 1,
            name="c5")
        model.addConstr(
            x[0] <= N,
            name="c6")
        model.addConstr(
            x[d - 1] <= K_val,
            name="c7")
        # # Set initial point
        # for i in range(d):
        #     x[i].start = y_bar[i]
        # model.update()
        # Set the objective function as constant
        model.setObjective(0, GRB.MAXIMIZE)
        # Solve the feasibility problem
        model.optimize()

        z = np.zeros((d,))
        try:
            for i in range(d):
                z[i] = x[i].X

            # Find the upper and lower bound of one-dim problem
            bound = [0, 0]
            for i in range(N):
                flag = False
                y = E @ (z + i * v)
                if np.max(y) < N + 1 and np.min(y) > 0:
                    bound[1] = i
                    flag = True
                y = E @ (z - i * v)
                if np.max(y) < N + 1 and np.min(y) > 0:
                    bound[0] = -i
                    flag = True
                if not flag:
                    break

            # Shift to a problem with leftmost point 1
            z = z + (bound[0] - 1) * v
            M = bound[1] - bound[0] + 1
            # Define an one-dimensional problem
            G = lambda kappa: float(F(z + kappa[0] * v))
            params_new = params.copy()
            params_new["N"] = M
            params_new["d"] = 1
            params_new["eps"] = eps
            params_new["delta"] = delta / 4
            # Use the uniform solver to solve the one-dim problem
            output_uniform = uniform_solver(G, params_new)
            # Update the total number of points
            total_samples += output_uniform["total"]
            # Optimal point
            x_uni = z + output_uniform["x_opt"] * v

            # Estimate the empirical mean of x_opt
            num_samples = required_samples(delta / 2, eps / 4, params)
            hat_F = 0
            for i in range(num_samples):
                hat_F = hat_F + float(F(x_uni))
            hat_F /= num_samples

            s = np.concatenate((x_uni.reshape((d, 1)), [[hat_F]]), axis=0)
            S = np.concatenate((S, s), axis=1)

        except AttributeError:
            print("One-dim problem failed.")
            print(y_bar, L[-1, :])

    # Find the point with minimal empirical mean in S
    i_min = np.argmin(S[-1, :])
    x_bar = S[:-1, i_min]
    # Round to an integral solution
    x_opt = RoundCons(F, x_bar, params)["x_opt"]

    # Stop timing
    stop_time = time.time()

    return {"x_opt": x_opt, "time": stop_time - start_time, "total": total_samples}


def random_walk_proj_approximator(F, Y, y_in, A_in, b_in, params, centroid=False):
    """
    Output the new polytope, its approximate centroid and approximate covariance matrix.
    
    Y: approximate covariance matrix
    y_set: warm-up distribution in P
    A_in, b_in: polytope
    """

    # Retrieve parameters
    d = params["d"] if "d" in params else 1
    N = params["N"] if "N" in params else 2
    eps = params["eps"] if "eps" in params else 1
    delta = params["delta"] if "delta" in params else 1e-6
    A = np.copy(A_in)
    b = np.copy(b_in)
    y_set = np.copy(y_in)

    # Generate the uniform distribution in P
    M = math.ceil(50 * 20 * d * math.log(d + 1) * max(20.0, math.log(d)))
    y_set = random_proj_walk(y_set, Y, A, b, params, M)
    # Approximate centroid
    y_bar = np.mean(y_set, axis=1, keepdims=True)

    # Only need to update the uniform distribution
    if centroid:
        temp = y_set - y_bar
        Y = np.zeros((d, d))
        for i in range(y_set.shape[1]):
            Y += (temp[:, i:i + 1] @ temp[:, i:i + 1].T)
        Y /= y_set.shape[1]
        # Remove negative eigenvalues
        u = min(np.min(np.linalg.eigvalsh(Y)), 0)
        Y -= u * np.eye(d)
        yield Y, y_bar[:, 0], y_set
        return

    # Constantly generate polytopes
    while True:

        # Separation oracle
        so = SOCons(F, y_bar[:, 0], eps / 4 * min(N, N) * params["eta"], delta / 4, params)
        c = -so["hat_grad"]
        hat_F = so["hat_F"]
        s = np.concatenate((y_bar, [[hat_F]]), axis=0)

        # Update A and b
        c = np.reshape(c, (1, d))
        A = np.concatenate((A, c), axis=0)
        b = np.concatenate((b, c @ y_bar), axis=0)

        # Warm-start distribution
        violation = np.min(A @ y_set - b, axis=0)
        y_set = y_set[:, violation >= 0]

        # Infeasible
        if y_set.shape[1] == 0:
            yield None, None, None, None, None, None

        # Estimate the covarance matrix
        y_bar = np.mean(y_set, axis=1, keepdims=True)
        temp = y_set - y_bar
        Y = np.zeros((d, d))
        for i in range(y_set.shape[1]):
            Y += (temp[:, i:i + 1] @ temp[:, i:i + 1].T)
        Y /= y_set.shape[1]
        # Remove negative eigenvalues
        u = min(np.min(np.linalg.eigvalsh(Y)), 0)
        Y -= u * np.eye(d)

        # Update uniform distribution in P
        M = math.ceil(5 * 20 * d * math.log(d + 1) * max(20.0, math.log(d)))
        y_set = random_proj_walk(y_set, Y, A, b, params, M)

        # Approximate centroid and covariance
        M = int(y_set.shape[1] / 2)
        y_bar = np.mean(y_set[:, :M], axis=1, keepdims=True)
        temp = y_set[:, :M] - y_bar
        Y = np.zeros((d, d))
        for i in range(M):
            Y += (temp[:, i:i + 1] @ temp[:, i:i + 1].T)
        Y /= M
        # Remove negative eigenvalues
        u = min(np.min(np.linalg.eigvalsh(Y)), 0)
        Y -= u * np.eye(d)

        # Update point set
        y_set = y_set[:, M:]

        # Output
        yield Y, y_bar[:, 0], A, b, y_set, s, so["total"]
