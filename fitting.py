"""
fitting.py
Funzioni di fitting: spline, lorentziana vincolata, sigmoide vincolata.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar, minimize
from typing import Dict, Any
from termcolor import colored

from constants import N_POINTS_FIT

def spline_fit(x, y, x_fit=None, n_points=N_POINTS_FIT) -> Dict[str, Any]:
    """
    Parameters
    ----------
    x, y : array-like
        Data points.
    x_fit : array-like, optional
        Pre‑computed x values for the fitted curve. If None, generate with np.linspace.
    n_points : int
        Only used if x_fit is None.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    fit_successful = False
    y_fit = None
    fit_label = ""
    try:
        spline = PchipInterpolator(x, y)
        if x_fit is None:
            x_fit = np.linspace(x.min(), x.max(), n_points)
        y_fit = spline(x_fit)
        fit_successful = True
        fit_label = "Spline Fit"
    except Exception as e:
        print(f"Spline fit failed: {e}")
    
    return {
            'x': x, 
            'y': y,
            'x_fit': x_fit, 
            'y_fit': y_fit,
            'fit_label': fit_label, 
            'fit_successful': fit_successful
        }    

def constrained_lorentzian(x, A, gamma, y_min):
    if gamma == 0.0:
        return np.full_like(x, A)
    return A - (A - y_min) * gamma**2 / (gamma**2 + x**2)

def estimate_constrained_lorentzian(x_data, y_data):
    x = np.asarray(x_data)
    y = np.asarray(y_data)
    y_min = np.min(y)
    y_max = np.max(y)
    if y_max == y_min:
        return y_max, 0.0
    def error_for_A(A):
        if A < y_max:
            return np.inf
        gamma_max = np.inf
        for xi, yi in zip(x, y):
            if yi <= y_min:
                continue
            bound_sq = (A - yi) / (yi - y_min) * xi**2
            if bound_sq <= 0:
                return np.inf
            gamma_max = min(gamma_max, np.sqrt(bound_sq))
        if gamma_max <= 0.0:
            return np.inf
        def mse(gamma):
            if gamma == 0.0:
                y_pred = np.full_like(x, A)
            else:
                y_pred = A - (A - y_min) * gamma**2 / (gamma**2 + x**2)
            return np.sum((y_pred - y)**2)
        res = minimize_scalar(mse, bounds=(0.0, gamma_max), method='bounded')
        return res.fun
    upper_A = y_max + 5 * (y_max - y_min) if y_max > y_min else y_max + 1.0
    res_A = minimize_scalar(error_for_A, bounds=(y_max, upper_A), method='bounded')
    best_A = res_A.x
    gamma_max = np.inf
    for xi, yi in zip(x, y):
        if yi <= y_min:
            continue
        bound_sq = (best_A - yi) / (yi - y_min) * xi**2
        gamma_max = min(gamma_max, np.sqrt(bound_sq))
    def mse(gamma):
        if gamma == 0.0:
            y_pred = np.full_like(x, best_A)
        else:
            y_pred = best_A - (best_A - y_min) * gamma**2 / (gamma**2 + x**2)
        return np.sum((y_pred - y)**2)
    res_gamma = minimize_scalar(mse, bounds=(0.0, gamma_max), method='bounded')
    best_gamma = res_gamma.x
    return best_A, best_gamma

def constrained_sigmoid(x, L, R, tau, x0=0.0):
    return R + (L - R) / (1.0 + np.exp(-(x - x0) / tau))

def estimate_constrained_sigmoid(x_data, y_data, fix_center=True, x0_fixed=0.0):
    x = np.asarray(x_data)
    y = np.asarray(y_data)
    x0 = x0_fixed
    def solve_LR_for_tau(tau):
        z = 1.0 / (1.0 + np.exp(-(x - x0) / tau))
        def mse(params):
            L, R = params
            y_pred = L * z + R * (1 - z)
            return np.sum((y_pred - y)**2)
        constraints = []
        for i in range(len(x)):
            A_i = np.array([z[i], 1 - z[i]])
            b_i = y[i]
            constraints.append({'type': 'ineq', 'fun': lambda p, A=A_i, b=b_i: A[0]*p[0] + A[1]*p[1] - b})
        mask_left = x > x0
        mask_right = x < x0
        L0 = np.max(y[mask_right]) if np.any(mask_right) else np.max(y)
        R0 = np.max(y[mask_left]) if np.any(mask_left) else np.max(y)
        res = minimize(mse, [L0, R0], method='SLSQP', constraints=constraints,
                       bounds=[(0, None), (0, None)], options={'maxiter': 1000})
        if res.success:
            return res.x[0], res.x[1], res.fun
        else:
            L = R = np.max(y)
            return L, R, np.sum((np.full_like(y, L) - y)**2)
    def objective_tau(tau):
        if tau <= 0:
            return np.inf
        _, _, err = solve_LR_for_tau(tau)
        return err
    tau_min = 1e-6
    tau_max = np.ptp(x) * 10
    res_tau = minimize_scalar(objective_tau, bounds=(tau_min, tau_max), method='bounded')
    if res_tau.success:
        tau_opt = res_tau.x
    else:
        tau_opt = np.ptp(x) / 4
        print(colored(f"Warning: optimization for tau failed, using fallback tau={tau_opt:.4f}", "yellow", attrs=["bold"]))
    L_opt, R_opt, _ = solve_LR_for_tau(tau_opt)
    return L_opt, R_opt, tau_opt
