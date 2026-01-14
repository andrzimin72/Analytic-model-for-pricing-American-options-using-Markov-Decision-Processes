"""
Enhanced American Option Pricer with:
- Market calibration (implied volatility)
- Greeks (Delta, Gamma, Theta)
- Automatic CSV/PNG output
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import root_scalar
import os
from datetime import datetime

# Import your existing functions here (or define them in this file)
# For brevity, we assume `american_option_mdp` is available
# Make sure to include its full definition from earlier

# ----------------------------------------------------------------------
# 1. Greeks from MDP Value Function
# ----------------------------------------------------------------------
def compute_greeks_mdp(mdp_result, S0, K, T, r, q, sigma, N_steps, option_type='put', h=0.01):
    """
    Compute Greeks using finite differences on the MDP value function.
    
    Parameters:
    - mdp_result: output from `american_option_mdp`
    - h: perturbation size (e.g., 1% of S0 for Delta)
    """
    grid = mdp_result['grid']
    V = mdp_result['V']  # Shape: (S, N+1)
    dt = mdp_result['dt']
    
    # Find index of S0
    s0_idx = np.argmin(np.abs(grid - S0))
    
    # Delta: dV/dS
    if s0_idx == 0:
        delta = (V[s0_idx + 1, 0] - V[s0_idx, 0]) / (grid[s0_idx + 1] - grid[s0_idx])
    elif s0_idx == len(grid) - 1:
        delta = (V[s0_idx, 0] - V[s0_idx - 1, 0]) / (grid[s0_idx] - grid[s0_idx - 1])
    else:
        delta = (V[s0_idx + 1, 0] - V[s0_idx - 1, 0]) / (grid[s0_idx + 1] - grid[s0_idx - 1])
    
    # Gamma: d²V/dS²
    if 0 < s0_idx < len(grid) - 1:
        gamma = (V[s0_idx + 1, 0] - 2 * V[s0_idx, 0] + V[s0_idx - 1, 0]) / ((grid[s0_idx + 1] - grid[s0_idx])**2)
    else:
        gamma = 0.0
    
    # Theta: -dV/dt (approximate using first time step)
    theta = -(V[s0_idx, 1] - V[s0_idx, 0]) / dt if V.shape[1] > 1 else 0.0
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'price': V[s0_idx, 0]
    }

# ----------------------------------------------------------------------
# 2. Calibration to Market Price (Implied Volatility)
# ----------------------------------------------------------------------
def calibrate_implied_volatility(
    market_price, S0, K, T, r, q, N_steps, option_type='put',
    sigma_guess=0.2, sigma_bounds=(0.01, 2.0), tol=1e-4
):
    """
    Calibrate implied volatility by matching MDP price to market price.
    """
    def price_error(sigma):
        try:
            res = american_option_mdp(S0, K, T, r, q, sigma, N_steps, option_type)
            return res['price'] - market_price
        except Exception:
            return np.inf  # penalize invalid sigma
    
    sol = root_scalar(price_error, bracket=sigma_bounds, method='brentq', xtol=tol)
    if sol.converged:
        return sol.root
    else:
        raise RuntimeError("Calibration failed to converge")

# ----------------------------------------------------------------------
# 3. Automatic Saving Utilities
# ----------------------------------------------------------------------
def save_results_to_csv(results, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"american_option_results_{timestamp}.csv"
    df = pd.DataFrame([results])
    df.to_csv(filename, index=False)
    print(f"Results saved to: {filename}")

def plot_and_save_exercise_boundary(mdp_result, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exercise_boundary_{timestamp}.png"
    
    V = mdp_result['V']
    policy = mdp_result['policy']
    grid = mdp_result['grid']
    T = mdp_result['T']
    dt = mdp_result['dt']
    opt_type = mdp_result['type']
    
    boundary, times = [], []
    for t in range(policy.shape[1]):
        exercised = policy[:, t] == 1
        if np.any(exercised):
            idx = np.where(exercised)[0][-1] if opt_type == 'put' else np.where(exercised)[0][0]
            boundary.append(grid[idx])
            times.append(T - t * dt)
    
    plt.figure(figsize=(8,5))
    plt.plot(times, boundary, 'o-', label='Exercise Boundary')
    plt.xlabel('Time to Maturity')
    plt.ylabel('Stock Price')
    plt.title(f'Early Exercise Boundary ({opt_type.capitalize()} Option)')
    plt.grid(True)
    plt.legend()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {filename}")

# ----------------------------------------------------------------------
# 4. Full Workflow Example
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Input parameters
    S0, K, T = 100, 100, 1.0
    r, q = 0.05, 0.02
    N_steps = 50
    option_type = 'put'
    market_price = 6.90  # hypothetical market price

    # Step 1: Calibrate implied volatility
    sigma_imp = calibrate_implied_volatility(
        market_price, S0, K, T, r, q, N_steps, option_type
    )
    print(f"Implied volatility: {sigma_imp:.4f}")

    # Step 2: Re-price with calibrated sigma
    mdp_res = american_option_mdp(S0, K, T, r, q, sigma_imp, N_steps, option_type)

    # Step 3: Compute Greeks
    greeks = compute_greeks_mdp(mdp_res, S0, K, T, r, q, sigma_imp, N_steps, option_type)

    # Step 4: Prepare full result dictionary
    full_result = {
        'S0': S0,
        'K': K,
        'T': T,
        'r': r,
        'q': q,
        'sigma': sigma_imp,
        'option_type': option_type,
        'market_price': market_price,
        'model_price': greeks['price'],
        'delta': greeks['delta'],
        'gamma': greeks['gamma'],
        'theta': greeks['theta'],
        'calibration_error': abs(greeks['price'] - market_price)
    }

    # Step 5: Save outputs
    save_results_to_csv(full_result)
    plot_and_save_exercise_boundary(mdp_res)