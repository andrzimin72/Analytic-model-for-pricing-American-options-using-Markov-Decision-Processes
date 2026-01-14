import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import root_scalar
from typing import Dict, Literal
from .pricers import american_option_mdp

def compute_greeks_mdp(mdp_result: Dict, S0: float, h: float = 0.01) -> Dict[str, float]:
    grid = mdp_result["grid"]
    V = mdp_result["V"]
    s0_idx = np.argmin(np.abs(grid - S0))
    
    if s0_idx == 0:
        delta = (V[s0_idx + 1, 0] - V[s0_idx, 0]) / (grid[s0_idx + 1] - grid[s0_idx])
    elif s0_idx == len(grid) - 1:
        delta = (V[s0_idx, 0] - V[s0_idx - 1, 0]) / (grid[s0_idx] - grid[s0_idx - 1])
    else:
        delta = (V[s0_idx + 1, 0] - V[s0_idx - 1, 0]) / (grid[s0_idx + 1] - grid[s0_idx - 1])
    
    if 0 < s0_idx < len(grid) - 1:
        gamma = (V[s0_idx + 1, 0] - 2 * V[s0_idx, 0] + V[s0_idx - 1, 0]) / ((grid[s0_idx + 1] - grid[s0_idx])**2)
    else:
        gamma = 0.0
    
    return {"delta": float(delta), "gamma": float(gamma), "price": float(V[s0_idx, 0])}

def calibrate_implied_volatility(
    market_price: float,
    S0: float, K: float, T: float, r: float, q: float,
    N_steps: int, option_type: Literal["put", "call"] = "put",
    sigma_guess: float = 0.2, tol: float = 1e-4
):
    def error_fn(sigma):
        try:
            res = american_option_mdp(S0, K, T, r, q, sigma, N_steps, option_type)
            return res["price"] - market_price
        except:
            return 1e6
    sol = root_scalar(error_fn, bracket=[0.01, 2.0], method="brentq", xtol=tol)
    return sol.root if sol.converged else sigma_guess

def plot_exercise_boundary(mdp_result, save_path: str = None):
    if "grid" not in mdp_result:
        print("Only 1D MDP supports boundary plot.")
        return
    grid = mdp_result["grid"]
    policy = mdp_result["policy"]
    T = mdp_result["T"]
    dt = mdp_result["dt"]
    opt_type = mdp_result["type"]
    
    boundary, times = [], []
    for t in range(policy.shape[1]):
        exercised = policy[:, t] == 1
        if np.any(exercised):
            idx = np.where(exercised)[0][-1] if opt_type == "put" else np.where(exercised)[0][0]
            boundary.append(grid[idx])
            times.append(T - t * dt)
    
    plt.figure(figsize=(8, 5))
    plt.plot(times, boundary, "o-", color="red")
    plt.xlabel("Time to Maturity")
    plt.ylabel("Stock Price")
    plt.title(f"Early Exercise Boundary ({opt_type.capitalize()} Option)")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

def plot_value_function(mdp_result, save_path: str = None):
    V = mdp_result["V"]
    grid = mdp_result["grid"]
    T = mdp_result["T"]
    time_axis = np.linspace(0, T, V.shape[1])
    S_mesh, T_mesh = np.meshgrid(time_axis, grid)
    plt.figure(figsize=(9, 6))
    plt.contourf(T_mesh, S_mesh, V, levels=30, cmap="viridis")
    plt.colorbar(label="Option Value")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("Value Function $V(S, t)$")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

def plot_policy_map(mdp_result, save_path: str = None):
    policy = mdp_result["policy"]
    grid = mdp_result["grid"]
    T = mdp_result["T"]
    time_axis = np.linspace(0, T, policy.shape[1])
    S_mesh, T_mesh = np.meshgrid(time_axis, grid)
    plt.figure(figsize=(9, 6))
    plt.pcolormesh(T_mesh, S_mesh, policy, shading="auto", cmap="RdYlBu_r")
    plt.colorbar(label="Action (1=Exercise, 0=Continue)")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("Optimal Policy Map")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

def save_results_to_csv(results: dict, filename: str = None):
    if filename is None:
        from datetime import datetime
        filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame([results]).to_csv(filename, index=False)
    print(f"Saved to {filename}")