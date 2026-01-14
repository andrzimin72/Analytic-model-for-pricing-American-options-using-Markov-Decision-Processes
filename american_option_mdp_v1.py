"""
American Option Pricing using Markov Decision Processes (MDP)

This module implements American option pricing by framing the problem
as a finite-horizon MDP, solved using mdptoolbox. It supports:
- Continuous dividend yield
- Call and put options
- Stochastic volatility (via discretized 2D state space)
- Visualization of early-exercise boundaries
- Benchmarking against binomial, trinomial, and LSM Monte Carlo

Dependencies:
- numpy
- scipy
- matplotlib
- mdptoolbox
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.sparse import lil_matrix
from mdptoolbox.mdp import FiniteHorizon


# ----------------------------------------------------------------------
# 1. Core MDP Pricer (Deterministic Volatility)
# ----------------------------------------------------------------------
def american_option_mdp(
    S0, K, T, r, q, sigma, N_steps,
    option_type='put',
    S_min_factor=0.2,
    S_max_factor=3.0,
    N_states=200
):
    """
    Price American option using MDPtoolbox with constant volatility.
    """
    assert option_type in ('call', 'put'), "option_type must be 'call' or 'put'"
    
    # Discretize stock price
    S_min = S0 * S_min_factor
    S_max = S0 * S_max_factor
    price_grid = np.linspace(S_min, S_max, N_states)
    s0_idx = np.argmin(np.abs(price_grid - S0))
    
    dt = T / N_steps
    discount = np.exp(-r * dt)
    mu = r - q  # risk-neutral drift
    
    # Trinomial parameters
    dx = sigma * np.sqrt(3 * dt)
    u = np.exp(dx)
    d = np.exp(-dx)
    p_u = ((sigma**2 * dt + (mu * dt)**2) / (2 * dx**2)) + (mu * dt) / (2 * dx)
    p_d = ((sigma**2 * dt + (mu * dt)**2) / (2 * dx**2)) - (mu * dt) / (2 * dx)
    p_m = 1 - p_u - p_d
    
    if not (0 <= p_u <= 1 and 0 <= p_m <= 1 and 0 <= p_d <= 1):
        raise ValueError("Invalid trinomial probabilities — reduce dt or adjust parameters.")
    
    # Build transition matrices
    A = 2  # 0: continue, 1: exercise
    P = [lil_matrix((N_states, N_states)) for _ in range(A)]
    
    for s in range(N_states):
        S = price_grid[s]
        Su, Sm, Sd = S * u, S, S * d
        iu = np.argmin(np.abs(price_grid - Su))
        im = np.argmin(np.abs(price_grid - Sm))
        id_ = np.argmin(np.abs(price_grid - Sd))
        P[0][s, iu] += p_u
        P[0][s, im] += p_m
        P[0][s, id_] += p_d
        if s == 0:
            P[0][s, :] = 0; P[0][s, 0] = 1.0
        elif s == N_states - 1:
            P[0][s, :] = 0; P[0][s, -1] = 1.0
    P[1].setdiag(1.0)
    P = [p.tocsr() for p in P]
    
    # Rewards
    R = [np.zeros(N_states), np.zeros(N_states)]
    if option_type == 'call':
        R[1][:] = np.maximum(price_grid - K, 0.0)
    else:
        R[1][:] = np.maximum(K - price_grid, 0.0)
    
    # Solve
    fh = FiniteHorizon(P, R, discount, N_steps, skip_check=True)
    fh.run()
    
    V = np.array(fh.V)
    policy = np.array(fh.policy)
    
    return {
        'price': V[s0_idx, 0],
        'V': V,
        'policy': policy,
        'grid': price_grid,
        'dt': dt,
        'T': T,
        'type': option_type,
        'vol_type': 'constant'
    }


# ----------------------------------------------------------------------
# 2. MDP Pricer with Stochastic Volatility (2D State Space)
# ----------------------------------------------------------------------
def american_option_mdp_stoch_vol(
    S0, K, T, r, q, v0, kappa, theta, xi, rho,
    N_steps, option_type='put',
    S_min_factor=0.2, S_max_factor=3.0, N_S=50,
    v_min=0.01, v_max=0.5, N_v=20
):
    """
    Price American option with stochastic volatility (Heston-like).
    State = (S, v). Volatility follows mean-reverting square-root process.
    Discretized via finite differences on a 2D grid.
    """
    assert option_type in ('call', 'put')
    
    # Discretize state space
    S_grid = np.linspace(S0 * S_min_factor, S0 * S_max_factor, N_S)
    v_grid = np.linspace(v_min, v_max, N_v)
    SS, VV = np.meshgrid(S_grid, v_grid, indexing='ij')
    states = np.stack([SS.ravel(), VV.ravel()], axis=1)  # (N, 2)
    N_states = len(states)
    
    # Find initial state index
    dist = np.sqrt((states[:, 0] - S0)**2 + (states[:, 1] - v0)**2)
    s0_idx = np.argmin(dist)
    
    dt = T / N_steps
    discount = np.exp(-r * dt)
    
    # Actions: 0 = continue, 1 = exercise
    A = 2
    P = [lil_matrix((N_states, N_states)) for _ in range(A)]
    
    # Precompute transition probabilities for each state
    for i, (S, v) in enumerate(states):
        if v <= 0:
            v = 1e-6
        
        # Stock move: dS = (r - q) S dt + sqrt(v) S dW1
        # Vol move: dv = kappa(theta - v) dt + xi sqrt(v) dW2
        # Correlation: dW1 dW2 = rho dt
        
        # Mean and std of log(S_{t+1})
        mu_S = np.log(S) + (r - q - 0.5 * v) * dt
        sigma_S = np.sqrt(v * dt)
        
        # Mean and std of v_{t+1} (approximate Euler-Maruyama)
        mu_v = v + kappa * (theta - v) * dt
        sigma_v = xi * np.sqrt(v * dt)
        
        # Generate 3x3 local grid (trinomial in both dimensions)
        dS_vals = [mu_S - sigma_S, mu_S, mu_S + sigma_S]
        dv_vals = [mu_v - sigma_v, mu_v, mu_v + sigma_v]
        
        probs = np.array([
            [0.25 * (1 - rho), 0.25, 0.25 * (1 + rho)],
            [0.25, 0.0, 0.25],
            [0.25 * (1 + rho), 0.25, 0.25 * (1 - rho)]
        ])
        probs = np.maximum(probs, 0)
        probs /= probs.sum()
        
        for di in range(3):
            for dj in range(3):
                S_next = np.exp(dS_vals[di])
                v_next = dv_vals[dj]
                if v_next < v_min:
                    v_next = v_min
                elif v_next > v_max:
                    v_next = v_max
                
                # Find nearest grid point
                dist_next = np.sqrt((states[:, 0] - S_next)**2 + (states[:, 1] - v_next)**2)
                j = np.argmin(dist_next)
                P[0][i, j] += probs[di, dj]
    
    # Normalize rows
    for i in range(N_states):
        row_sum = P[0][i, :].sum()
        if row_sum > 0:
            P[0][i, :] /= row_sum
        else:
            P[0][i, i] = 1.0
    
    P[1].setdiag(1.0)
    P = [p.tocsr() for p in P]
    
    # Reward for exercise
    R = [np.zeros(N_states), np.zeros(N_states)]
    if option_type == 'call':
        R[1][:] = np.maximum(states[:, 0] - K, 0.0)
    else:
        R[1][:] = np.maximum(K - states[:, 0], 0.0)
    
    # Solve MDP
    fh = FiniteHorizon(P, R, discount, N_steps, skip_check=True)
    fh.run()
    
    V = np.array(fh.V)
    policy = np.array(fh.policy)
    
    return {
        'price': V[s0_idx, 0],
        'V': V,
        'policy': policy,
        'states': states,
        'S_grid': S_grid,
        'v_grid': v_grid,
        'dt': dt,
        'T': T,
        'type': option_type,
        'vol_type': 'stochastic'
    }


# ----------------------------------------------------------------------
# 3. Benchmark Methods
# ----------------------------------------------------------------------
def binomial_american(S0, K, T, r, q, sigma, N, option_type='put'):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    
    ST = S0 * d**np.arange(N, -1, -1) * u**np.arange(0, N+1)
    V = np.maximum(ST - K, 0) if option_type == 'call' else np.maximum(K - ST, 0)
    
    for i in range(N-1, -1, -1):
        V = disc * (p * V[1:i+2] + (1-p) * V[0:i+1])
        S = S0 * d**np.arange(i, -1, -1) * u**np.arange(0, i+1)
        intrinsic = np.maximum(S - K, 0) if option_type == 'call' else np.maximum(K - S, 0)
        V = np.maximum(V, intrinsic)
    return V[0]


def trinomial_american(S0, K, T, r, q, sigma, N, option_type='put'):
    dt = T / N
    dx = sigma * np.sqrt(3 * dt)
    u = np.exp(dx)
    d = np.exp(-dx)
    disc = np.exp(-r * dt)
    mu = r - q
    p_u = ((sigma**2 * dt + (mu * dt)**2) / (2 * dx**2)) + (mu * dt) / (2 * dx)
    p_d = ((sigma**2 * dt + (mu * dt)**2) / (2 * dx**2)) - (mu * dt) / (2 * dx)
    p_m = 1 - p_u - p_d
    
    j_vals = np.arange(-N, N+1)
    ST = S0 * np.exp(j_vals * dx)
    V = np.maximum(ST - K, 0) if option_type == 'call' else np.maximum(K - ST, 0)
    
    for n in range(N-1, -1, -1):
        V_new = np.zeros(2*n + 1)
        for i in range(2*n + 1):
            val = disc * (p_u * V[i+2] + p_m * V[i+1] + p_d * V[i])
            S = S0 * np.exp((i - n) * dx)
            intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
            V_new[i] = max(val, intrinsic)
        V = V_new
    return V[0]


def lsm_american(S0, K, T, r, q, sigma, N_steps=50, N_paths=20000, option_type='put'):
    dt = T / N_steps
    disc = np.exp(-r * dt)
    Z = np.random.randn(N_paths, N_steps)
    S = np.zeros((N_paths, N_steps + 1))
    S[:, 0] = S0
    for t in range(1, N_steps + 1):
        S[:, t] = S[:, t-1] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
    
    payoff = np.maximum(S[:, -1] - K, 0) if option_type == 'call' else np.maximum(K - S[:, -1], 0)
    
    for t in range(N_steps - 1, 0, -1):
        in_money = (S[:, t] > K) if option_type == 'call' else (S[:, t] < K)
        if not np.any(in_money):
            payoff *= disc
            continue
        X = S[in_money, t]
        Y = payoff[in_money] * disc**(N_steps - t)
        reg = np.polyfit(X, Y, deg=2)
        cont_val = np.polyval(reg, X)
        exercise = (X - K) if option_type == 'call' else (K - X)
        payoff[in_money] = np.where(exercise > cont_val, exercise, payoff[in_money] * disc)
        payoff[~in_money] *= disc
    return np.mean(payoff * disc)


# ----------------------------------------------------------------------
# 4. Visualization
# ----------------------------------------------------------------------
def plot_exercise_boundary(result):
    """Plot early-exercise boundary from MDP result."""
    if result['vol_type'] == 'stochastic':
        print("Visualization for stochastic volatility not implemented (3D).")
        return
    
    V = result['V']
    policy = result['policy']
    grid = result['grid']
    T = result['T']
    dt = result['dt']
    opt_type = result['type']
    
    N = policy.shape[1]
    boundary = []
    times = []
    
    for t in range(N):
        exercised = policy[:, t] == 1
        if np.any(exercised):
            if opt_type == 'put':
                idx = np.where(exercised)[0][-1]
            else:
                idx = np.where(exercised)[0][0]
            boundary.append(grid[idx])
            times.append(T - t * dt)
    
    plt.figure(figsize=(8, 5))
    plt.plot(times, boundary, 'o-', label='MDP Exercise Boundary')
    plt.xlabel('Time to Maturity')
    plt.ylabel('Stock Price')
    plt.title(f'Early Exercise Boundary ({opt_type.capitalize()} Option)')
    plt.grid(True)
    plt.legend()
    plt.show()


# ----------------------------------------------------------------------
# 5. Example Usage & Comparison
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Parameters
    S0, K, T = 100, 100, 1.0
    r, q, sigma = 0.05, 0.02, 0.2
    N_steps = 50
    
    # Constant volatility
    mdp_res = american_option_mdp(S0, K, T, r, q, sigma, N_steps, 'put')
    print(f"MDP (const vol): {mdp_res['price']:.4f}")
    
    # Stochastic volatility (Heston-like)
    stoch_res = american_option_mdp_stoch_vol(
        S0, K, T, r, q, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5,
        N_steps=N_steps, option_type='put'
    )
    print(f"MDP (stoch vol): {stoch_res['price']:.4f}")
    
    # Benchmarks
    print(f"Binomial:        {binomial_american(S0, K, T, r, q, sigma, N_steps, 'put'):.4f}")
    print(f"Trinomial:       {trinomial_american(S0, K, T, r, q, sigma, N_steps, 'put'):.4f}")
    print(f"LSM MC:          {lsm_american(S0, K, T, r, q, sigma, N_steps, 20000, 'put'):.4f}")
    
    # Plot boundary
    plot_exercise_boundary(mdp_res)
