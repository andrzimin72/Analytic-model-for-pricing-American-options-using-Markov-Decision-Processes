import numpy as np
from scipy.sparse import lil_matrix
from mdptoolbox.mdp import FiniteHorizon
from typing import Dict, Literal

def american_option_mdp(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    N_steps: int,
    option_type: Literal["put", "call"] = "put",
    S_min_factor: float = 0.2,
    S_max_factor: float = 3.0,
    N_states: int = 200,
) -> Dict:
    if option_type not in ("put", "call"):
        raise ValueError("option_type must be 'put' or 'call'")
    
    S_min = S0 * S_min_factor
    S_max = S0 * S_max_factor
    price_grid = np.linspace(S_min, S_max, N_states)
    s0_idx = np.argmin(np.abs(price_grid - S0))
    
    dt = T / N_steps
    discount = np.exp(-r * dt)
    mu = r - q
    
    dx = sigma * np.sqrt(3 * dt)
    u = np.exp(dx)
    d = np.exp(-dx)
    p_u = ((sigma**2 * dt + (mu * dt)**2) / (2 * dx**2)) + (mu * dt) / (2 * dx)
    p_d = ((sigma**2 * dt + (mu * dt)**2) / (2 * dx**2)) - (mu * dt) / (2 * dx)
    p_m = 1 - p_u - p_d
    
    if not (0 <= p_u <= 1 and 0 <= p_m <= 1 and 0 <= p_d <= 1):
        raise ValueError("Invalid trinomial probabilities — reduce dt.")
    
    A = 2
    P = [lil_matrix((N_states, N_states), dtype=np.float32) for _ in range(A)]
    
    for s in range(N_states):
        S = price_grid[s]
        Su, Sm, Sd = S * u, S, S * d
        iu = np.argmin(np.abs(price_grid - Su))
        im = np.argmin(np.abs(price_grid - Sm))
        id_ = np.argmin(np.abs(price_grid - Sd))
        P[0][s, iu] += p_u
        P[0][s, im] += p_m
        P[0][s, id_] += p_d
        if s == 0 or s == N_states - 1:
            P[0][s, :] = 0
            P[0][s, s] = 1.0
    P[1].setdiag(1.0)
    P = [p.tocsr() for p in P]
    
    R = [np.zeros(N_states, dtype=np.float32), np.zeros(N_states, dtype=np.float32)]
    if option_type == "call":
        R[1][:] = np.maximum(price_grid - K, 0.0)
    else:
        R[1][:] = np.maximum(K - price_grid, 0.0)
    
    fh = FiniteHorizon(P, R, discount, N_steps, skip_check=True)
    fh.run()
    
    return {
        "price": float(fh.V[s0_idx, 0]),
        "V": fh.V,
        "policy": fh.policy,
        "grid": price_grid,
        "dt": dt,
        "T": T,
        "type": option_type,
    }

def american_option_mdp_stoch_vol(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    N_steps: int,
    option_type: Literal["put", "call"] = "put",
    S_min_factor: float = 0.2,
    S_max_factor: float = 3.0,
    N_S: int = 50,
    v_min: float = 0.01,
    v_max: float = 0.5,
    N_v: int = 20,
) -> Dict:
    S_grid = np.linspace(S0 * S_min_factor, S0 * S_max_factor, N_S)
    v_grid = np.linspace(v_min, v_max, N_v)
    SS, VV = np.meshgrid(S_grid, v_grid, indexing='ij')
    states = np.stack([SS.ravel(), VV.ravel()], axis=1)
    N_states = len(states)
    s0_idx = np.argmin(np.linalg.norm(states - [S0, v0], axis=1))
    
    dt = T / N_steps
    discount = np.exp(-r * dt)
    A = 2
    P = [lil_matrix((N_states, N_states), dtype=np.float32) for _ in range(A)]
    
    for i, (S, v) in enumerate(states):
        if v <= 0: v = 1e-6
        mu_S = np.log(S) + (r - q - 0.5 * v) * dt
        sigma_S = np.sqrt(v * dt)
        mu_v = v + kappa * (theta - v) * dt
        sigma_v = xi * np.sqrt(v * dt)
        
        dS_vals = [mu_S - sigma_S, mu_S, mu_S + sigma_S]
        dv_vals = [mu_v - sigma_v, mu_v, mu_v + sigma_v]
        probs = np.full((3, 3), 1/9)
        
        for di in range(3):
            for dj in range(3):
                S_next = np.exp(dS_vals[di])
                v_next = np.clip(dv_vals[dj], v_min, v_max)
                j = np.argmin(np.linalg.norm(states - [S_next, v_next], axis=1))
                P[0][i, j] += probs[di, dj]
        row_sum = P[0][i, :].sum()
        if row_sum > 0:
            P[0][i, :] /= row_sum
        else:
            P[0][i, i] = 1.0
    P[1].setdiag(1.0)
    P = [p.tocsr() for p in P]
    
    R = [np.zeros(N_states, dtype=np.float32), np.zeros(N_states, dtype=np.float32)]
    payoff = np.maximum(states[:, 0] - K, 0.0) if option_type == "call" else np.maximum(K - states[:, 0], 0.0)
    R[1][:] = payoff
    
    fh = FiniteHorizon(P, R, discount, N_steps, skip_check=True)
    fh.run()
    
    return {
        "price": float(fh.V[s0_idx, 0]),
        "V": fh.V,
        "policy": fh.policy,
        "states": states,
        "S_grid": S_grid,
        "v_grid": v_grid,
        "type": option_type,
    }