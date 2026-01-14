# Analytic-model-for-pricing-American-options-using-Markov-Decision-Processes

American options involve an optimal stopping problem: at each time step, the holder chooses between exercising (and receiving intrinsic value) or continuing (and holding the option). This fits perfectly into a finite-horizon Markov Decision Process (MDP) framework. Below is a complete Python module american_option_mdp that:
- uses MDPtoolbox correctly via FiniteHorizon;
- prices American options under constant and stochastic volatility;
- computes Greeks from the MDP value function;
- supports calibration to market prices;
- includes visualizations: exercise boundary, value surface, policy map;
- saves results to .csv and plots to .png;
- comes with unit tests against binomial benchmarks.

Additionally below is a complete Python module american_option_pricer.py that:
- packages our MDP-based American option pricer (with dividends, calls/puts, and exercise boundary visualization);
- adds support for stochastic volatility via a discretized Heston-like model (using a 2D state space: stock price × volatility);
- includes comparison utilities against binomial, trinomial, and LSM methods.

I suppose we're exploring sits at a intersection of quantitative finance, stochastic control, and computational decision theory. Let me clarify the practical relevance and best-use scenarios for the scripts american_option_mdp  and american_option_pricer.py.

## 1. Analyzing American options

These scripts are specialized tools for pricing and analyzing American-style options, which differ from European options because they can be exercised at any time before expiration. This early-exercise feature makes them path-dependent and non-trivial to price.

Primary Use Cases:

a) Derivatives pricing: accurate valuation of American puts/calls on equities, indices, or commodities—especially when dividends or stochastic volatility matter.

b) Risk management: computing Greeks (e.g., delta, gamma) via perturbation of MDP value functions; stress-testing early-exercise behavior under volatile regimes.

c) Hedging: extracting optimal exercise policies to inform automated execution or dynamic hedging strategies.

d) Structured products: pricing exotic payoffs with embedded American features (e.g., callable convertibles, auto-callables).

e) Academic research and teaching: demonstrating how optimal stopping = MDP with 2 actions, linking reinforcement learning and classical finance.
May be the MDP formulation is exact for discretized state spaces and provides both price and policy—not just a number, but a decision rule (“exercise if S < X(t)”).

## 2. When should prefer these scripts over standard methods
a) Low-dimensional problems (1–2 state variables: e.g., stock + vol) - MDP (FiniteHorizon), highly accurate, gives full policy.
b) High-dimensional problems (multi-asset, path-dependent) - use DQN version - scales better than grid-based MDP.
c) Benchmarking / Validation - compare MDP vs. binomial vs. LSM - builds confidence in results.
d) Real-time pricing in production - avoid MDPtoolbox (slow for large grids); prefer analytical approximations or precomputed lookup tables.
e) Stochastic volatility models (e.g., Heston-like) - 2D MDP or DQN  traditional trees struggle here.
I suppose MDPtoolbox uses explicit state discretization - suffers from the curse of dimensionality. For >2 state variables, we should switch to deep RL or Monte Carlo with regression (LSM).

## 3. Recommendations for Analysis
a) Start begin with constant volatility, no dividends, and compare:
- MDP result ↔ Binomial tree ↔ Black-Scholes (for European reference);
- verify convergence as you increase N_states and N_steps.
b) Add Realism Gradually:
- introduce dividends (q > 0) → observe earlier exercise for calls;
- switch to stochastic volatility → see how uncertainty in v delays exercise (volatility smile effect);
- plot the early-exercise boundary — it should tilt upward for puts as maturity approaches.
c) Validate Against Market Data:
- calibrate sigma (or Heston parameters) to market prices of American options (e.g., on ETFs like SPY);
- use our MDP pricer to infer implied early-exercise premiums.
d) Use Policy Output for Strategy Design:
- the policy array tells you exactly when to exercise;
- feed this into a backtester: simulate paths, apply the policy, compute P&L;
- compare against naive strategies (e.g., “exercise at-the-money”).
e) Hybrid Workflow:
- use MDP for offline policy learning;
- use DQN or LSM for online/inference in high-dim settings;
- use binomial/trinomial for quick sanity checks.

I think traditional finance often treats option pricing as a pure valuation problem. But American options are fundamentally decision problems -and that’s where MDPs shine. These scripts don’t just give a price; they reveal how a rational agent should behave under uncertainty.
May be this perspective is increasingly valuable in real options analysis in corporate finance.

## 4. Possible Questions:
- pricing & policy extraction for American options under dividends or stochastic volatility;
- use MDP for accuracy in 1–2D; use DQN/LSM for scalability;
- output to leverage - the exercise policy, not just the price;
- avoid in high-frequency or high-dimensional production systems without approximation.

## 5. How to Run american_option_mdp

5.1. Install Dependencies
```
# Install core dependencies
pip install numpy scipy matplotlib pandas

# Install MDPtoolbox (Python version)
pip install pymdptoolbox
```
The official package is often called pymdptoolbox on PyPI. If that fails, install from source:
```
git clone https://github.com/sawcordwell/pymdptoolbox.git
cd pymdptoolbox
pip install -e .
```

5.2. Install our Package
From the root directory (where setup.py is):
```
pip install -e .
```
The -e flag installs in editable mode, so changes to your code take effect immediately.

5.3. Run a Simple Example
```
python example.py
```

5.4. Run Unit Tests
From the root directory:
```
pip install pytest
pytest tests/ -v
```
5.5. Use Stochastic Volatility
Modify example.py to include stochastic volatility:
```
from american_option_mdp import american_option_mdp_stoch_vol

res_sv = american_option_mdp_stoch_vol(
    S0=100, K=100, T=1.0, r=0.05, q=0.02,
    v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5,
    N_steps=30, option_type="put"
)
print(f"Stochastic vol price: ${res_sv['price']:.4f}")
```

## 6. How to Run american_option_pricer.py:

6.1. Install dependencies:
```
pip install numpy matplotlib scipy pandas
```
6.2. Run: 
‘’’
python3 american_option_pricer.py
```

## 7. Outputs:
a) american_option_mdp:
- plot_exercise_boundary.png;
- plot_value_function.png;
- save_results.csv;
b) american_option_pricer.py:
- american_option_results_YYYYMMDD_HHMMSS.csv, contains: calibrated σ, model price, Greeks, calibration error;
- exercise_boundary_YYYYMMDD_HHMMSS.png, high-resolution plot of the early-exercise boundary.
