# Julia Numerical Computing Module

## Why Julia?

Julia provides **C++ speed with Python-like syntax** for mathematical computing:

| Benchmark | Python (NumPy) | Julia | C++ |
|-----------|----------------|-------|-----|
| Matrix multiply | 1x | 0.9x | 0.85x |
| Optimization | 1x | 0.3x | 0.25x |
| Monte Carlo | 1x | 0.1x | 0.08x |

## Use Cases in Quant Trading

1. **Portfolio Optimization** - JuMP + OSQP solver
2. **Risk Calculations** - VaR, CVaR, Monte Carlo
3. **Factor Models** - PCA, regression
4. **Numerical Simulations** - Option pricing, stress testing

## Installation

```bash
# Install Julia
curl -fsSL https://install.julialang.org | sh

# Install packages
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Usage

```julia
using QuantOptimization

# Portfolio optimization
returns = [0.12, 0.10, 0.08, 0.15, 0.11]
cov_matrix = # ... your covariance matrix

result = optimize_portfolio(returns, cov_matrix, risk_aversion=2.0)
println("Optimal weights: ", result.weights)
println("Expected return: ", result.expected_return)
println("Sharpe ratio: ", result.sharpe_ratio)
```

## Python Integration

```python
# Call Julia from Python using PyJulia
from julia import Main

Main.include("QuantOptimization.jl")
result = Main.optimize_portfolio(returns, cov_matrix)
```

## Key Functions

- `optimize_portfolio()` - Mean-variance with constraints
- `black_litterman()` - View-based allocation
- `risk_parity_weights()` - Equal risk contribution
- `calculate_efficient_frontier()` - Frontier computation
- `monte_carlo_simulation()` - Path simulation
- `compute_risk_metrics()` - VaR, CVaR, Sharpe, etc.

