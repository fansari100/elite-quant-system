# R Statistical Analysis

Renaissance Technologies-style statistical modeling in R.

## Features

- **Regime Detection** - Hidden Markov Models for market states
- **Dynamic Copulas** - Tail dependency modeling
- **DCC-GARCH** - Time-varying correlations
- **Ledoit-Wolf Shrinkage** - Robust covariance estimation
- **Factor Models** - PCA-based risk decomposition
- **Elastic Net** - Regularized alpha selection
- **Cointegration Testing** - Pairs trading foundation
- **Bootstrap Inference** - Confidence intervals for Sharpe

## Usage

```r
source("statistical_analysis.R")

# Load returns data
returns <- read.csv("returns.csv")

# Detect market regimes
regimes <- detect_market_regimes(returns$SPY, n_states = 3)

# Calculate risk metrics
risk <- calculate_risk_metrics(returns$AAPL)

# Test for cointegration (pairs trading)
coint <- test_cointegration(prices$AAPL, prices$MSFT)
```

## Functions

| Function | Description |
|----------|-------------|
| `detect_market_regimes()` | HMM-based regime detection |
| `fit_dynamic_copula()` | Tail dependency estimation |
| `fit_dcc_garch()` | Dynamic correlation |
| `ledoit_wolf_shrinkage()` | Robust covariance |
| `estimate_factor_model()` | PCA factor decomposition |
| `elastic_net_alpha()` | Regularized factor selection |
| `calculate_risk_metrics()` | VaR, ES, Sharpe, Sortino |
| `test_cointegration()` | Engle-Granger test |
| `bootstrap_sharpe()` | BCa confidence intervals |

## Dependencies

```r
install.packages(c(
  "tidyverse", "zoo", "xts", "quantmod",
  "PerformanceAnalytics", "rmgarch", "rugarch",
  "forecast", "tseries", "fGarch", "copula",
  "MASS", "glmnet", "caret", "jsonlite",
  "depmixS4", "boot", "moments"
))
```

