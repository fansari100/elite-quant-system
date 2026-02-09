#=
Elite Quant System - Julia Module
==================================
High-performance numerical computing for portfolio optimization.

Julia provides:
- C++ speed with Python-like syntax
- Native parallel computing
- Excellent for mathematical optimization
- JuMP for algebraic modeling

Used by: Hedge funds for research-intensive trading models
=#

module QuantOptimization

using LinearAlgebra
using Statistics
using Distributions
using Optim
using JuMP
using OSQP  # Quadratic programming solver

export optimize_portfolio, 
       calculate_efficient_frontier,
       compute_risk_metrics,
       mean_variance_optimize,
       black_litterman,
       risk_parity_weights

"""
    PortfolioResult

Result structure for portfolio optimization.
"""
struct PortfolioResult
    weights::Vector{Float64}
    expected_return::Float64
    volatility::Float64
    sharpe_ratio::Float64
    optimization_status::Symbol
end

"""
    optimize_portfolio(returns, cov_matrix; kwargs...)

Mean-variance portfolio optimization using JuMP.

# Arguments
- `returns::Vector{Float64}`: Expected returns for each asset
- `cov_matrix::Matrix{Float64}`: Covariance matrix
- `risk_aversion::Float64=1.0`: Risk aversion parameter (λ)
- `max_weight::Float64=0.1`: Maximum weight per asset
- `min_weight::Float64=0.0`: Minimum weight per asset

# Returns
- `PortfolioResult`: Optimal weights and metrics
"""
function optimize_portfolio(
    returns::Vector{Float64},
    cov_matrix::Matrix{Float64};
    risk_aversion::Float64 = 1.0,
    max_weight::Float64 = 0.1,
    min_weight::Float64 = 0.0
)::PortfolioResult
    
    n_assets = length(returns)
    
    # Create optimization model
    model = Model(OSQP.Optimizer)
    set_silent(model)
    
    # Decision variables: portfolio weights
    @variable(model, min_weight <= w[1:n_assets] <= max_weight)
    
    # Constraint: weights sum to 1
    @constraint(model, sum(w) == 1.0)
    
    # Objective: maximize return - (λ/2) * variance
    # Quadratic form for variance: w' * Σ * w
    @objective(model, Max, 
        dot(returns, w) - (risk_aversion / 2) * dot(w, cov_matrix * w)
    )
    
    # Solve
    optimize!(model)
    
    # Extract results
    if termination_status(model) == MOI.OPTIMAL
        weights = value.(w)
        exp_ret = dot(returns, weights)
        vol = sqrt(dot(weights, cov_matrix * weights))
        sharpe = exp_ret / vol
        
        return PortfolioResult(weights, exp_ret, vol, sharpe, :Optimal)
    else
        return PortfolioResult(
            fill(1.0/n_assets, n_assets), 
            0.0, 0.0, 0.0, 
            :Failed
        )
    end
end

"""
    mean_variance_optimize(returns_matrix; target_return=nothing)

Classic Markowitz mean-variance optimization.
"""
function mean_variance_optimize(
    returns_matrix::Matrix{Float64};
    target_return::Union{Float64, Nothing} = nothing,
    risk_free_rate::Float64 = 0.0
)::PortfolioResult
    
    n_assets = size(returns_matrix, 2)
    
    # Calculate expected returns and covariance
    μ = vec(mean(returns_matrix, dims=1))
    Σ = cov(returns_matrix)
    
    # If no target, optimize Sharpe ratio
    if isnothing(target_return)
        # Maximum Sharpe Ratio portfolio
        ones_vec = ones(n_assets)
        Σ_inv = inv(Σ)
        excess_returns = μ .- risk_free_rate
        
        weights = Σ_inv * excess_returns
        weights = weights / sum(weights)  # Normalize
        
        exp_ret = dot(μ, weights)
        vol = sqrt(dot(weights, Σ * weights))
        sharpe = (exp_ret - risk_free_rate) / vol
        
        return PortfolioResult(weights, exp_ret, vol, sharpe, :Optimal)
    else
        return optimize_portfolio(μ, Σ, risk_aversion=1.0)
    end
end

"""
    calculate_efficient_frontier(returns, cov_matrix; n_points=50)

Compute the efficient frontier.
"""
function calculate_efficient_frontier(
    returns::Vector{Float64},
    cov_matrix::Matrix{Float64};
    n_points::Int = 50
)::Tuple{Vector{Float64}, Vector{Float64}, Matrix{Float64}}
    
    n_assets = length(returns)
    
    # Find min and max achievable returns
    min_ret = minimum(returns)
    max_ret = maximum(returns)
    
    target_returns = range(min_ret, max_ret, length=n_points)
    frontier_vols = zeros(n_points)
    frontier_weights = zeros(n_points, n_assets)
    
    for (i, target) in enumerate(target_returns)
        # Minimize variance for each target return
        model = Model(OSQP.Optimizer)
        set_silent(model)
        
        @variable(model, 0 <= w[1:n_assets] <= 1)
        @constraint(model, sum(w) == 1.0)
        @constraint(model, dot(returns, w) >= target)
        
        @objective(model, Min, dot(w, cov_matrix * w))
        
        optimize!(model)
        
        if termination_status(model) == MOI.OPTIMAL
            weights = value.(w)
            frontier_vols[i] = sqrt(dot(weights, cov_matrix * weights))
            frontier_weights[i, :] = weights
        end
    end
    
    return (collect(target_returns), frontier_vols, frontier_weights)
end

"""
    black_litterman(returns, cov_matrix, P, Q; tau=0.05)

Black-Litterman model for incorporating views.

# Arguments
- `returns`: Equilibrium returns (from market cap weights)
- `cov_matrix`: Covariance matrix
- `P`: Pick matrix (views on assets)
- `Q`: View returns
- `tau`: Uncertainty in prior (typically 0.025-0.05)
- `omega`: Uncertainty in views (if nothing, uses proportional to P*Σ*P')
"""
function black_litterman(
    returns::Vector{Float64},
    cov_matrix::Matrix{Float64},
    P::Matrix{Float64},
    Q::Vector{Float64};
    tau::Float64 = 0.05,
    omega::Union{Matrix{Float64}, Nothing} = nothing
)::Vector{Float64}
    
    # Prior covariance
    Σ = cov_matrix
    π = returns  # Equilibrium returns
    
    # View uncertainty (if not provided)
    if isnothing(omega)
        Ω = tau * P * Σ * P'
    else
        Ω = omega
    end
    
    # Black-Litterman formula
    τΣ = tau * Σ
    
    # Posterior expected returns
    M = inv(inv(τΣ) + P' * inv(Ω) * P)
    posterior_returns = M * (inv(τΣ) * π + P' * inv(Ω) * Q)
    
    return posterior_returns
end

"""
    risk_parity_weights(cov_matrix; target_risk=nothing)

Risk parity portfolio allocation.
Each asset contributes equally to portfolio risk.
"""
function risk_parity_weights(
    cov_matrix::Matrix{Float64};
    target_risk::Union{Float64, Nothing} = nothing
)::Vector{Float64}
    
    n_assets = size(cov_matrix, 1)
    
    # Initial guess: equal weights
    w0 = fill(1.0/n_assets, n_assets)
    
    # Objective: minimize sum of squared risk contribution differences
    function objective(w)
        w = abs.(w)  # Ensure positive
        w = w / sum(w)  # Normalize
        
        # Portfolio volatility
        σ_p = sqrt(dot(w, cov_matrix * w))
        
        # Marginal risk contributions
        mrc = cov_matrix * w / σ_p
        
        # Risk contributions
        rc = w .* mrc
        
        # Target: equal risk contribution
        target_rc = σ_p / n_assets
        
        return sum((rc .- target_rc).^2)
    end
    
    # Optimize
    result = Optim.optimize(objective, w0, LBFGS())
    
    # Extract and normalize weights
    weights = abs.(Optim.minimizer(result))
    weights = weights / sum(weights)
    
    return weights
end

"""
    compute_risk_metrics(returns; confidence=0.95)

Compute comprehensive risk metrics.
"""
function compute_risk_metrics(
    returns::Vector{Float64};
    confidence::Float64 = 0.95,
    risk_free_rate::Float64 = 0.0
)::Dict{Symbol, Float64}
    
    n = length(returns)
    
    # Basic statistics
    μ = mean(returns)
    σ = std(returns)
    
    # Sharpe ratio (annualized)
    sharpe = (μ - risk_free_rate/252) / σ * sqrt(252)
    
    # Sortino ratio (downside deviation)
    negative_returns = returns[returns .< 0]
    downside_std = length(negative_returns) > 0 ? std(negative_returns) : σ
    sortino = (μ - risk_free_rate/252) / downside_std * sqrt(252)
    
    # VaR (Historical)
    sorted_returns = sort(returns)
    var_idx = Int(floor((1 - confidence) * n))
    var = sorted_returns[max(1, var_idx)]
    
    # CVaR (Expected Shortfall)
    cvar = mean(sorted_returns[1:var_idx])
    
    # Maximum Drawdown
    cumulative = cumprod(1 .+ returns)
    running_max = accumulate(max, cumulative)
    drawdowns = (cumulative .- running_max) ./ running_max
    max_dd = minimum(drawdowns)
    
    # Calmar ratio
    ann_return = (cumulative[end])^(252/n) - 1
    calmar = -ann_return / max_dd
    
    return Dict(
        :mean_return => μ * 252,  # Annualized
        :volatility => σ * sqrt(252),  # Annualized
        :sharpe_ratio => sharpe,
        :sortino_ratio => sortino,
        :var_95 => var,
        :cvar_95 => cvar,
        :max_drawdown => max_dd,
        :calmar_ratio => calmar,
        :skewness => skewness(returns),
        :kurtosis => kurtosis(returns)
    )
end

"""
    monte_carlo_simulation(returns, cov_matrix, n_sims, n_days)

Monte Carlo simulation for portfolio returns.
"""
function monte_carlo_simulation(
    weights::Vector{Float64},
    returns::Vector{Float64},
    cov_matrix::Matrix{Float64};
    n_simulations::Int = 10000,
    n_days::Int = 252
)::Matrix{Float64}
    
    n_assets = length(weights)
    
    # Cholesky decomposition for correlated random numbers
    L = cholesky(cov_matrix).L
    
    # Generate simulations
    simulated_values = zeros(n_simulations, n_days)
    
    for sim in 1:n_simulations
        portfolio_value = 1.0
        
        for day in 1:n_days
            # Generate correlated random returns
            z = randn(n_assets)
            asset_returns = returns + L * z
            
            # Portfolio return
            port_return = dot(weights, asset_returns)
            portfolio_value *= (1 + port_return)
            
            simulated_values[sim, day] = portfolio_value
        end
    end
    
    return simulated_values
end

# Precompile key functions
precompile(optimize_portfolio, (Vector{Float64}, Matrix{Float64}))
precompile(mean_variance_optimize, (Matrix{Float64},))
precompile(risk_parity_weights, (Matrix{Float64},))

end  # module

