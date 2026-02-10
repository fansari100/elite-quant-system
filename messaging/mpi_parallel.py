"""
MPI-based parallel computing for large-scale Monte Carlo simulations
and distributed risk calculations.

Uses mpi4py for inter-process communication across cluster nodes.
Run with: mpiexec -n 8 python -m messaging.mpi_parallel
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def parallel_monte_carlo(
    n_paths: int,
    n_steps: int,
    S0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
) -> Optional[dict]:
    """
    Distributed Monte Carlo simulation of GBM using MPI.

    Each MPI rank simulates n_paths/n_ranks paths, then results
    are gathered at rank 0 for aggregation.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_paths = n_paths // size
    dt = T / n_steps
    rng = np.random.default_rng(seed=42 + rank)

    # Each rank simulates its share of paths
    S = np.zeros((local_paths, n_steps + 1))
    S[:, 0] = S0

    for i in range(n_steps):
        dW = rng.normal(0, np.sqrt(dt), local_paths)
        S[:, i + 1] = S[:, i] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)

    local_terminal = S[:, -1]
    local_mean = np.mean(local_terminal)
    local_var = np.var(local_terminal)

    # Gather results at rank 0
    all_means = comm.gather(local_mean, root=0)
    all_vars = comm.gather(local_var, root=0)

    if rank == 0:
        global_mean = np.mean(all_means)
        global_std = np.sqrt(np.mean(all_vars))
        return {
            "mean": float(global_mean),
            "std": float(global_std),
            "n_paths": n_paths,
            "n_ranks": size,
            "paths_per_rank": local_paths,
        }
    return None


def parallel_var_calculation(
    portfolio_returns: np.ndarray,
    n_simulations: int = 100_000,
    confidence: float = 0.99,
) -> Optional[dict]:
    """
    Distributed Value-at-Risk calculation using MPI.

    Parallelizes bootstrap resampling across MPI ranks.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Broadcast portfolio returns to all ranks
    portfolio_returns = comm.bcast(portfolio_returns, root=0)

    local_sims = n_simulations // size
    rng = np.random.default_rng(seed=123 + rank)

    # Bootstrap resampling
    local_losses = np.zeros(local_sims)
    n_days = len(portfolio_returns)

    for i in range(local_sims):
        sample = rng.choice(portfolio_returns, size=n_days, replace=True)
        local_losses[i] = np.sum(sample)

    # Gather all losses at rank 0
    all_losses = comm.gather(local_losses, root=0)

    if rank == 0:
        combined = np.concatenate(all_losses)
        var = float(np.percentile(combined, (1 - confidence) * 100))
        cvar = float(np.mean(combined[combined <= var]))
        return {
            "VaR": var,
            "CVaR": cvar,
            "confidence": confidence,
            "n_simulations": n_simulations,
        }
    return None


if __name__ == "__main__":
    result = parallel_monte_carlo(n_paths=1_000_000, n_steps=252)
    if result:
        print(f"Monte Carlo Results (MPI): {result}")
