"""
Professional Backtesting Engine
================================
Vectorized backtesting with realistic transaction costs and slippage.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    returns: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, float]
    equity_curve: pd.Series
    drawdowns: pd.Series
    
    def summary(self) -> str:
        """Return formatted summary."""
        lines = [
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            f"Total Return:      {self.metrics['total_return']:.2%}",
            f"Annualized Return: {self.metrics['annual_return']:.2%}",
            f"Volatility:        {self.metrics['volatility']:.2%}",
            f"Sharpe Ratio:      {self.metrics['sharpe_ratio']:.2f}",
            f"Sortino Ratio:     {self.metrics['sortino_ratio']:.2f}",
            f"Max Drawdown:      {self.metrics['max_drawdown']:.2%}",
            f"Calmar Ratio:      {self.metrics['calmar_ratio']:.2f}",
            f"Win Rate:          {self.metrics['win_rate']:.2%}",
            f"Avg Win/Loss:      {self.metrics['avg_win_loss']:.2f}",
            f"Turnover (ann.):   {self.metrics['annual_turnover']:.2%}",
            f"Transaction Costs: {self.metrics['total_costs']:.2%}",
            "=" * 60
        ]
        return "\n".join(lines)


class BacktestEngine:
    """
    Professional-grade backtesting engine.
    """
    
    def __init__(
        self,
        transaction_cost_bps: float = 5.0,
        slippage_bps: float = 2.0,
        risk_free_rate: float = 0.05,
        rebalance_frequency: str = "daily"
    ):
        self.transaction_cost = transaction_cost_bps / 10000
        self.slippage = slippage_bps / 10000
        self.risk_free_rate = risk_free_rate
        self.rebalance_frequency = rebalance_frequency
    
    def run(
        self,
        model: torch.nn.Module,
        test_loader: Any,
        tickers: List[str],
        dates: Optional[List[datetime]] = None
    ) -> BacktestResult:
        """
        Run backtest on test data.
        """
        model.eval()
        device = next(model.parameters()).device
        
        all_weights = []
        all_returns = []
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(device)
                target_returns = batch['target_returns'].to(device)
                
                # Get model predictions
                output = model(features)
                weights = output['weights'].cpu().numpy()
                returns = target_returns.cpu().numpy()
                
                all_weights.extend(weights)
                all_returns.extend(returns)
        
        all_weights = np.array(all_weights)
        all_returns = np.array(all_returns)
        
        # Create date index
        if dates is None:
            dates = pd.date_range('2024-01-01', periods=len(all_weights), freq='B')
        
        # Create DataFrames
        weights_df = pd.DataFrame(all_weights, index=dates[:len(all_weights)], columns=tickers)
        returns_df = pd.DataFrame(all_returns, index=dates[:len(all_returns)], columns=tickers)
        
        # Run simulation
        result = self._simulate(weights_df, returns_df)
        
        return result
    
    def run_from_arrays(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        tickers: List[str],
        dates: Optional[pd.DatetimeIndex] = None
    ) -> BacktestResult:
        """Run backtest from numpy arrays."""
        if dates is None:
            dates = pd.date_range('2024-01-01', periods=len(weights), freq='B')
        
        weights_df = pd.DataFrame(weights, index=dates, columns=tickers)
        returns_df = pd.DataFrame(returns, index=dates, columns=tickers)
        
        return self._simulate(weights_df, returns_df)
    
    def _simulate(
        self,
        weights_df: pd.DataFrame,
        returns_df: pd.DataFrame
    ) -> BacktestResult:
        """Run vectorized backtest simulation."""
        n_days = len(weights_df)
        tickers = weights_df.columns.tolist()
        
        # Initialize
        equity = [1.0]
        portfolio_returns = []
        trades_list = []
        total_costs = 0.0
        
        prev_weights = np.zeros(len(tickers))
        
        for i in range(n_days):
            current_weights = weights_df.iloc[i].values
            day_returns = returns_df.iloc[i].values
            
            # Calculate trades and costs
            trades = current_weights - prev_weights
            trade_volume = np.abs(trades).sum()
            day_cost = trade_volume * (self.transaction_cost + self.slippage)
            total_costs += day_cost
            
            # Portfolio return
            gross_return = (current_weights * day_returns).sum()
            net_return = gross_return - day_cost
            
            # Update equity
            new_equity = equity[-1] * (1 + net_return)
            equity.append(new_equity)
            portfolio_returns.append(net_return)
            
            # Record trades
            trades_list.append({
                'date': weights_df.index[i],
                'turnover': trade_volume,
                'cost': day_cost,
                'gross_return': gross_return,
                'net_return': net_return
            })
            
            prev_weights = current_weights
        
        # Create result series
        returns_series = pd.Series(portfolio_returns, index=weights_df.index)
        equity_series = pd.Series(equity[1:], index=weights_df.index)
        trades_df = pd.DataFrame(trades_list)
        
        # Compute drawdowns
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        
        # Compute metrics
        metrics = self._compute_metrics(returns_series, equity_series, drawdowns, total_costs)
        
        return BacktestResult(
            returns=returns_series,
            weights=weights_df,
            trades=trades_df,
            metrics=metrics,
            equity_curve=equity_series,
            drawdowns=drawdowns
        )
    
    def _compute_metrics(
        self,
        returns: pd.Series,
        equity: pd.Series,
        drawdowns: pd.Series,
        total_costs: float
    ) -> Dict[str, float]:
        """Compute performance metrics."""
        n_days = len(returns)
        annual_factor = 252
        
        # Basic metrics
        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Annualized
        annual_return = (1 + total_return) ** (annual_factor / n_days) - 1
        annual_vol = std_return * np.sqrt(annual_factor)
        
        # Sharpe ratio
        excess_return = annual_return - self.risk_free_rate
        sharpe = excess_return / annual_vol if annual_vol > 0 else 0
        
        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() * np.sqrt(annual_factor) if len(negative_returns) > 0 else 1e-10
        sortino = excess_return / downside_std
        
        # Drawdown metrics
        max_dd = drawdowns.min()
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        # Win rate
        winning_days = (returns > 0).sum()
        win_rate = winning_days / n_days
        
        # Avg win / avg loss
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1
        avg_win_loss = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Turnover
        annual_turnover = n_days / annual_factor  # Simplified
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'avg_win_loss': avg_win_loss,
            'annual_turnover': annual_turnover,
            'total_costs': total_costs,
            'n_days': n_days
        }


class BacktestVisualizer:
    """Visualization utilities for backtest results."""
    
    @staticmethod
    def plot_equity_curve(result: BacktestResult, benchmark: Optional[pd.Series] = None):
        """Plot equity curve with drawdowns."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Equity curve
        axes[0].plot(result.equity_curve, label='Strategy', linewidth=2, color='#2ecc71')
        if benchmark is not None:
            axes[0].plot(benchmark, label='Benchmark', linewidth=2, color='#3498db', alpha=0.7)
        axes[0].set_ylabel('Equity')
        axes[0].set_title('Portfolio Equity Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Drawdowns
        axes[1].fill_between(result.drawdowns.index, result.drawdowns * 100, 0, 
                            color='#e74c3c', alpha=0.5)
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].set_title('Drawdowns')
        axes[1].grid(True, alpha=0.3)
        
        # Rolling Sharpe
        rolling_sharpe = result.returns.rolling(63).mean() / result.returns.rolling(63).std() * np.sqrt(252)
        axes[2].plot(rolling_sharpe, color='#9b59b6', linewidth=1.5)
        axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[2].set_ylabel('Rolling Sharpe (63d)')
        axes[2].set_title('Rolling Sharpe Ratio')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_weights(result: BacktestResult, top_n: int = 10):
        """Plot portfolio weight evolution."""
        # Get top holdings by average weight
        avg_weights = result.weights.mean().sort_values(ascending=False)
        top_tickers = avg_weights.head(top_n).index.tolist()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        result.weights[top_tickers].plot.area(ax=ax, alpha=0.7)
        ax.set_ylabel('Portfolio Weight')
        ax.set_title(f'Top {top_n} Holdings Over Time')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_monthly_returns(result: BacktestResult):
        """Plot monthly returns heatmap."""
        monthly = result.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Reshape for heatmap
        monthly_df = pd.DataFrame({
            'year': monthly.index.year,
            'month': monthly.index.month,
            'return': monthly.values * 100
        })
        
        pivot = monthly_df.pivot(index='year', columns='month', values='return')
        
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(
            pivot, 
            annot=True, 
            fmt='.1f', 
            cmap='RdYlGn',
            center=0,
            ax=ax
        )
        ax.set_title('Monthly Returns (%)')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        plt.tight_layout()
        return fig

