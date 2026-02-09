"""
Walk-Forward Validation Framework
===================================
Rigorous out-of-sample testing to prevent overfitting.

Based on arXiv:2512.12924 - Interpretable Hypothesis-Driven Trading

Key Features:
- Rolling window validation
- Regime-aware splits
- Realistic transaction costs
- Statistical significance testing
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats


class SplitMethod(Enum):
    """Method for creating train/test splits."""
    ROLLING = "rolling"           # Fixed window that rolls forward
    EXPANDING = "expanding"       # Expanding training window
    ANCHORED = "anchored"         # Fixed start, expanding end
    REGIME_AWARE = "regime_aware" # Split based on market regimes


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    # Window sizes (in trading days)
    train_window: int = 252 * 2     # 2 years training
    test_window: int = 63           # ~3 months testing
    step_size: int = 21             # ~1 month step
    
    # Validation parameters
    min_train_samples: int = 500
    n_splits: int = 10
    gap: int = 5                    # Gap between train and test to avoid lookahead
    
    # Method
    split_method: SplitMethod = SplitMethod.ROLLING
    
    # Regime detection
    use_regime_aware: bool = False
    regime_lookback: int = 60
    
    # Transaction costs for realistic backtest
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0


@dataclass
class FoldResult:
    """Results from a single validation fold."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    
    # Metrics
    train_sharpe: float
    test_sharpe: float
    train_return: float
    test_return: float
    test_volatility: float
    max_drawdown: float
    
    # Predictions
    predictions: Optional[np.ndarray] = None
    actuals: Optional[np.ndarray] = None
    
    def is_overfit(self, threshold: float = 0.5) -> bool:
        """Check if model is likely overfit (train >> test performance)."""
        if self.train_sharpe <= 0:
            return False
        degradation = 1 - (self.test_sharpe / self.train_sharpe)
        return degradation > threshold


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation results."""
    config: WalkForwardConfig
    folds: List[FoldResult]
    
    # Aggregate metrics
    mean_test_sharpe: float = 0.0
    std_test_sharpe: float = 0.0
    mean_test_return: float = 0.0
    hit_rate: float = 0.0  # % of folds with positive Sharpe
    
    # Statistical tests
    t_statistic: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False
    
    def compute_aggregates(self):
        """Compute aggregate statistics."""
        test_sharpes = [f.test_sharpe for f in self.folds]
        test_returns = [f.test_return for f in self.folds]
        
        self.mean_test_sharpe = np.mean(test_sharpes)
        self.std_test_sharpe = np.std(test_sharpes)
        self.mean_test_return = np.mean(test_returns)
        self.hit_rate = np.mean([s > 0 for s in test_sharpes])
        
        # One-sample t-test: is mean Sharpe significantly > 0?
        if len(test_sharpes) > 2:
            self.t_statistic, self.p_value = stats.ttest_1samp(test_sharpes, 0)
            self.is_significant = self.p_value < 0.05 and self.mean_test_sharpe > 0
    
    def summary(self) -> str:
        """Return formatted summary."""
        lines = [
            "=" * 70,
            "WALK-FORWARD VALIDATION RESULTS",
            "=" * 70,
            f"Number of folds:    {len(self.folds)}",
            f"Mean Test Sharpe:   {self.mean_test_sharpe:.3f} ± {self.std_test_sharpe:.3f}",
            f"Mean Test Return:   {self.mean_test_return:.2%}",
            f"Hit Rate:           {self.hit_rate:.1%}",
            f"",
            f"Statistical Significance:",
            f"  t-statistic:      {self.t_statistic:.3f}",
            f"  p-value:          {self.p_value:.4f}",
            f"  Significant:      {'Yes ✓' if self.is_significant else 'No ✗'}",
            f"",
            f"Fold Details:",
        ]
        
        for fold in self.folds:
            overfit_flag = " [OVERFIT]" if fold.is_overfit() else ""
            lines.append(
                f"  Fold {fold.fold_id}: Train Sharpe={fold.train_sharpe:.2f}, "
                f"Test Sharpe={fold.test_sharpe:.2f}, "
                f"Return={fold.test_return:.2%}{overfit_flag}"
            )
        
        lines.append("=" * 70)
        return "\n".join(lines)


class WalkForwardValidator:
    """
    Walk-forward validation framework.
    
    Implements rigorous out-of-sample testing with:
    - Multiple split methods
    - Regime awareness
    - Statistical significance testing
    - Overfitting detection
    """
    
    def __init__(self, config: WalkForwardConfig):
        self.config = config
    
    def create_splits(
        self,
        data: pd.DataFrame,
        date_column: str = 'date'
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create train/test splits based on configuration.
        
        Returns:
            List of (train_indices, test_indices) tuples
        """
        dates = pd.to_datetime(data[date_column])
        n_samples = len(data)
        
        splits = []
        
        if self.config.split_method == SplitMethod.ROLLING:
            splits = self._create_rolling_splits(n_samples)
        elif self.config.split_method == SplitMethod.EXPANDING:
            splits = self._create_expanding_splits(n_samples)
        elif self.config.split_method == SplitMethod.ANCHORED:
            splits = self._create_anchored_splits(n_samples)
        elif self.config.split_method == SplitMethod.REGIME_AWARE:
            splits = self._create_regime_aware_splits(data, dates)
        
        return splits
    
    def _create_rolling_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create rolling window splits."""
        splits = []
        
        train_size = self.config.train_window
        test_size = self.config.test_window
        gap = self.config.gap
        step = self.config.step_size
        
        start = 0
        while start + train_size + gap + test_size <= n_samples:
            train_end = start + train_size
            test_start = train_end + gap
            test_end = test_start + test_size
            
            train_idx = np.arange(start, train_end)
            test_idx = np.arange(test_start, min(test_end, n_samples))
            
            splits.append((train_idx, test_idx))
            
            start += step
            
            if len(splits) >= self.config.n_splits:
                break
        
        return splits
    
    def _create_expanding_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create expanding window splits."""
        splits = []
        
        min_train = self.config.min_train_samples
        test_size = self.config.test_window
        gap = self.config.gap
        step = self.config.step_size
        
        train_end = min_train
        while train_end + gap + test_size <= n_samples:
            test_start = train_end + gap
            test_end = test_start + test_size
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, min(test_end, n_samples))
            
            splits.append((train_idx, test_idx))
            
            train_end += step
            
            if len(splits) >= self.config.n_splits:
                break
        
        return splits
    
    def _create_anchored_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create anchored expanding splits (fixed start)."""
        return self._create_expanding_splits(n_samples)
    
    def _create_regime_aware_splits(
        self,
        data: pd.DataFrame,
        dates: pd.Series
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create splits that respect market regimes."""
        # Compute volatility regime
        if 'returns' in data.columns:
            returns = data['returns'].values
        else:
            returns = np.random.randn(len(data)) * 0.02  # Fallback
        
        # Rolling volatility
        lookback = self.config.regime_lookback
        vol = pd.Series(returns).rolling(lookback).std().values
        
        # High/low volatility regimes
        median_vol = np.nanmedian(vol)
        high_vol = vol > median_vol
        
        # Create splits avoiding regime transitions
        base_splits = self._create_rolling_splits(len(data))
        
        filtered_splits = []
        for train_idx, test_idx in base_splits:
            # Check regime consistency in test period
            test_regimes = high_vol[test_idx]
            if np.nanstd(test_regimes) < 0.3:  # Consistent regime
                filtered_splits.append((train_idx, test_idx))
        
        return filtered_splits if filtered_splits else base_splits
    
    def validate(
        self,
        model_factory: Callable,
        data: pd.DataFrame,
        features_columns: List[str],
        target_column: str,
        date_column: str = 'date'
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.
        
        Args:
            model_factory: Callable that returns a fresh model instance
            data: DataFrame with features and target
            features_columns: List of feature column names
            target_column: Name of target column
            date_column: Name of date column
            
        Returns:
            WalkForwardResult with aggregated metrics
        """
        splits = self.create_splits(data, date_column)
        dates = pd.to_datetime(data[date_column])
        
        folds = []
        
        for fold_id, (train_idx, test_idx) in enumerate(splits):
            # Get data splits
            X_train = data.iloc[train_idx][features_columns].values
            y_train = data.iloc[train_idx][target_column].values
            X_test = data.iloc[test_idx][features_columns].values
            y_test = data.iloc[test_idx][target_column].values
            
            # Create and train model
            model = model_factory()
            
            # Fit model (assumes sklearn-like interface)
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
            
            # Get predictions
            if hasattr(model, 'predict'):
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
            else:
                y_pred_train = y_train
                y_pred_test = y_test
            
            # Compute metrics
            train_metrics = self._compute_metrics(y_train, y_pred_train)
            test_metrics = self._compute_metrics(y_test, y_pred_test)
            
            fold_result = FoldResult(
                fold_id=fold_id,
                train_start=dates.iloc[train_idx[0]],
                train_end=dates.iloc[train_idx[-1]],
                test_start=dates.iloc[test_idx[0]],
                test_end=dates.iloc[test_idx[-1]],
                train_sharpe=train_metrics['sharpe'],
                test_sharpe=test_metrics['sharpe'],
                train_return=train_metrics['total_return'],
                test_return=test_metrics['total_return'],
                test_volatility=test_metrics['volatility'],
                max_drawdown=test_metrics['max_drawdown'],
                predictions=y_pred_test,
                actuals=y_test
            )
            
            folds.append(fold_result)
        
        result = WalkForwardResult(config=self.config, folds=folds)
        result.compute_aggregates()
        
        return result
    
    def _compute_metrics(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray
    ) -> Dict[str, float]:
        """Compute performance metrics."""
        # Treat predictions as signals, compute strategy returns
        strategy_returns = actuals * np.sign(predictions)
        
        # Apply transaction costs
        cost = (self.config.transaction_cost_bps + self.config.slippage_bps) / 10000
        turnover = np.abs(np.diff(np.sign(predictions))).sum() / len(predictions)
        strategy_returns = strategy_returns - cost * turnover
        
        # Metrics
        mean_return = np.mean(strategy_returns)
        std_return = np.std(strategy_returns)
        
        sharpe = mean_return / (std_return + 1e-8) * np.sqrt(252)
        total_return = np.prod(1 + strategy_returns) - 1
        volatility = std_return * np.sqrt(252)
        
        # Max drawdown
        cum_returns = np.cumprod(1 + strategy_returns)
        rolling_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)
        
        return {
            'sharpe': sharpe,
            'total_return': total_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown
        }


class CombinatoricsPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).
    
    From Lopez de Prado's "Advances in Financial Machine Learning"
    
    Generates all possible train/test combinations while:
    - Purging overlapping samples
    - Embargoing to prevent leakage
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        purge_gap: int = 5,
        embargo_pct: float = 0.01
    ):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        groups: np.ndarray = None
    ):
        """Generate train/test splits."""
        from itertools import combinations
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Create fold indices
        fold_size = n_samples // self.n_splits
        folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(self.n_splits)]
        
        # Generate all combinations of test folds
        for test_fold_indices in combinations(range(self.n_splits), self.n_test_splits):
            test_idx = np.concatenate([folds[i] for i in test_fold_indices])
            train_folds = [i for i in range(self.n_splits) if i not in test_fold_indices]
            train_idx = np.concatenate([folds[i] for i in train_folds])
            
            # Apply embargo
            embargo_size = int(self.embargo_pct * n_samples)
            test_min, test_max = test_idx.min(), test_idx.max()
            
            # Remove samples too close to test set
            train_mask = (train_idx < test_min - self.purge_gap - embargo_size) | \
                        (train_idx > test_max + self.purge_gap + embargo_size)
            train_idx = train_idx[train_mask]
            
            yield train_idx, test_idx

