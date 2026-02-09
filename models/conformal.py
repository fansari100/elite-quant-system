"""
Conformal Prediction for Uncertainty Quantification
=====================================================
Provides distribution-free prediction intervals for portfolio returns.

Based on:
- Conformalized Quantile Regression (CQR)
- Adaptive Conformal Inference (ACI)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ConformalInterval:
    """Prediction interval with conformal guarantee."""
    lower: np.ndarray
    upper: np.ndarray
    coverage: float  # Target coverage (e.g., 0.95)
    width: np.ndarray  # Interval width
    
    @property
    def mean_width(self) -> float:
        return self.width.mean()


class ConformizedQuantileRegression(nn.Module):
    """
    Conformalized Quantile Regression (CQR).
    
    Combines quantile regression with conformal prediction
    for valid prediction intervals regardless of data distribution.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        quantiles: Tuple[float, ...] = (0.025, 0.5, 0.975),
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Quantile heads
        self.quantile_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in quantiles
        ])
        
        # Calibration scores (computed on calibration set)
        self.register_buffer('calibration_scores', torch.zeros(1000))
        self.n_calibration = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) features
            
        Returns:
            quantiles: (batch, n_quantiles) predicted quantiles
        """
        features = self.backbone(x)
        
        quantile_preds = []
        for head in self.quantile_heads:
            quantile_preds.append(head(features))
        
        quantiles = torch.cat(quantile_preds, dim=-1)
        
        # Ensure quantile ordering (monotonicity)
        quantiles = torch.sort(quantiles, dim=-1)[0]
        
        return quantiles
    
    def quantile_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Pinball loss for quantile regression.
        """
        total_loss = 0.0
        
        for i, q in enumerate(self.quantiles):
            pred = predictions[:, i]
            error = targets - pred
            loss = torch.max(q * error, (q - 1) * error)
            total_loss += loss.mean()
        
        return total_loss / len(self.quantiles)
    
    def calibrate(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.1
    ):
        """
        Calibrate the model on held-out calibration set.
        
        Args:
            predictions: (n_cal, n_quantiles) predicted quantiles
            targets: (n_cal,) true values
            alpha: Miscoverage rate (1 - alpha = coverage)
        """
        # Compute conformity scores
        lower_idx = 0  # First quantile (lower bound)
        upper_idx = -1  # Last quantile (upper bound)
        
        lower_preds = predictions[:, lower_idx]
        upper_preds = predictions[:, upper_idx]
        
        # Non-conformity score: max(lower - y, y - upper)
        scores = torch.max(
            lower_preds - targets,
            targets - upper_preds
        )
        
        # Store calibration scores
        n = len(scores)
        self.calibration_scores[:n] = scores.sort()[0]
        self.n_calibration = n
        
        # Compute conformal quantile
        q_level = np.ceil((1 - alpha) * (n + 1)) / n
        q_level = min(q_level, 1.0)
        
        idx = int(q_level * n)
        self.conformal_quantile = self.calibration_scores[idx].item()
    
    def predict_interval(
        self,
        x: torch.Tensor,
        alpha: float = 0.1
    ) -> ConformalInterval:
        """
        Generate conformalized prediction intervals.
        
        Args:
            x: (batch, input_dim) features
            alpha: Miscoverage rate
            
        Returns:
            ConformalInterval with guaranteed coverage
        """
        self.eval()
        
        with torch.no_grad():
            quantiles = self(x)
            
            lower = quantiles[:, 0].cpu().numpy()
            upper = quantiles[:, -1].cpu().numpy()
            
            # Apply conformal correction
            if hasattr(self, 'conformal_quantile'):
                lower = lower - self.conformal_quantile
                upper = upper + self.conformal_quantile
        
        return ConformalInterval(
            lower=lower,
            upper=upper,
            coverage=1 - alpha,
            width=upper - lower
        )


class AdaptiveConformalInference:
    """
    Adaptive Conformal Inference (ACI) for non-exchangeable data.
    
    Maintains valid coverage even when data distribution shifts over time.
    """
    
    def __init__(
        self,
        target_coverage: float = 0.95,
        gamma: float = 0.01,
        initial_alpha: float = 0.05
    ):
        self.target_coverage = target_coverage
        self.gamma = gamma  # Learning rate for alpha adaptation
        self.alpha_t = initial_alpha
        self.alpha_history = [initial_alpha]
        self.coverage_history = []
    
    def update(self, error_t: float):
        """
        Update alpha based on observed error.
        
        Args:
            error_t: 1 if prediction interval missed, 0 otherwise
        """
        # Adaptive update rule
        self.alpha_t = self.alpha_t + self.gamma * (error_t - (1 - self.target_coverage))
        
        # Clip to valid range
        self.alpha_t = np.clip(self.alpha_t, 0.001, 0.999)
        
        self.alpha_history.append(self.alpha_t)
        self.coverage_history.append(1 - error_t)
    
    def get_alpha(self) -> float:
        """Get current alpha for prediction interval."""
        return self.alpha_t
    
    def get_running_coverage(self, window: int = 100) -> float:
        """Compute running coverage over recent predictions."""
        if len(self.coverage_history) < window:
            return np.mean(self.coverage_history) if self.coverage_history else self.target_coverage
        return np.mean(self.coverage_history[-window:])


class UncertaintyAwarePortfolio(nn.Module):
    """
    Portfolio optimization with uncertainty quantification.
    
    Adjusts positions based on prediction confidence.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        conformal_model: ConformizedQuantileRegression,
        risk_aversion: float = 1.0,
        uncertainty_penalty: float = 0.5
    ):
        super().__init__()
        
        self.base_model = base_model
        self.conformal_model = conformal_model
        self.risk_aversion = risk_aversion
        self.uncertainty_penalty = uncertainty_penalty
        self.aci = AdaptiveConformalInference()
    
    def forward(
        self,
        features: torch.Tensor,
        return_uncertainty: bool = True
    ) -> dict:
        """
        Generate uncertainty-aware portfolio weights.
        """
        # Get base predictions
        base_output = self.base_model(features)
        base_weights = base_output['weights']
        embeddings = base_output['embeddings']
        
        batch, n_assets, d_model = embeddings.shape
        
        # Get prediction intervals for each asset
        embeddings_flat = embeddings.view(batch * n_assets, d_model)
        
        alpha = self.aci.get_alpha()
        intervals = self.conformal_model.predict_interval(embeddings_flat, alpha)
        
        # Reshape intervals
        interval_widths = torch.FloatTensor(intervals.width).view(batch, n_assets)
        
        # Normalize widths to get uncertainty scores
        uncertainty = interval_widths / (interval_widths.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        # Adjust weights based on uncertainty
        # Reduce position size for high-uncertainty assets
        confidence = 1 - uncertainty
        adjusted_weights = base_weights * confidence
        
        # Renormalize
        adjusted_weights = adjusted_weights / (adjusted_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        output = {
            'weights': adjusted_weights,
            'base_weights': base_weights,
            'uncertainty': uncertainty,
            'interval_lower': intervals.lower.reshape(batch, n_assets),
            'interval_upper': intervals.upper.reshape(batch, n_assets)
        }
        
        return output
    
    def update_coverage(self, predictions: np.ndarray, actuals: np.ndarray):
        """Update ACI with observed prediction errors."""
        for pred, actual in zip(predictions, actuals):
            # Check if actual was outside interval (simplified)
            error = 0 if pred > actual else 1
            self.aci.update(error)

