"""
PyTorch Lightning Training Module
==================================
Optimized for NVIDIA H200 with mixed precision and distributed training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar
)
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from typing import Dict, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass

import sys
sys.path.append('..')

from models.signature_transformer import SignatureInformedTransformer
from agents.multi_agent import MultiAgentCoordinator, MarketState


class QuantLightningModule(L.LightningModule):
    """
    Lightning module combining SIT with multi-agent coordination.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_assets: int,
        config: Any
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.n_assets = n_assets
        
        # Main model: Signature-Informed Transformer
        self.sit_model = SignatureInformedTransformer(
            input_dim=input_dim,
            n_assets=n_assets,
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            n_layers=config.model.n_encoder_layers,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
            signature_depth=config.model.signature_depth,
            max_position=config.model.max_position_size,
            risk_aversion=config.model.risk_aversion
        )
        
        # Multi-agent coordinator (optional refinement)
        self.use_agents = config.agent.enable_alpha_agent
        if self.use_agents:
            self.agent_coordinator = MultiAgentCoordinator(
                input_dim=input_dim,
                n_assets=n_assets,
                d_model=config.model.d_model,
                config=config
            )
        
        # Loss components
        self.sharpe_weight = 1.0
        self.turnover_penalty = 0.001
        self.concentration_penalty = 0.01
        
        # Metrics tracking
        self.train_returns = []
        self.val_returns = []
        
        # Previous weights for turnover calculation
        self.register_buffer('prev_weights', torch.zeros(n_assets))
    
    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            features: (batch, n_assets, seq_len, n_features)
        """
        # Get SIT predictions
        sit_output = self.sit_model(features, return_attention=return_attention)
        
        return sit_output
    
    def compute_sharpe_loss(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        risk_free_rate: float = 0.0
    ) -> torch.Tensor:
        """
        Compute negative Sharpe ratio as loss.
        """
        # Portfolio returns
        portfolio_returns = (weights * returns).sum(dim=-1)  # (batch,)
        
        # Annualized metrics (assuming daily returns)
        mean_return = portfolio_returns.mean() * 252
        std_return = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (negative for minimization)
        sharpe = (mean_return - risk_free_rate) / (std_return + 1e-8)
        
        return -sharpe
    
    def compute_turnover_loss(
        self,
        weights: torch.Tensor,
        prev_weights: torch.Tensor
    ) -> torch.Tensor:
        """Penalize excessive trading."""
        turnover = (weights - prev_weights).abs().sum(dim=-1).mean()
        return turnover
    
    def compute_concentration_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """Penalize concentrated portfolios (maximize diversification)."""
        # Herfindahl index
        hhi = (weights ** 2).sum(dim=-1).mean()
        return hhi
    
    def compute_loss(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        prev_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with multiple objectives.
        """
        # Sharpe loss
        sharpe_loss = self.compute_sharpe_loss(weights, returns)
        
        # Turnover penalty
        if prev_weights is None:
            prev_weights = self.prev_weights.unsqueeze(0).expand(weights.shape[0], -1)
        turnover_loss = self.compute_turnover_loss(weights, prev_weights)
        
        # Concentration penalty
        concentration_loss = self.compute_concentration_loss(weights)
        
        # Total loss
        total_loss = (
            self.sharpe_weight * sharpe_loss +
            self.turnover_penalty * turnover_loss +
            self.concentration_penalty * concentration_loss
        )
        
        # Metrics
        metrics = {
            'sharpe_loss': sharpe_loss.item(),
            'turnover_loss': turnover_loss.item(),
            'concentration_loss': concentration_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, metrics
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        features = batch['features']
        returns = batch['target_returns']
        
        # Forward pass
        output = self(features)
        weights = output['weights']
        
        # Compute loss
        loss, metrics = self.compute_loss(weights, returns)
        
        # Log metrics
        for key, value in metrics.items():
            self.log(f'train/{key}', value, on_step=True, on_epoch=True, prog_bar=(key == 'total_loss'))
        
        # Track returns
        portfolio_returns = (weights * returns).sum(dim=-1)
        self.train_returns.extend(portfolio_returns.detach().cpu().numpy().tolist())
        
        # Update previous weights
        self.prev_weights = weights[-1].detach()
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        features = batch['features']
        returns = batch['target_returns']
        
        # Forward pass
        output = self(features)
        weights = output['weights']
        
        # Compute loss
        loss, metrics = self.compute_loss(weights, returns)
        
        # Log metrics
        for key, value in metrics.items():
            self.log(f'val/{key}', value, on_epoch=True, prog_bar=(key == 'sharpe_loss'))
        
        # Track returns
        portfolio_returns = (weights * returns).sum(dim=-1)
        self.val_returns.extend(portfolio_returns.detach().cpu().numpy().tolist())
        
        return loss
    
    def on_validation_epoch_end(self):
        """Compute validation Sharpe ratio."""
        if len(self.val_returns) > 10:
            returns = np.array(self.val_returns)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            self.log('val/sharpe_ratio', sharpe, prog_bar=True)
            
            # Clear for next epoch
            self.val_returns = []
    
    def on_train_epoch_end(self):
        """Compute training Sharpe ratio."""
        if len(self.train_returns) > 10:
            returns = np.array(self.train_returns)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            self.log('train/sharpe_ratio', sharpe)
            
            self.train_returns = []
    
    def configure_optimizers(self):
        """Configure optimizer with learning rate scheduling."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Cosine annealing with warmup
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.training.learning_rate * 10,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos'
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }


def get_callbacks(config) -> list:
    """Get training callbacks."""
    callbacks = [
        # Checkpoint best model
        ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            filename='sit-{epoch:02d}-{val/sharpe_loss:.4f}',
            monitor='val/sharpe_loss',
            mode='min',
            save_top_k=config.training.save_top_k,
            save_last=True
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val/sharpe_loss',
            patience=config.training.patience,
            mode='min',
            verbose=True
        ),
        
        # LR monitoring
        LearningRateMonitor(logging_interval='step'),
        
        # Rich progress bar
        RichProgressBar()
    ]
    
    return callbacks


def get_logger(config, name: str = "elite-quant") -> Any:
    """Get experiment logger."""
    try:
        # Try wandb first
        logger = WandbLogger(
            project=name,
            save_dir=config.log_dir,
            log_model=True
        )
    except Exception:
        # Fallback to TensorBoard
        logger = TensorBoardLogger(
            save_dir=config.log_dir,
            name=name
        )
    
    return logger

