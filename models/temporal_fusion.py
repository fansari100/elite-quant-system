"""
Temporal Fusion Transformer (TFT) for Finance
==============================================
State-of-the-art interpretable time series forecasting.

Based on:
- Google's TFT paper (Lim et al., 2021)
- Enhanced with path signatures and conformal prediction

Key Features:
- Variable selection networks for interpretability
- Static covariate encoders (sector, market cap)
- Temporal attention for long-range dependencies
- Multi-horizon forecasting with quantile outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


class GatedLinearUnit(nn.Module):
    """GLU activation with optional dropout."""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.fc(x)
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    """GRN with ELU activation and optional context vector."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        
        # Main layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Context projection
        if context_dim is not None:
            self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        
        # GLU for output
        self.glu = GatedLinearUnit(hidden_dim, output_dim, dropout)
        
        # Residual connection
        if input_dim != output_dim:
            self.skip_proj = nn.Linear(input_dim, output_dim)
        else:
            self.skip_proj = None
        
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Residual
        residual = x if self.skip_proj is None else self.skip_proj(x)
        
        # Main path
        x = self.fc1(x)
        
        # Add context if provided
        if context is not None and self.context_dim is not None:
            x = x + self.context_proj(context)
        
        x = F.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.glu(x)
        
        # Add residual and normalize
        x = self.layer_norm(x + residual)
        
        return x


class VariableSelectionNetwork(nn.Module):
    """
    Variable selection for interpretability.
    Learns which features are most important.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_vars: int,
        hidden_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_vars = n_vars
        self.hidden_dim = hidden_dim
        
        # Variable-specific GRNs
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim // n_vars, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(n_vars)
        ])
        
        # Flattened GRN for variable weights
        self.weight_grn = GatedResidualNetwork(
            n_vars * hidden_dim,
            hidden_dim,
            n_vars,
            context_dim=context_dim,
            dropout=dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, n_vars * var_dim) or (batch, n_vars * var_dim)
            
        Returns:
            output: (batch, seq_len, hidden_dim) or (batch, hidden_dim)
            weights: (batch, seq_len, n_vars) or (batch, n_vars)
        """
        # Handle both 2D and 3D inputs
        is_temporal = x.dim() == 3
        if not is_temporal:
            x = x.unsqueeze(1)
        
        batch, seq_len, total_dim = x.shape
        var_dim = total_dim // self.n_vars
        
        # Split into variables
        x_vars = x.view(batch, seq_len, self.n_vars, var_dim)
        
        # Apply variable-specific GRNs
        var_outputs = []
        for i, grn in enumerate(self.var_grns):
            var_out = grn(x_vars[:, :, i, :])
            var_outputs.append(var_out)
        
        # Stack variable outputs
        var_outputs = torch.stack(var_outputs, dim=-2)  # (batch, seq_len, n_vars, hidden)
        
        # Flatten for weight computation
        var_flat = var_outputs.view(batch, seq_len, -1)
        
        # Compute variable weights
        if context is not None:
            context = context.unsqueeze(1).expand(-1, seq_len, -1)
        weights = self.weight_grn(var_flat, context)
        weights = F.softmax(weights, dim=-1)  # (batch, seq_len, n_vars)
        
        # Weighted sum
        output = (var_outputs * weights.unsqueeze(-1)).sum(dim=-2)
        
        if not is_temporal:
            output = output.squeeze(1)
            weights = weights.squeeze(1)
        
        return output, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention with interpretable attention weights.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = query.shape
        
        # Linear projections
        Q = self.w_q(query).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(key).view(batch, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(value).view(batch, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention
        output = torch.matmul(attention, V)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.w_o(output)
        
        # Average attention across heads for interpretability
        attention_weights = attention.mean(dim=1)
        
        return output, attention_weights


class TemporalFusionTransformer(nn.Module):
    """
    Full TFT implementation for financial time series.
    """
    
    def __init__(
        self,
        n_temporal_features: int,
        n_static_features: int,
        n_assets: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_lstm_layers: int = 2,
        dropout: float = 0.1,
        n_quantiles: int = 3,
        forecast_horizon: int = 5
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_assets = n_assets
        self.forecast_horizon = forecast_horizon
        self.n_quantiles = n_quantiles
        
        # Static variable selection
        self.static_vsn = VariableSelectionNetwork(
            n_static_features, 
            max(1, n_static_features // 8),
            d_model,
            dropout=dropout
        )
        
        # Static context encoders
        self.static_context_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
        self.static_enrichment_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
        
        # Temporal variable selection
        self.temporal_vsn = VariableSelectionNetwork(
            n_temporal_features,
            max(1, n_temporal_features // 4),
            d_model,
            context_dim=d_model,
            dropout=dropout
        )
        
        # LSTM encoder
        self.lstm_encoder = nn.LSTM(
            d_model, d_model, n_lstm_layers,
            batch_first=True, dropout=dropout if n_lstm_layers > 1 else 0
        )
        
        # Temporal self-attention
        self.attention = InterpretableMultiHeadAttention(d_model, n_heads, dropout)
        self.attention_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
        self.attention_layer_norm = nn.LayerNorm(d_model)
        
        # Position-wise feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_layer_norm = nn.LayerNorm(d_model)
        
        # Output layers - quantile predictions
        self.output_layer = nn.Linear(d_model, n_quantiles * forecast_horizon)
        
        # Portfolio allocation head
        self.portfolio_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(
        self,
        temporal_features: torch.Tensor,
        static_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            temporal_features: (batch, n_assets, seq_len, n_temporal_features)
            static_features: (batch, n_assets, n_static_features) - optional
            
        Returns:
            dict with predictions, attention weights, variable importance
        """
        batch, n_assets, seq_len, n_features = temporal_features.shape
        
        # Flatten batch and assets
        temporal_flat = temporal_features.view(batch * n_assets, seq_len, n_features)
        
        # Static encoding
        if static_features is not None:
            static_flat = static_features.view(batch * n_assets, -1)
            static_encoded, static_weights = self.static_vsn(static_flat)
            static_context = self.static_context_grn(static_encoded)
            static_enrichment = self.static_enrichment_grn(static_encoded)
        else:
            static_context = None
            static_enrichment = None
            static_weights = None
        
        # Temporal variable selection
        temporal_encoded, temporal_weights = self.temporal_vsn(temporal_flat, static_context)
        
        # LSTM encoding
        lstm_out, _ = self.lstm_encoder(temporal_encoded)
        
        # Add static enrichment
        if static_enrichment is not None:
            lstm_out = lstm_out + static_enrichment.unsqueeze(1)
        
        # Temporal self-attention (use last portion for efficiency)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attention_grn(attn_out)
        attn_out = self.attention_layer_norm(lstm_out + attn_out)
        
        # FFN
        ffn_out = self.ffn(attn_out)
        output = self.ffn_layer_norm(attn_out + ffn_out)
        
        # Use last timestep for predictions
        final_output = output[:, -1, :]  # (batch * n_assets, d_model)
        
        # Quantile predictions
        quantile_preds = self.output_layer(final_output)
        quantile_preds = quantile_preds.view(batch * n_assets, self.forecast_horizon, self.n_quantiles)
        quantile_preds = quantile_preds.view(batch, n_assets, self.forecast_horizon, self.n_quantiles)
        
        # Portfolio weights
        scores = self.portfolio_head(final_output).view(batch, n_assets)
        weights = F.softmax(scores, dim=-1)
        
        # Point predictions (median quantile)
        point_predictions = quantile_preds[:, :, :, self.n_quantiles // 2]
        
        return {
            'quantile_predictions': quantile_preds,
            'point_predictions': point_predictions,
            'weights': weights,
            'temporal_attention': attn_weights.view(batch, n_assets, seq_len, seq_len),
            'temporal_variable_importance': temporal_weights.view(batch, n_assets, seq_len, -1),
            'static_variable_importance': static_weights.view(batch, n_assets, -1) if static_weights is not None else None
        }


class RegimeDetector(nn.Module):
    """
    Market regime detection using Hidden Markov Model-inspired neural network.
    
    Identifies market states:
    - Bull (trending up)
    - Bear (trending down)
    - High volatility
    - Low volatility / Range-bound
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_regimes: int = 4,
        seq_len: int = 60
    ):
        super().__init__()
        
        self.n_regimes = n_regimes
        
        # Temporal encoder
        self.lstm = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True, bidirectional=True)
        
        # Regime classifier
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_regimes)
        )
        
        # Transition matrix (learnable)
        self.transition_logits = nn.Parameter(torch.zeros(n_regimes, n_regimes))
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
            
        Returns:
            regime_probs: (batch, n_regimes)
            regime_sequence: (batch, seq_len, n_regimes)
        """
        # Encode sequence
        lstm_out, (h_n, _) = self.lstm(x)
        
        # Regime probabilities at each timestep
        regime_logits = self.regime_head(lstm_out)
        regime_probs_seq = F.softmax(regime_logits, dim=-1)
        
        # Final regime (from last hidden state)
        h_final = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        final_logits = self.regime_head(h_final.unsqueeze(1)).squeeze(1)
        regime_probs = F.softmax(final_logits, dim=-1)
        
        # Transition matrix
        transition_matrix = F.softmax(self.transition_logits, dim=-1)
        
        return {
            'regime_probs': regime_probs,
            'regime_sequence': regime_probs_seq,
            'transition_matrix': transition_matrix,
            'regime_labels': regime_probs.argmax(dim=-1)
        }

