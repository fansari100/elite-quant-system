"""
Signature-Informed Transformer (SIT)
=====================================
State-of-art architecture combining path signatures with transformers.
Based on arXiv:2510.03129

Key innovations:
1. Path signatures for geometric representation of asset dynamics
2. Signature-augmented attention with financial inductive biases
3. End-to-end optimization of risk-aware objectives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PathSignatureLayer(nn.Module):
    """
    Compute path signatures and project to embedding space.
    """
    
    def __init__(self, input_dim: int, output_dim: int, depth: int = 4):
        super().__init__()
        self.depth = depth
        self.input_dim = input_dim
        
        # Estimate signature dimension
        sig_dim = sum(input_dim ** i for i in range(1, depth + 1))
        
        self.projection = nn.Sequential(
            nn.Linear(sig_dim, output_dim * 2),
            nn.GELU(),
            nn.LayerNorm(output_dim * 2),
            nn.Linear(output_dim * 2, output_dim)
        )
    
    def _compute_signature(self, path: torch.Tensor) -> torch.Tensor:
        """
        Compute path signature (fallback implementation).
        For production, use signatory library for GPU acceleration.
        """
        batch, seq_len, d = path.shape
        device = path.device
        
        # Add time augmentation
        time = torch.linspace(0, 1, seq_len, device=device).view(1, -1, 1).expand(batch, -1, 1)
        path = torch.cat([time, path], dim=-1)
        d = d + 1
        
        # Compute increments
        increments = path[:, 1:] - path[:, :-1]  # (batch, seq_len-1, d)
        
        # Level 1: sum of increments
        sig_1 = increments.sum(dim=1)  # (batch, d)
        
        # Level 2: iterated integrals (areas)
        cumsum = torch.cumsum(increments, dim=1)
        sig_2_list = []
        for i in range(d):
            for j in range(d):
                area = (cumsum[:, :-1, i] * increments[:, 1:, j]).sum(dim=1)
                sig_2_list.append(area)
        sig_2 = torch.stack(sig_2_list, dim=1)  # (batch, d*d)
        
        # Level 3: simplified third-order terms
        sig_3_list = []
        for i in range(min(d, 3)):
            for j in range(min(d, 3)):
                for k in range(min(d, 3)):
                    if len(sig_3_list) < 27:  # Limit dimension
                        term = (cumsum[:, :-1, i] * cumsum[:, :-1, j] * increments[:, 1:, k]).sum(dim=1)
                        sig_3_list.append(term)
        sig_3 = torch.stack(sig_3_list, dim=1) if sig_3_list else torch.zeros(batch, 1, device=device)
        
        return torch.cat([sig_1, sig_2, sig_3], dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_assets, seq_len, features)
            
        Returns:
            sig_features: (batch, n_assets, output_dim)
        """
        batch, n_assets, seq_len, features = x.shape
        
        # Compute signature for each asset
        x_flat = x.view(batch * n_assets, seq_len, features)
        sig = self._compute_signature(x_flat)
        sig = sig.view(batch, n_assets, -1)
        
        # Project to output dimension
        sig_features = self.projection(sig)
        
        return sig_features


class SignatureAugmentedAttention(nn.Module):
    """
    Multi-head attention augmented with path signature biases.
    Captures lead-lag relationships between assets.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Signature bias network
        self.sig_bias_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_heads)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sig_features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query, key, value: (batch, n_assets, d_model)
            sig_features: (batch, n_assets, d_model) path signature features
            
        Returns:
            output: (batch, n_assets, d_model)
            attention: (batch, n_heads, n_assets, n_assets)
        """
        batch, n_assets, _ = query.shape
        
        # Linear projections
        Q = self.w_q(query).view(batch, n_assets, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(key).view(batch, n_assets, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(value).view(batch, n_assets, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, n_heads, n_assets, n_assets)
        
        # Add signature-based bias (captures lead-lag relationships)
        if sig_features is not None:
            # Compute pairwise signature similarity
            sig_norm = F.normalize(sig_features, dim=-1)
            sig_similarity = torch.matmul(sig_norm, sig_norm.transpose(-2, -1))  # (batch, n_assets, n_assets)
            
            # Compute asymmetric lead-lag bias
            sig_bias = self.sig_bias_net(sig_features)  # (batch, n_assets, n_heads)
            sig_bias = sig_bias.unsqueeze(2) - sig_bias.unsqueeze(1)  # (batch, n_assets, n_assets, n_heads)
            sig_bias = sig_bias.permute(0, 3, 1, 2)  # (batch, n_heads, n_assets, n_assets)
            
            scores = scores + sig_bias * 0.1
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        output = torch.matmul(attention, V)
        output = output.transpose(1, 2).contiguous().view(batch, n_assets, self.d_model)
        output = self.w_o(output)
        
        return output, attention


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with signature attention."""
    
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = SignatureAugmentedAttention(d_model, n_heads, dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        sig_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with signature augmentation
        attn_output, attn_weights = self.self_attn(x, x, x, sig_features)
        x = self.norm1(x + self.dropout(attn_output))
        
        # FFN
        x = self.norm2(x + self.ffn(x))
        
        return x, attn_weights


class PortfolioHead(nn.Module):
    """
    Portfolio allocation head with risk-aware output.
    """
    
    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        max_position: float = 0.1
    ):
        super().__init__()
        self.max_position = max_position
        
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_assets, d_model)
            
        Returns:
            weights: (batch, n_assets) portfolio weights summing to 1
        """
        # Get raw scores
        scores = self.mlp(x).squeeze(-1)  # (batch, n_assets)
        
        # Apply softmax for long-only portfolio
        weights = F.softmax(scores, dim=-1)
        
        # Clip to max position
        weights = torch.clamp(weights, 0, self.max_position)
        
        # Renormalize
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        return weights


class SignatureInformedTransformer(nn.Module):
    """
    Complete Signature-Informed Transformer for portfolio allocation.
    
    Architecture:
    1. Feature embedding with path signature augmentation
    2. Transformer encoder with signature-augmented attention
    3. Portfolio head with risk-aware output
    """
    
    def __init__(
        self,
        input_dim: int,
        n_assets: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        signature_depth: int = 4,
        max_position: float = 0.1,
        risk_aversion: float = 1.0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_assets = n_assets
        self.risk_aversion = risk_aversion
        
        # Feature embedding
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        
        # Path signature layer
        self.signature_layer = PathSignatureLayer(input_dim, d_model, signature_depth)
        
        # Positional encoding for sequence
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Temporal aggregation
        self.temporal_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])
        
        # Portfolio head
        self.portfolio_head = PortfolioHead(d_model, d_model * 2, dropout, max_position)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, n_assets, seq_len, n_features) input features
            
        Returns:
            dict with 'weights', 'embeddings', 'attention_maps'
        """
        batch, n_assets, seq_len, n_features = x.shape
        
        # Compute path signature features
        sig_features = self.signature_layer(x)  # (batch, n_assets, d_model)
        
        # Embed features
        x = x.view(batch * n_assets, seq_len, n_features)
        x = self.feature_embedding(x)  # (batch*n_assets, seq_len, d_model)
        x = self.pos_encoder(x)
        
        # Temporal aggregation (attention pooling over time)
        attn_weights = self.temporal_pool(x)  # (batch*n_assets, seq_len, 1)
        x = (x * attn_weights).sum(dim=1)  # (batch*n_assets, d_model)
        x = x.view(batch, n_assets, self.d_model)
        
        # Add signature features
        x = x + sig_features
        
        # Apply transformer encoder with signature attention
        attention_maps = []
        for layer in self.encoder_layers:
            x, attn = layer(x, sig_features)
            if return_attention:
                attention_maps.append(attn)
        
        # Get portfolio weights
        weights = self.portfolio_head(x)
        
        output = {
            'weights': weights,
            'embeddings': x,
            'signature_features': sig_features
        }
        
        if return_attention:
            output['attention_maps'] = attention_maps
        
        return output
    
    def compute_portfolio_return(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor
    ) -> torch.Tensor:
        """Compute portfolio return."""
        return (weights * returns).sum(dim=-1)
    
    def compute_loss(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        covariance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute risk-aware loss function.
        
        Maximize: E[R] - (lambda/2) * Var[R]
        """
        # Expected return
        portfolio_return = self.compute_portfolio_return(weights, returns)
        
        # Variance (if covariance provided)
        if covariance is not None:
            variance = torch.einsum('bi,bij,bj->b', weights, covariance, weights)
        else:
            # Use sample variance from returns
            variance = (weights ** 2 * returns ** 2).sum(dim=-1)
        
        # Risk-adjusted objective (negative because we minimize)
        loss = -(portfolio_return - self.risk_aversion / 2 * variance)
        
        return loss.mean()

