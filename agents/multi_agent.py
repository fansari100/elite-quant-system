"""
Multi-Agent Alpha Generation System
=====================================
Inspired by R&D-Agent-Quant and QuantAgents frameworks.

Implements a collaborative multi-agent system where specialized agents
communicate to generate robust alpha signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender: str
    content: Dict[str, Any]
    confidence: float = 1.0
    timestamp: int = 0


@dataclass 
class MarketState:
    """Current market state representation."""
    features: torch.Tensor           # (n_assets, seq_len, n_features)
    returns: torch.Tensor            # (n_assets, horizon)
    volatility: torch.Tensor         # (n_assets,)
    correlations: torch.Tensor       # (n_assets, n_assets)
    sentiment_scores: Optional[torch.Tensor] = None
    sector_ids: Optional[torch.Tensor] = None


class BaseAgent(ABC, nn.Module):
    """Base class for all trading agents."""
    
    def __init__(self, name: str, d_model: int = 256):
        super().__init__()
        self.name = name
        self.d_model = d_model
        self.message_inbox: List[AgentMessage] = []
    
    @abstractmethod
    def forward(self, state: MarketState) -> Dict[str, torch.Tensor]:
        """Process market state and return agent output."""
        pass
    
    def receive_message(self, message: AgentMessage):
        """Receive message from another agent."""
        self.message_inbox.append(message)
    
    def send_message(self, content: Dict[str, Any], confidence: float = 1.0) -> AgentMessage:
        """Create message to send to other agents."""
        return AgentMessage(
            sender=self.name,
            content=content,
            confidence=confidence
        )
    
    def clear_inbox(self):
        """Clear message inbox."""
        self.message_inbox = []


class AlphaAgent(BaseAgent):
    """
    Alpha generation agent.
    
    Responsibilities:
    - Generate alpha signals from features
    - Score assets by expected return
    - Identify lead-lag relationships
    """
    
    def __init__(self, input_dim: int, n_assets: int, d_model: int = 256, n_factors: int = 50):
        super().__init__("alpha_agent", d_model)
        
        self.n_factors = n_factors
        self.n_assets = n_assets
        
        # Factor extraction network
        self.factor_net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_factors)
        )
        
        # Alpha scoring network
        self.alpha_net = nn.Sequential(
            nn.Linear(n_factors, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Lead-lag detection (cross-asset attention)
        self.lead_lag_attention = nn.MultiheadAttention(n_factors, num_heads=5, dropout=0.1)
        
        # Factor momentum (exponential moving average)
        self.register_buffer('factor_ema', torch.zeros(n_factors))
        self.ema_decay = 0.95
    
    def forward(self, state: MarketState) -> Dict[str, torch.Tensor]:
        # Aggregate temporal features
        features = state.features.mean(dim=1)  # (n_assets, n_features)
        
        # Extract factors
        factors = self.factor_net(features)  # (n_assets, n_factors)
        
        # Lead-lag cross-asset attention
        factors_t = factors.unsqueeze(0)  # (1, n_assets, n_factors)
        lead_lag_factors, attention = self.lead_lag_attention(
            factors_t, factors_t, factors_t
        )
        factors = factors + lead_lag_factors.squeeze(0)
        
        # Generate alpha scores
        alpha_scores = self.alpha_net(factors).squeeze(-1)  # (n_assets,)
        
        # Normalize to z-scores
        alpha_scores = (alpha_scores - alpha_scores.mean()) / (alpha_scores.std() + 1e-8)
        
        # Create message
        message = self.send_message({
            'alpha_scores': alpha_scores,
            'factors': factors,
            'lead_lag_attention': attention
        }, confidence=0.8)
        
        return {
            'alpha_scores': alpha_scores,
            'factors': factors,
            'attention': attention,
            'message': message
        }


class RiskAgent(BaseAgent):
    """
    Risk management agent.
    
    Responsibilities:
    - Compute risk metrics (VaR, CVaR, volatility)
    - Enforce position limits
    - Monitor correlation/concentration risks
    """
    
    def __init__(
        self,
        n_assets: int,
        d_model: int = 256,
        var_confidence: float = 0.99,
        max_position: float = 0.10,
        max_sector: float = 0.30
    ):
        super().__init__("risk_agent", d_model)
        
        self.n_assets = n_assets
        self.var_confidence = var_confidence
        self.max_position = max_position
        self.max_sector = max_sector
        
        # Risk prediction network
        self.risk_net = nn.Sequential(
            nn.Linear(n_assets, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_assets)
        )
        
        # Covariance estimation network
        self.cov_net = nn.Sequential(
            nn.Linear(n_assets, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_assets * n_assets)
        )
    
    def forward(self, state: MarketState) -> Dict[str, torch.Tensor]:
        volatility = state.volatility
        correlations = state.correlations
        
        # Risk scores (higher = riskier)
        risk_scores = self.risk_net(volatility.unsqueeze(0)).squeeze(0)
        risk_scores = torch.sigmoid(risk_scores)
        
        # Estimate covariance matrix
        cov_flat = self.cov_net(volatility.unsqueeze(0))
        cov_matrix = cov_flat.view(self.n_assets, self.n_assets)
        cov_matrix = torch.mm(cov_matrix, cov_matrix.t())  # Ensure PSD
        
        # Compute position limits based on risk
        position_limits = self.max_position * (1 - 0.5 * risk_scores)
        
        message = self.send_message({
            'risk_scores': risk_scores,
            'covariance': cov_matrix,
            'position_limits': position_limits
        }, confidence=0.9)
        
        return {
            'risk_scores': risk_scores,
            'covariance': cov_matrix,
            'position_limits': position_limits,
            'message': message
        }
    
    def apply_constraints(
        self,
        weights: torch.Tensor,
        position_limits: torch.Tensor,
        sector_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply risk constraints to portfolio weights."""
        # Position limits
        weights = torch.clamp(weights, -position_limits, position_limits)
        
        # Sector limits
        if sector_ids is not None:
            for sector in sector_ids.unique():
                mask = sector_ids == sector
                sector_weight = weights[mask].sum()
                if sector_weight > self.max_sector:
                    weights[mask] *= self.max_sector / sector_weight
        
        # Renormalize
        weights = weights / (weights.abs().sum() + 1e-8)
        
        return weights


class SentimentAgent(BaseAgent):
    """
    Sentiment analysis agent using FinBERT.
    
    Responsibilities:
    - Process news/social media sentiment
    - Generate sentiment scores per asset
    - Detect sentiment momentum
    """
    
    def __init__(self, n_assets: int, d_model: int = 256, model_name: str = "ProsusAI/finbert"):
        super().__init__("sentiment_agent", d_model)
        
        self.n_assets = n_assets
        self.model_name = model_name
        self._model_loaded = False
        
        # Sentiment projection
        self.sentiment_projection = nn.Sequential(
            nn.Linear(768, d_model),  # FinBERT hidden size
            nn.GELU(),
            nn.Linear(d_model, 3)  # positive, negative, neutral
        )
        
        # Sentiment momentum tracker
        self.register_buffer('sentiment_ema', torch.zeros(n_assets))
        self.ema_decay = 0.9
    
    def _load_model(self):
        """Lazy load the sentiment model."""
        if not self._model_loaded:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.bert_model = AutoModel.from_pretrained(self.model_name)
                self.bert_model.eval()
                for param in self.bert_model.parameters():
                    param.requires_grad = False
                self._model_loaded = True
            except Exception as e:
                logger.warning(f"Could not load sentiment model: {e}")
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode texts using FinBERT."""
        self._load_model()
        
        if not self._model_loaded:
            return torch.zeros(len(texts), 768)
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]  # CLS token
        
        return embeddings
    
    def forward(
        self,
        state: MarketState,
        news_texts: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process sentiment data.
        
        Args:
            state: Current market state
            news_texts: Dict mapping ticker to list of news texts
        """
        if state.sentiment_scores is not None:
            # Use pre-computed sentiment
            sentiment = state.sentiment_scores
        else:
            # Default neutral sentiment
            sentiment = torch.zeros(self.n_assets, 3)
            sentiment[:, 2] = 1.0  # Neutral
        
        # Sentiment momentum
        sentiment_signal = sentiment[:, 0] - sentiment[:, 1]  # positive - negative
        self.sentiment_ema = self.ema_decay * self.sentiment_ema + (1 - self.ema_decay) * sentiment_signal.detach()
        
        sentiment_momentum = sentiment_signal - self.sentiment_ema
        
        message = self.send_message({
            'sentiment': sentiment,
            'sentiment_signal': sentiment_signal,
            'sentiment_momentum': sentiment_momentum
        }, confidence=0.6)
        
        return {
            'sentiment': sentiment,
            'sentiment_signal': sentiment_signal,
            'sentiment_momentum': sentiment_momentum,
            'message': message
        }


class ExecutionAgent(BaseAgent):
    """
    Trade execution agent.
    
    Responsibilities:
    - Optimize execution timing
    - Minimize market impact
    - Handle transaction costs
    """
    
    def __init__(
        self,
        n_assets: int,
        d_model: int = 256,
        transaction_cost_bps: float = 5.0,
        min_trade_size: float = 0.001
    ):
        super().__init__("execution_agent", d_model)
        
        self.n_assets = n_assets
        self.transaction_cost = transaction_cost_bps / 10000
        self.min_trade_size = min_trade_size
        
        # Execution timing network
        self.timing_net = nn.Sequential(
            nn.Linear(n_assets * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_assets)
        )
        
        # Current positions
        self.register_buffer('current_positions', torch.zeros(n_assets))
    
    def forward(
        self,
        target_weights: torch.Tensor,
        current_weights: Optional[torch.Tensor] = None,
        state: Optional[MarketState] = None
    ) -> Dict[str, torch.Tensor]:
        if current_weights is None:
            current_weights = self.current_positions
        
        # Compute trades
        raw_trades = target_weights - current_weights
        
        # Apply minimum trade filter
        trade_mask = raw_trades.abs() > self.min_trade_size
        trades = raw_trades * trade_mask.float()
        
        # Estimate transaction costs
        costs = trades.abs() * self.transaction_cost
        
        # Execution timing scores (higher = execute now)
        timing_input = torch.cat([trades.unsqueeze(0), current_weights.unsqueeze(0)], dim=-1)
        timing_scores = torch.sigmoid(self.timing_net(timing_input).squeeze(0))
        
        # Apply timing filter (only execute high-priority trades)
        final_trades = trades * (timing_scores > 0.5).float()
        
        # Update positions
        new_positions = current_weights + final_trades
        self.current_positions = new_positions.detach()
        
        message = self.send_message({
            'trades': final_trades,
            'costs': costs.sum(),
            'timing_scores': timing_scores
        }, confidence=0.95)
        
        return {
            'trades': final_trades,
            'transaction_costs': costs,
            'timing_scores': timing_scores,
            'new_positions': new_positions,
            'message': message
        }


class MultiAgentCoordinator(nn.Module):
    """
    Coordinator that orchestrates multi-agent communication and decision-making.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_assets: int,
        d_model: int = 256,
        config: Optional[Any] = None
    ):
        super().__init__()
        
        self.n_assets = n_assets
        self.d_model = d_model
        
        # Initialize agents
        self.alpha_agent = AlphaAgent(input_dim, n_assets, d_model)
        self.risk_agent = RiskAgent(n_assets, d_model)
        self.sentiment_agent = SentimentAgent(n_assets, d_model)
        self.execution_agent = ExecutionAgent(n_assets, d_model)
        
        # Aggregation network
        self.aggregator = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, n_assets)
        )
        
        # Confidence weighting
        self.confidence_net = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 4),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        state: MarketState,
        current_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Run multi-agent coordination loop.
        """
        # Phase 1: Individual agent analysis
        alpha_output = self.alpha_agent(state)
        risk_output = self.risk_agent(state)
        sentiment_output = self.sentiment_agent(state)
        
        # Phase 2: Message passing
        messages = [
            alpha_output['message'],
            risk_output['message'],
            sentiment_output['message']
        ]
        
        # Aggregate confidences
        confidences = torch.tensor([m.confidence for m in messages])
        
        # Phase 3: Generate target allocation
        alpha_scores = alpha_output['alpha_scores']
        risk_scores = risk_output['risk_scores']
        sentiment_signal = sentiment_output['sentiment_signal']
        
        # Combine signals with risk adjustment
        combined_signal = (
            alpha_scores * (1 - risk_scores * 0.5) + 
            0.2 * sentiment_signal
        )
        
        # Convert to weights (softmax for long-only)
        target_weights = F.softmax(combined_signal, dim=-1)
        
        # Apply risk constraints
        target_weights = self.risk_agent.apply_constraints(
            target_weights,
            risk_output['position_limits'],
            state.sector_ids
        )
        
        # Phase 4: Execution
        execution_output = self.execution_agent(target_weights, current_weights, state)
        
        return {
            'target_weights': target_weights,
            'alpha_scores': alpha_scores,
            'risk_scores': risk_scores,
            'sentiment_signal': sentiment_signal,
            'trades': execution_output['trades'],
            'transaction_costs': execution_output['transaction_costs'],
            'new_positions': execution_output['new_positions'],
            'agent_outputs': {
                'alpha': alpha_output,
                'risk': risk_output,
                'sentiment': sentiment_output,
                'execution': execution_output
            }
        }

