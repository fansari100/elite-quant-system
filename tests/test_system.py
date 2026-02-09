"""
Elite Quant System - Comprehensive Tests
=========================================
Unit and integration tests for all components.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSignatureTransformer:
    """Tests for Signature-Informed Transformer."""
    
    def test_model_creation(self):
        """Test model initialization."""
        from models.signature_transformer import SignatureInformedTransformer
        
        model = SignatureInformedTransformer(
            input_dim=18,
            n_assets=20,
            d_model=64,
            n_heads=4,
            n_layers=2
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
    
    def test_forward_pass(self):
        """Test forward pass."""
        from models.signature_transformer import SignatureInformedTransformer
        
        model = SignatureInformedTransformer(
            input_dim=18,
            n_assets=10,
            d_model=64,
            n_heads=4,
            n_layers=2
        )
        
        x = torch.randn(4, 10, 60, 18)
        output = model(x)
        
        assert 'weights' in output
        assert output['weights'].shape == (4, 10)
        assert torch.allclose(output['weights'].sum(dim=-1), torch.ones(4), atol=1e-5)
    
    def test_attention_output(self):
        """Test attention map output."""
        from models.signature_transformer import SignatureInformedTransformer
        
        model = SignatureInformedTransformer(
            input_dim=18,
            n_assets=10,
            d_model=64,
            n_heads=4,
            n_layers=2
        )
        
        x = torch.randn(2, 10, 60, 18)
        output = model(x, return_attention=True)
        
        assert 'attention_maps' in output
        assert len(output['attention_maps']) == 2  # n_layers


class TestTemporalFusionTransformer:
    """Tests for TFT model."""
    
    def test_tft_creation(self):
        """Test TFT initialization."""
        from models.temporal_fusion import TemporalFusionTransformer
        
        model = TemporalFusionTransformer(
            n_temporal_features=18,
            n_static_features=4,
            n_assets=10,
            d_model=64
        )
        
        assert model is not None
    
    def test_tft_forward(self):
        """Test TFT forward pass."""
        from models.temporal_fusion import TemporalFusionTransformer
        
        model = TemporalFusionTransformer(
            n_temporal_features=18,
            n_static_features=4,
            n_assets=10,
            d_model=64
        )
        
        temporal = torch.randn(4, 10, 60, 18)
        static = torch.randn(4, 10, 4)
        
        output = model(temporal, static)
        
        assert 'weights' in output
        assert 'quantile_predictions' in output
        assert 'temporal_attention' in output


class TestRegimeDetector:
    """Tests for regime detection."""
    
    def test_regime_detector(self):
        """Test regime detector."""
        from models.temporal_fusion import RegimeDetector
        
        detector = RegimeDetector(input_dim=18, hidden_dim=64, n_regimes=4)
        
        x = torch.randn(4, 60, 18)
        output = detector(x)
        
        assert 'regime_probs' in output
        assert output['regime_probs'].shape == (4, 4)
        assert torch.allclose(output['regime_probs'].sum(dim=-1), torch.ones(4), atol=1e-5)


class TestReinforcementLearning:
    """Tests for RL agents."""
    
    def test_ppo_agent(self):
        """Test PPO agent."""
        from models.reinforcement import PPOAgent
        
        agent = PPOAgent(state_dim=100, action_dim=10)
        
        state = torch.randn(4, 100)
        action = agent.get_action(state)
        
        assert action.shape == (4, 10)
        assert torch.allclose(action.sum(dim=-1), torch.ones(4), atol=1e-5)
    
    def test_sac_agent(self):
        """Test SAC agent."""
        from models.reinforcement import SACAgent
        
        agent = SACAgent(state_dim=100, action_dim=10)
        
        state = torch.randn(4, 100)
        action = agent.get_action(state)
        
        assert action.shape == (4, 10)


class TestConformalPrediction:
    """Tests for conformal prediction."""
    
    def test_cqr_creation(self):
        """Test CQR model creation."""
        from models.conformal import ConformizedQuantileRegression
        
        model = ConformizedQuantileRegression(input_dim=64, hidden_dim=32)
        assert model is not None
    
    def test_cqr_forward(self):
        """Test CQR forward pass."""
        from models.conformal import ConformizedQuantileRegression
        
        model = ConformizedQuantileRegression(input_dim=64, hidden_dim=32)
        
        x = torch.randn(10, 64)
        quantiles = model(x)
        
        assert quantiles.shape == (10, 3)  # 3 default quantiles
        # Check monotonicity
        assert torch.all(quantiles[:, 0] <= quantiles[:, 1])
        assert torch.all(quantiles[:, 1] <= quantiles[:, 2])


class TestMultiAgentSystem:
    """Tests for multi-agent coordination."""
    
    def test_alpha_agent(self):
        """Test alpha agent."""
        from agents.multi_agent import AlphaAgent, MarketState
        
        agent = AlphaAgent(input_dim=18, n_assets=10, d_model=64)
        
        state = MarketState(
            features=torch.randn(10, 60, 18),
            returns=torch.randn(10, 5),
            volatility=torch.randn(10),
            correlations=torch.randn(10, 10)
        )
        
        output = agent(state)
        
        assert 'alpha_scores' in output
        assert output['alpha_scores'].shape == (10,)
    
    def test_risk_agent(self):
        """Test risk agent."""
        from agents.multi_agent import RiskAgent, MarketState
        
        agent = RiskAgent(n_assets=10, d_model=64)
        
        state = MarketState(
            features=torch.randn(10, 60, 18),
            returns=torch.randn(10, 5),
            volatility=torch.randn(10),
            correlations=torch.randn(10, 10)
        )
        
        output = agent(state)
        
        assert 'risk_scores' in output
        assert 'position_limits' in output


class TestWalkForwardValidation:
    """Tests for walk-forward validation."""
    
    def test_rolling_splits(self):
        """Test rolling window splits."""
        from backtest.validation import WalkForwardValidator, WalkForwardConfig
        import pandas as pd
        
        config = WalkForwardConfig(
            train_window=100,
            test_window=20,
            step_size=20,
            n_splits=5
        )
        
        validator = WalkForwardValidator(config)
        
        # Create dummy data
        n_samples = 300
        data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=n_samples),
            'feature': np.random.randn(n_samples),
            'target': np.random.randn(n_samples)
        })
        
        splits = validator.create_splits(data)
        
        assert len(splits) > 0
        
        for train_idx, test_idx in splits:
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0
            # Test comes after train
            assert train_idx.max() < test_idx.min()


class TestBacktestEngine:
    """Tests for backtesting engine."""
    
    def test_backtest_from_arrays(self):
        """Test backtest from numpy arrays."""
        from backtest.engine import BacktestEngine
        
        engine = BacktestEngine(
            transaction_cost_bps=5.0,
            slippage_bps=2.0
        )
        
        n_days = 252
        n_assets = 10
        
        # Random weights and returns
        weights = np.random.dirichlet(np.ones(n_assets), n_days)
        returns = np.random.randn(n_days, n_assets) * 0.02
        tickers = [f'ASSET_{i}' for i in range(n_assets)]
        
        result = engine.run_from_arrays(weights, returns, tickers)
        
        assert result.metrics['sharpe_ratio'] is not None
        assert len(result.returns) == n_days


class TestDataLoader:
    """Tests for data loading."""
    
    def test_technical_indicators(self):
        """Test technical indicator computation."""
        from data.data_loader import TechnicalIndicators
        import pandas as pd
        
        n = 100
        df = pd.DataFrame({
            'open': np.random.randn(n).cumsum() + 100,
            'high': np.random.randn(n).cumsum() + 101,
            'low': np.random.randn(n).cumsum() + 99,
            'close': np.random.randn(n).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, n)
        })
        
        # Ensure high > low
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        result = TechnicalIndicators.compute_all(df)
        
        assert 'rsi' in result.columns
        assert 'macd' in result.columns
        assert 'bb_width' in result.columns


class TestConfig:
    """Tests for configuration."""
    
    def test_config_creation(self):
        """Test config creation."""
        from config import get_config
        
        config = get_config()
        
        assert config.data.sequence_length > 0
        assert config.model.d_model > 0
        assert config.training.batch_size > 0


# Integration test
class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_pipeline(self):
        """Test complete pipeline from data to prediction."""
        from models.signature_transformer import SignatureInformedTransformer
        from agents.multi_agent import MultiAgentCoordinator, MarketState
        
        # Create model
        n_assets = 10
        model = SignatureInformedTransformer(
            input_dim=18,
            n_assets=n_assets,
            d_model=64,
            n_heads=4,
            n_layers=2
        )
        
        # Create coordinator
        coordinator = MultiAgentCoordinator(
            input_dim=18,
            n_assets=n_assets,
            d_model=64
        )
        
        # Generate synthetic data
        features = torch.randn(4, n_assets, 60, 18)
        returns = torch.randn(4, n_assets, 5)
        
        # Model prediction
        model.eval()
        with torch.no_grad():
            output = model(features)
        
        assert output['weights'].shape == (4, n_assets)
        
        # Agent coordination
        state = MarketState(
            features=features[0],
            returns=returns[0, :, 0],
            volatility=torch.rand(n_assets),
            correlations=torch.eye(n_assets)
        )
        
        agent_output = coordinator(state)
        assert agent_output['target_weights'].shape == (n_assets,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

