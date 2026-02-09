"""
Elite Quant System Configuration
================================
Optimized for NVIDIA H200 on Lightning.ai
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import torch


@dataclass
class DataConfig:
    """Data configuration."""
    # Data sources
    universe: List[str] = field(default_factory=lambda: [
        # S&P 100 representative sample
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
        "UNH", "JNJ", "JPM", "V", "PG", "XOM", "MA", "HD", "CVX", "MRK",
        "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO",
        "ACN", "TMO", "ABT", "DHR", "VZ", "ADBE", "CRM", "NKE", "CMCSA",
        "PFE", "INTC", "ORCL", "TXN", "AMD", "QCOM", "HON", "UPS", "PM",
        "LOW", "BA", "CAT", "GE", "IBM"
    ])
    
    # Time parameters
    lookback_days: int = 252  # 1 year of trading days
    sequence_length: int = 60  # 60-day sequences
    prediction_horizon: int = 5  # 5-day forward returns
    
    # Data splits
    train_start: str = "2015-01-01"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2024-12-31"
    
    # Feature engineering
    use_technical_indicators: bool = True
    use_path_signatures: bool = True
    signature_depth: int = 4
    
    # Preprocessing
    normalize_method: str = "robust"  # robust, zscore, minmax
    handle_missing: str = "forward_fill"


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Signature-Informed Transformer
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 6
    n_decoder_layers: int = 2
    dim_feedforward: int = 1024
    dropout: float = 0.1
    
    # Path Signature
    signature_depth: int = 4
    signature_augment_time: bool = True
    use_logsignature: bool = True
    
    # Portfolio Head
    use_attention_pooling: bool = True
    portfolio_hidden_dim: int = 512
    
    # Risk parameters
    risk_aversion: float = 1.0
    max_position_size: float = 0.10  # 10% max per asset
    target_volatility: float = 0.15  # 15% annualized
    
    # Activation
    activation: str = "gelu"


@dataclass
class AgentConfig:
    """Multi-agent system configuration."""
    # Agent types
    enable_alpha_agent: bool = True
    enable_risk_agent: bool = True
    enable_sentiment_agent: bool = True
    enable_execution_agent: bool = True
    
    # Alpha agent
    n_factors: int = 50
    factor_decay: float = 0.95
    
    # Risk agent
    var_confidence: float = 0.99
    max_drawdown_limit: float = 0.10
    position_limit: float = 0.10
    sector_limit: float = 0.30
    
    # Sentiment agent (LLM)
    sentiment_model: str = "ProsusAI/finbert"
    max_news_per_day: int = 100
    sentiment_lookback_days: int = 7


@dataclass
class TrainingConfig:
    """Training configuration optimized for H200."""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    max_epochs: int = 100
    patience: int = 15
    
    # Batch sizes (H200 optimized - 141GB HBM3e)
    batch_size: int = 256
    accumulate_grad_batches: int = 1
    
    # Mixed precision (H200 Transformer Engine)
    precision: str = "bf16-mixed"  # or "16-mixed", "32"
    use_torch_compile: bool = True
    
    # Distributed training
    strategy: str = "auto"  # ddp, fsdp, deepspeed
    num_nodes: int = 1
    devices: int = -1  # Use all available GPUs
    
    # Checkpointing
    save_top_k: int = 3
    checkpoint_every_n_epochs: int = 1
    
    # Logging
    log_every_n_steps: int = 50
    val_check_interval: float = 0.25  # Validate 4x per epoch


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    # Execution
    transaction_cost_bps: float = 5.0  # 5 basis points
    slippage_bps: float = 2.0
    
    # Rebalancing
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    min_trade_size: float = 0.001  # 0.1% minimum trade
    
    # Risk management
    use_stop_loss: bool = True
    stop_loss_pct: float = 0.02  # 2% daily stop loss
    use_take_profit: bool = False
    
    # Position sizing
    sizing_method: str = "volatility_target"  # equal, volatility_target, kelly
    
    # Metrics
    risk_free_rate: float = 0.05  # 5% annual


@dataclass
class Config:
    """Master configuration."""
    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    
    # Paths
    project_root: Path = Path(__file__).parent
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data_cache")
    checkpoint_dir: Path = field(default_factory=lambda: Path(__file__).parent / "checkpoints")
    log_dir: Path = field(default_factory=lambda: Path(__file__).parent / "logs")
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Create directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """Get default configuration."""
    return Config()

