"""
Elite Quant System 2026
========================
Signature-Informed Multi-Agent Quantitative Trading System
Optimized for NVIDIA H200 on Lightning.ai

Modules:
    - config: Configuration management
    - data: Data loading and feature engineering
    - models: Neural network architectures (SIT)
    - agents: Multi-agent coordination system
    - training: PyTorch Lightning training
    - backtest: Backtesting engine
    - api: FastAPI server
"""

__version__ = "1.0.0"
__author__ = "Elite Quant Team"

from .config import Config, get_config

__all__ = [
    'Config',
    'get_config',
    '__version__'
]

