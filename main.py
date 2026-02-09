#!/usr/bin/env python3
"""
Elite Quant System - Main Entry Point
======================================
Signature-Informed Multi-Agent Quantitative Trading System
Optimized for NVIDIA H200 on Lightning.ai

Usage:
    python main.py train          # Train model
    python main.py backtest       # Run backtest
    python main.py demo           # Quick demonstration
    python main.py serve          # Start API server
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import lightning as L
from lightning.pytorch import Trainer

from config import get_config, Config
from data.data_loader import create_dataloaders, EquityDataLoader
from models.signature_transformer import SignatureInformedTransformer
from agents.multi_agent import MultiAgentCoordinator, MarketState
from training.lightning_module import QuantLightningModule, get_callbacks, get_logger
from backtest.engine import BacktestEngine, BacktestVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console = Console()


def print_banner():
    """Print system banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ███████╗██╗     ██╗████████╗███████╗                       ║
    ║   ██╔════╝██║     ██║╚══██╔══╝██╔════╝                       ║
    ║   █████╗  ██║     ██║   ██║   █████╗                         ║
    ║   ██╔══╝  ██║     ██║   ██║   ██╔══╝                         ║
    ║   ███████╗███████╗██║   ██║   ███████╗                       ║
    ║   ╚══════╝╚══════╝╚═╝   ╚═╝   ╚══════╝                       ║
    ║                                                               ║
    ║   QUANT SYSTEM 2026 - Signature-Informed Transformer         ║
    ║   Optimized for NVIDIA H200 • Lightning.ai                   ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")


def check_gpu():
    """Check GPU availability and print info."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(f"\n[green]✓ GPU Available: {gpu_name}")
        console.print(f"[green]  Memory: {gpu_memory:.1f} GB")
        
        # Check for H100/H200
        if 'H100' in gpu_name or 'H200' in gpu_name:
            console.print("[green]  ✓ High-performance Hopper GPU detected!")
            console.print("[green]  ✓ Enabling BF16 and Flash Attention optimizations")
    else:
        console.print("\n[yellow]⚠ No GPU detected. Running on CPU (slower)")
    
    return torch.cuda.is_available()


def train(config: Config):
    """Train the model."""
    console.print(Panel.fit("🚀 Starting Training Pipeline", style="bold green"))
    
    # Set seeds for reproducibility
    L.seed_everything(config.seed)
    
    # Create dataloaders
    console.print("\n[cyan]Loading and preprocessing data...")
    train_loader, val_loader, test_loader, tickers = create_dataloaders(config)
    
    console.print(f"[green]✓ Loaded {len(tickers)} assets")
    console.print(f"[green]✓ Train batches: {len(train_loader)}")
    console.print(f"[green]✓ Val batches: {len(val_loader)}")
    console.print(f"[green]✓ Test batches: {len(test_loader)}")
    
    # Get input dimension from data
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch['features'].shape[-1]
    n_assets = sample_batch['features'].shape[1]
    
    console.print(f"\n[cyan]Model configuration:")
    console.print(f"  • Input dimension: {input_dim}")
    console.print(f"  • Number of assets: {n_assets}")
    console.print(f"  • Model dimension: {config.model.d_model}")
    console.print(f"  • Attention heads: {config.model.n_heads}")
    console.print(f"  • Encoder layers: {config.model.n_encoder_layers}")
    console.print(f"  • Signature depth: {config.model.signature_depth}")
    
    # Create Lightning module
    model = QuantLightningModule(
        input_dim=input_dim,
        n_assets=n_assets,
        config=config
    )
    
    # Create trainer
    trainer = Trainer(
        max_epochs=config.training.max_epochs,
        accelerator='auto',
        devices=config.training.devices,
        strategy=config.training.strategy,
        precision=config.training.precision,
        callbacks=get_callbacks(config),
        logger=get_logger(config),
        log_every_n_steps=config.training.log_every_n_steps,
        val_check_interval=config.training.val_check_interval,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        gradient_clip_val=1.0,
        deterministic=config.deterministic
    )
    
    # Compile model for speed (PyTorch 2.0+)
    if config.training.use_torch_compile and hasattr(torch, 'compile'):
        console.print("\n[cyan]Compiling model with torch.compile...")
        model.sit_model = torch.compile(model.sit_model)
    
    # Train
    console.print("\n[green]Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    console.print("\n[cyan]Running final test...")
    trainer.test(model, test_loader)
    
    # Save final model
    final_path = config.checkpoint_dir / "final_model.ckpt"
    trainer.save_checkpoint(final_path)
    console.print(f"\n[green]✓ Model saved to {final_path}")
    
    return model, test_loader, tickers


def backtest(config: Config, model=None, test_loader=None, tickers=None):
    """Run backtest on trained model."""
    console.print(Panel.fit("📊 Running Backtest", style="bold blue"))
    
    # Load model if not provided
    if model is None:
        checkpoint_path = config.checkpoint_dir / "final_model.ckpt"
        if checkpoint_path.exists():
            console.print(f"\n[cyan]Loading model from {checkpoint_path}")
            model = QuantLightningModule.load_from_checkpoint(checkpoint_path)
        else:
            console.print("[red]No trained model found. Please train first.")
            return
    
    # Load test data if not provided
    if test_loader is None or tickers is None:
        console.print("\n[cyan]Loading test data...")
        _, _, test_loader, tickers = create_dataloaders(config)
    
    # Create backtest engine
    engine = BacktestEngine(
        transaction_cost_bps=config.backtest.transaction_cost_bps,
        slippage_bps=config.backtest.slippage_bps,
        risk_free_rate=config.backtest.risk_free_rate
    )
    
    # Run backtest
    console.print("\n[cyan]Running simulation...")
    result = engine.run(model, test_loader, tickers)
    
    # Print results
    console.print(f"\n{result.summary()}")
    
    # Generate plots
    console.print("\n[cyan]Generating visualizations...")
    
    fig1 = BacktestVisualizer.plot_equity_curve(result)
    fig1.savefig(config.log_dir / 'equity_curve.png', dpi=150, bbox_inches='tight')
    
    fig2 = BacktestVisualizer.plot_weights(result)
    fig2.savefig(config.log_dir / 'weight_evolution.png', dpi=150, bbox_inches='tight')
    
    fig3 = BacktestVisualizer.plot_monthly_returns(result)
    fig3.savefig(config.log_dir / 'monthly_returns.png', dpi=150, bbox_inches='tight')
    
    console.print(f"[green]✓ Plots saved to {config.log_dir}")
    
    return result


def demo(config: Config):
    """Run a quick demonstration with synthetic data."""
    console.print(Panel.fit("🎯 Quick Demonstration Mode", style="bold yellow"))
    
    # Create synthetic data
    console.print("\n[cyan]Generating synthetic market data...")
    
    n_samples = 500
    n_assets = 20
    seq_len = 60
    n_features = 18
    horizon = 5
    
    np.random.seed(config.seed)
    
    # Generate features
    features = np.random.randn(n_samples, n_assets, seq_len, n_features) * 0.1
    
    # Generate correlated returns
    factor_returns = np.random.randn(n_samples, 3) * 0.02
    asset_loadings = np.random.randn(n_assets, 3)
    idiosyncratic = np.random.randn(n_samples, n_assets) * 0.01
    returns = factor_returns @ asset_loadings.T + idiosyncratic
    
    # Create tensors
    features_tensor = torch.FloatTensor(features)
    returns_tensor = torch.FloatTensor(returns)
    
    console.print(f"[green]✓ Generated {n_samples} samples")
    console.print(f"[green]✓ {n_assets} assets, {seq_len} timesteps, {n_features} features")
    
    # Create model
    console.print("\n[cyan]Initializing Signature-Informed Transformer...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = SignatureInformedTransformer(
        input_dim=n_features,
        n_assets=n_assets,
        d_model=128,
        n_heads=4,
        n_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        signature_depth=3,
        max_position=0.15,
        risk_aversion=1.0
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"[green]✓ Model initialized: {n_params:,} parameters")
    
    # Quick training loop
    console.print("\n[cyan]Running quick training (10 epochs)...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    
    batch_size = 32
    n_batches = n_samples // batch_size
    
    for epoch in range(10):
        epoch_loss = 0.0
        epoch_returns = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_features = features_tensor[start_idx:end_idx].to(device)
            batch_returns = returns_tensor[start_idx:end_idx].to(device)
            
            optimizer.zero_grad()
            
            output = model(batch_features)
            weights = output['weights']
            
            # Compute loss (maximize returns, minimize variance)
            portfolio_returns = (weights * batch_returns).sum(dim=-1)
            mean_return = portfolio_returns.mean()
            variance = portfolio_returns.var()
            loss = -(mean_return - 0.5 * variance)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_returns.extend(portfolio_returns.detach().cpu().numpy().tolist())
        
        avg_loss = epoch_loss / n_batches
        avg_return = np.mean(epoch_returns)
        sharpe = np.mean(epoch_returns) / (np.std(epoch_returns) + 1e-8) * np.sqrt(252)
        
        console.print(f"  Epoch {epoch+1}/10 | Loss: {avg_loss:.4f} | Sharpe: {sharpe:.2f}")
    
    # Test inference
    console.print("\n[cyan]Testing inference...")
    model.eval()
    
    with torch.no_grad():
        test_features = features_tensor[:1].to(device)
        output = model(test_features, return_attention=True)
        weights = output['weights'][0]
        
        console.print(f"\n[green]Sample portfolio allocation:")
        sorted_weights = sorted(zip(range(n_assets), weights.cpu().numpy()), 
                               key=lambda x: x[1], reverse=True)
        for asset_id, weight in sorted_weights[:5]:
            console.print(f"  Asset {asset_id}: {weight:.2%}")
    
    console.print("\n[green]✓ Demonstration complete!")
    console.print("[cyan]Run 'python main.py train' with real data for full training.")


def serve(config: Config):
    """Start FastAPI server for model inference."""
    console.print(Panel.fit("🌐 Starting API Server", style="bold magenta"))
    
    try:
        from api import create_app
        import uvicorn
        
        app = create_app(config)
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError:
        console.print("[yellow]API module not found. Creating basic server...")
        
        # Create basic FastAPI app inline
        from fastapi import FastAPI
        from pydantic import BaseModel
        import uvicorn
        
        app = FastAPI(title="Elite Quant System API")
        
        @app.get("/health")
        def health():
            return {"status": "healthy", "gpu": torch.cuda.is_available()}
        
        @app.get("/config")
        def get_config_endpoint():
            return {
                "n_assets": len(config.data.universe),
                "model_dim": config.model.d_model,
                "n_heads": config.model.n_heads
            }
        
        console.print("[green]Starting server on http://0.0.0.0:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000)


def main():
    """Main entry point."""
    import typer
    
    app = typer.Typer()
    
    @app.command()
    def train_cmd():
        """Train the quantitative model."""
        print_banner()
        check_gpu()
        config = get_config()
        model, test_loader, tickers = train(config)
        backtest(config, model, test_loader, tickers)
    
    @app.command()
    def backtest_cmd():
        """Run backtest on trained model."""
        print_banner()
        check_gpu()
        config = get_config()
        backtest(config)
    
    @app.command()
    def demo_cmd():
        """Run quick demonstration."""
        print_banner()
        check_gpu()
        config = get_config()
        demo(config)
    
    @app.command()
    def serve_cmd():
        """Start API server."""
        print_banner()
        check_gpu()
        config = get_config()
        serve(config)
    
    # Parse arguments
    if len(sys.argv) < 2:
        # Default to demo if no command specified
        print_banner()
        check_gpu()
        config = get_config()
        demo(config)
    else:
        app()


if __name__ == "__main__":
    main()

