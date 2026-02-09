#!/bin/bash
# =============================================================================
# Elite Quant System - Lightning.ai Setup Script
# =============================================================================
# Run this script after creating a new Lightning.ai Studio with H200 GPU
#
# Usage:
#   chmod +x setup_lightning.sh
#   ./setup_lightning.sh
# =============================================================================

set -e

echo "=============================================="
echo "  Elite Quant System - Lightning.ai Setup"
echo "=============================================="

# Check GPU
echo ""
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✓ GPU detected"
else
    echo "⚠ No GPU detected - will run on CPU"
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA (Lightning.ai should have this, but ensure latest)
echo ""
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install main dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Install signatory for path signatures (may need special handling)
echo ""
echo "Installing signatory for path signatures..."
pip install signatory || echo "⚠ Signatory install failed - will use fallback implementation"

# Create directories
echo ""
echo "Creating directories..."
mkdir -p checkpoints logs data_cache

# Download sample data
echo ""
echo "Testing data download..."
python -c "
import yfinance as yf
import pandas as pd
test = yf.download('AAPL', period='5d', progress=False)
if len(test) > 0:
    print('✓ Yahoo Finance data access confirmed')
else:
    print('⚠ Could not fetch data from Yahoo Finance')
"

# Test PyTorch
echo ""
echo "Testing PyTorch..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Test Lightning
echo ""
echo "Testing Lightning..."
python -c "
import lightning as L
print(f'Lightning version: {L.__version__}')
"

# Run quick validation
echo ""
echo "Running system validation..."
python -c "
from config import get_config
from models.signature_transformer import SignatureInformedTransformer
import torch

config = get_config()
model = SignatureInformedTransformer(
    input_dim=18,
    n_assets=10,
    d_model=64,
    n_heads=4,
    n_layers=2,
    dim_feedforward=128
)

# Test forward pass
x = torch.randn(2, 10, 60, 18)
output = model(x)
print(f'✓ Model forward pass successful')
print(f'  Output weights shape: {output[\"weights\"].shape}')
"

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Quick start commands:"
echo "  python main.py demo      # Run demo with synthetic data"
echo "  python main.py train     # Full training pipeline"
echo "  python main.py backtest  # Backtest trained model"
echo "  python main.py serve     # Start API server"
echo ""

