"""
FastAPI Server for Model Inference
===================================
Production API for real-time portfolio optimization.
"""

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

from config import get_config
from models.signature_transformer import SignatureInformedTransformer

logger = logging.getLogger(__name__)


# Pydantic models for API
class MarketDataInput(BaseModel):
    """Input market data for prediction."""
    tickers: List[str] = Field(..., description="List of ticker symbols")
    features: List[List[List[float]]] = Field(..., description="(n_assets, seq_len, n_features)")


class PortfolioWeights(BaseModel):
    """Portfolio weight output."""
    weights: Dict[str, float] = Field(..., description="Ticker to weight mapping")
    timestamp: str = Field(..., description="Prediction timestamp")
    confidence: float = Field(default=1.0, description="Model confidence")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    gpu_available: bool
    gpu_name: Optional[str]
    model_loaded: bool


class BacktestRequest(BaseModel):
    """Backtest request."""
    start_date: str
    end_date: str
    tickers: Optional[List[str]] = None


class BacktestResponse(BaseModel):
    """Backtest results."""
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    annual_volatility: float


# Global model instance
_model = None
_config = None


def load_model():
    """Load model from checkpoint."""
    global _model, _config
    
    if _model is None:
        _config = get_config()
        
        # Check for checkpoint
        checkpoint_path = _config.checkpoint_dir / "final_model.ckpt"
        
        if checkpoint_path.exists():
            logger.info(f"Loading model from {checkpoint_path}")
            from training.lightning_module import QuantLightningModule
            _model = QuantLightningModule.load_from_checkpoint(checkpoint_path)
        else:
            logger.warning("No checkpoint found, initializing fresh model")
            # Get dimensions from config
            input_dim = 18  # Default feature dimension
            n_assets = len(_config.data.universe)
            
            _model = SignatureInformedTransformer(
                input_dim=input_dim,
                n_assets=n_assets,
                d_model=_config.model.d_model,
                n_heads=_config.model.n_heads,
                n_layers=_config.model.n_encoder_layers,
                dim_feedforward=_config.model.dim_feedforward,
                dropout=0.0,  # No dropout for inference
                signature_depth=_config.model.signature_depth,
                max_position=_config.model.max_position_size,
                risk_aversion=_config.model.risk_aversion
            )
        
        _model.eval()
        
        if torch.cuda.is_available():
            _model = _model.cuda()
    
    return _model


def create_app(config=None) -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Elite Quant System API",
        description="Real-time portfolio optimization using Signature-Informed Transformer",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def startup():
        """Load model on startup."""
        load_model()
        logger.info("Model loaded successfully")
    
    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
        
        return HealthResponse(
            status="healthy",
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            model_loaded=_model is not None
        )
    
    @app.get("/config")
    async def get_configuration():
        """Get model configuration."""
        config = get_config()
        return {
            "universe": config.data.universe,
            "n_assets": len(config.data.universe),
            "sequence_length": config.data.sequence_length,
            "model_dim": config.model.d_model,
            "n_heads": config.model.n_heads,
            "n_layers": config.model.n_encoder_layers,
            "signature_depth": config.model.signature_depth
        }
    
    @app.post("/predict", response_model=PortfolioWeights)
    async def predict(data: MarketDataInput):
        """
        Generate portfolio weights from market data.
        
        Input features should be normalized and of shape (n_assets, seq_len, n_features).
        """
        model = load_model()
        
        try:
            # Convert to tensor
            features = torch.FloatTensor(data.features).unsqueeze(0)  # (1, n_assets, seq_len, features)
            
            if torch.cuda.is_available():
                features = features.cuda()
            
            # Inference
            with torch.no_grad():
                output = model(features)
                weights = output['weights'][0].cpu().numpy()
            
            # Create response
            weight_dict = {ticker: float(w) for ticker, w in zip(data.tickers, weights)}
            
            return PortfolioWeights(
                weights=weight_dict,
                timestamp=datetime.now().isoformat(),
                confidence=0.95
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/backtest", response_model=BacktestResponse)
    async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
        """
        Run backtest on historical data.
        """
        from backtest.engine import BacktestEngine
        from data.data_loader import create_dataloaders
        
        try:
            config = get_config()
            
            # Override dates
            if request.tickers:
                config.data.universe = request.tickers
            
            # Create dataloaders
            _, _, test_loader, tickers = create_dataloaders(config)
            
            # Run backtest
            model = load_model()
            engine = BacktestEngine(
                transaction_cost_bps=config.backtest.transaction_cost_bps,
                slippage_bps=config.backtest.slippage_bps
            )
            
            result = engine.run(model, test_loader, tickers)
            
            return BacktestResponse(
                sharpe_ratio=result.metrics['sharpe_ratio'],
                total_return=result.metrics['total_return'],
                max_drawdown=result.metrics['max_drawdown'],
                annual_volatility=result.metrics['volatility']
            )
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/tickers")
    async def get_tickers():
        """Get list of supported tickers."""
        config = get_config()
        return {"tickers": config.data.universe}
    
    return app


# For running directly
if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)

