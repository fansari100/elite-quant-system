"""
Data Loading and Feature Engineering
=====================================
Professional-grade data pipeline for quant trading
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import yfinance as yf
from pathlib import Path
import warnings
from functools import lru_cache
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class PathSignatureEngine:
    """
    GPU-accelerated path signature computation.
    Based on rough path theory - provides universal features for sequential data.
    """
    
    def __init__(self, depth: int = 4, augment_time: bool = True):
        self.depth = depth
        self.augment_time = augment_time
        
    def compute_signature(self, path: torch.Tensor) -> torch.Tensor:
        """
        Compute path signature up to specified depth.
        
        Args:
            path: (batch, seq_len, d) tensor
            
        Returns:
            signature: (batch, sig_dim) tensor
        """
        try:
            import signatory
            
            if self.augment_time:
                batch, seq_len, d = path.shape
                time = torch.linspace(0, 1, seq_len, device=path.device)
                time = time.unsqueeze(0).unsqueeze(-1).expand(batch, -1, 1)
                path = torch.cat([time, path], dim=-1)
            
            sig = signatory.signature(path, self.depth)
            return sig
            
        except ImportError:
            # Fallback: compute simplified signature terms
            return self._compute_fallback_signature(path)
    
    def _compute_fallback_signature(self, path: torch.Tensor) -> torch.Tensor:
        """Fallback signature computation without signatory."""
        batch, seq_len, d = path.shape
        
        # Compute increments
        increments = path[:, 1:] - path[:, :-1]
        
        # Level 1: sum of increments
        sig_1 = increments.sum(dim=1)  # (batch, d)
        
        # Level 2: areas (simplified)
        cumsum = torch.cumsum(increments, dim=1)
        sig_2_list = []
        for i in range(d):
            for j in range(d):
                area = (cumsum[:, :-1, i] * increments[:, 1:, j]).sum(dim=1)
                sig_2_list.append(area)
        sig_2 = torch.stack(sig_2_list, dim=1)  # (batch, d*d)
        
        return torch.cat([sig_1, sig_2], dim=1)


class TechnicalIndicators:
    """Technical indicator computation."""
    
    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators."""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Trend indicators
        df['sma_5'] = close.rolling(5).mean()
        df['sma_20'] = close.rolling(20).mean()
        df['sma_60'] = close.rolling(60).mean()
        df['ema_12'] = close.ewm(span=12).mean()
        df['ema_26'] = close.ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # Volatility
        df['atr'] = TechnicalIndicators._compute_atr(high, low, close, 14)
        df['volatility_20'] = close.pct_change().rolling(20).std() * np.sqrt(252)
        
        # Momentum
        df['momentum_5'] = close.pct_change(5)
        df['momentum_20'] = close.pct_change(20)
        df['momentum_60'] = close.pct_change(60)
        
        # Volume indicators
        df['volume_sma'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / (df['volume_sma'] + 1e-10)
        
        # Price position
        df['high_52w'] = high.rolling(252).max()
        df['low_52w'] = low.rolling(252).min()
        df['position_52w'] = (close - df['low_52w']) / (df['high_52w'] - df['low_52w'] + 1e-10)
        
        return df
    
    @staticmethod
    def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Compute Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()


class EquityDataLoader:
    """
    Professional equity data loader with caching.
    """
    
    FEATURE_COLUMNS = [
        'returns', 'log_returns', 'sma_5', 'sma_20', 'sma_60',
        'macd', 'macd_signal', 'macd_hist', 'rsi', 'bb_width', 'bb_position',
        'atr', 'volatility_20', 'momentum_5', 'momentum_20', 'momentum_60',
        'volume_ratio', 'position_52w'
    ]
    
    def __init__(self, config, cache_dir: Optional[Path] = None):
        self.config = config
        self.cache_dir = cache_dir or Path("data_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.signature_engine = PathSignatureEngine(
            depth=config.data.signature_depth,
            augment_time=True
        )
    
    def download_data(self, tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        """Download OHLCV data for all tickers."""
        logger.info(f"Downloading data for {len(tickers)} tickers...")
        
        all_data = {}
        for ticker in tickers:
            cache_file = self.cache_dir / f"{ticker}_{start}_{end}.parquet"
            
            if cache_file.exists():
                df = pd.read_parquet(cache_file)
            else:
                try:
                    df = yf.download(ticker, start=start, end=end, progress=False)
                    if len(df) > 0:
                        df.columns = [c.lower() for c in df.columns]
                        df.to_parquet(cache_file)
                except Exception as e:
                    logger.warning(f"Failed to download {ticker}: {e}")
                    continue
            
            if len(df) > 100:  # Minimum data requirement
                all_data[ticker] = df
        
        logger.info(f"Successfully loaded {len(all_data)} tickers")
        return all_data
    
    def prepare_features(self, data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrices for all assets.
        
        Returns:
            features: (n_samples, n_assets, seq_len, n_features)
            returns: (n_samples, n_assets, horizon)
            dates: list of dates
        """
        # Get common dates
        all_dates = None
        for ticker, df in data.items():
            dates = set(df.index)
            all_dates = dates if all_dates is None else all_dates & dates
        
        common_dates = sorted(list(all_dates))
        tickers = list(data.keys())
        n_assets = len(tickers)
        seq_len = self.config.data.sequence_length
        horizon = self.config.data.prediction_horizon
        
        # Process each ticker
        processed = {}
        for ticker in tickers:
            df = data[ticker].loc[common_dates].copy()
            
            # Compute returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Add technical indicators
            df = TechnicalIndicators.compute_all(df)
            
            # Forward fill and normalize
            df = df.fillna(method='ffill').fillna(0)
            
            processed[ticker] = df
        
        # Create sequences
        n_samples = len(common_dates) - seq_len - horizon + 1
        
        features = np.zeros((n_samples, n_assets, seq_len, len(self.FEATURE_COLUMNS)))
        future_returns = np.zeros((n_samples, n_assets, horizon))
        sample_dates = []
        
        for i in range(n_samples):
            sample_dates.append(common_dates[i + seq_len - 1])
            
            for j, ticker in enumerate(tickers):
                df = processed[ticker]
                
                # Extract features
                for k, col in enumerate(self.FEATURE_COLUMNS):
                    if col in df.columns:
                        features[i, j, :, k] = df[col].iloc[i:i+seq_len].values
                
                # Future returns
                future_prices = df['close'].iloc[i+seq_len:i+seq_len+horizon].values
                current_price = df['close'].iloc[i+seq_len-1]
                future_returns[i, j] = future_prices / current_price - 1
        
        # Normalize features
        features = self._normalize_features(features)
        
        return features, future_returns, sample_dates, tickers
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Robust normalization of features."""
        # Per-feature normalization using rolling statistics
        n_samples, n_assets, seq_len, n_features = features.shape
        
        for k in range(n_features):
            feature_data = features[:, :, :, k].flatten()
            
            # Robust scaling using median and IQR
            median = np.median(feature_data)
            q75, q25 = np.percentile(feature_data, [75, 25])
            iqr = q75 - q25
            
            if iqr > 1e-10:
                features[:, :, :, k] = (features[:, :, :, k] - median) / iqr
            
            # Clip outliers
            features[:, :, :, k] = np.clip(features[:, :, :, k], -5, 5)
        
        return features


class QuantDataset(Dataset):
    """PyTorch dataset for quant trading."""
    
    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        signature_engine: Optional[PathSignatureEngine] = None
    ):
        self.features = torch.FloatTensor(features)
        self.returns = torch.FloatTensor(returns)
        self.signature_engine = signature_engine
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        features = self.features[idx]  # (n_assets, seq_len, n_features)
        returns = self.returns[idx]    # (n_assets, horizon)
        
        return {
            'features': features,
            'returns': returns,
            'target_returns': returns.mean(dim=-1)  # Average over horizon
        }


def create_dataloaders(
    config,
    batch_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create train/val/test dataloaders.
    """
    loader = EquityDataLoader(config)
    
    # Download all data
    all_data = loader.download_data(
        config.data.universe,
        config.data.train_start,
        config.data.test_end
    )
    
    # Prepare features
    features, returns, dates, tickers = loader.prepare_features(all_data)
    
    # Split by date
    dates = pd.to_datetime(dates)
    train_mask = dates < config.data.val_start
    val_mask = (dates >= config.data.val_start) & (dates < config.data.test_start)
    test_mask = dates >= config.data.test_start
    
    train_features = features[train_mask]
    train_returns = returns[train_mask]
    val_features = features[val_mask]
    val_returns = returns[val_mask]
    test_features = features[test_mask]
    test_returns = returns[test_mask]
    
    logger.info(f"Train samples: {len(train_features)}, Val: {len(val_features)}, Test: {len(test_features)}")
    
    # Create datasets
    train_dataset = QuantDataset(train_features, train_returns)
    val_dataset = QuantDataset(val_features, val_returns)
    test_dataset = QuantDataset(test_features, test_returns)
    
    bs = batch_size or config.training.batch_size
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, tickers

