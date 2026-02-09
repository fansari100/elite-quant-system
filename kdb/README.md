# KDB+/Q Time Series Database Layer

## Overview

**KDB+ with Q** is the industry standard for quantitative finance time series data. It's used by:
- Citadel
- Two Sigma
- Jane Street
- DE Shaw
- Goldman Sachs
- Morgan Stanley
- JPMorgan

## Why KDB+?

| Feature | Performance |
|---------|-------------|
| Query Speed | Billions of rows in milliseconds |
| Storage | 10x compression vs SQL |
| In-Memory | Tick data in RAM |
| Column-Oriented | Optimized for time series |
| Vectorized | Array operations on all data |

## Installation

### Option 1: KX Personal License (Free for learning)
```bash
# Download from kx.com
# Get personal/evaluation license
```

### Option 2: Open-source q (limited)
```bash
# Community edition available at code.kx.com
```

## Usage

### Start the server
```bash
q market_data.q
```

### Connect from Python
```python
# Using qPython library
from qpython import qconnection

q = qconnection.QConnection('localhost', 5001)
q.open()

# Get features for ML model
features = q.sync('getFeatures', 'AAPL', 252)

# Execute trade
order_id = q.sync('executeTrade', 'AAPL', 'buy', 100.0, 150.50)

q.close()
```

## Key Functions

### Data Ingestion
- `insertTrade` - Add single trade tick
- `insertQuote` - Add single quote tick
- `bulkInsertTrades` - Batch insert from Python

### Aggregation
- `aggTo1mBars` - Trades → 1-minute bars
- `aggToDailyBars` - Trades → Daily OHLCV

### Technical Indicators
- `sma` / `ema` - Moving averages
- `rsi` - Relative Strength Index
- `macd` - MACD indicator
- `bollinger` - Bollinger Bands
- `atr` - Average True Range

### Alpha Factors
- `momentum` - Price momentum
- `realized_vol` - Realized volatility
- `alphaSignal` - Combined factor signal

### Risk Analytics
- `varHist` / `cvarHist` - Value at Risk
- `maxDrawdown` - Maximum drawdown
- `sharpe` / `calmar` - Risk-adjusted returns

## Schema

### trade
| Column | Type | Description |
|--------|------|-------------|
| time | timestamp | Nanosecond precision |
| sym | symbol | Ticker symbol |
| price | float | Trade price |
| size | long | Trade size |
| exchange | symbol | Exchange code |

### quote
| Column | Type | Description |
|--------|------|-------------|
| time | timestamp | Nanosecond precision |
| sym | symbol | Ticker symbol |
| bid/ask | float | Prices |
| bidsize/asksize | long | Sizes |

### daily
| Column | Type | Description |
|--------|------|-------------|
| date | date | Trading date |
| sym | symbol | Ticker symbol |
| open/high/low/close | float | OHLC prices |
| volume | long | Daily volume |
| vwap | float | Volume-weighted avg |

## Performance Tips

1. **Use symbols, not strings**: `\`AAPL` not `"AAPL"`
2. **Vectorize operations**: Avoid loops
3. **Pre-sort data by time**: Enables binary search
4. **Use attributes**: `` `s#`` for sorted, `` `p#`` for partitioned

