# Elite Quant System - Feature Store Configuration
# Two Sigma "Data as Code" approach with Feast
# Manages ML features with versioning, point-in-time correctness, and serving

from datetime import timedelta
from feast import Entity, Feature, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Float64, Int64, String

# =============================================================================
# ENTITIES - Core business objects
# =============================================================================

# Stock entity
stock = Entity(
    name="stock",
    join_keys=["symbol"],
    description="Stock ticker symbol (e.g., AAPL, GOOG)",
)

# Portfolio entity
portfolio = Entity(
    name="portfolio",
    join_keys=["portfolio_id"],
    description="Portfolio identifier for multi-strategy systems",
)

# =============================================================================
# DATA SOURCES - Raw data locations
# =============================================================================

# Market data source (Parquet files from KDB+ export)
market_data_source = FileSource(
    name="market_data_source",
    path="/data/features/market_data.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Technical indicators source
technical_indicators_source = FileSource(
    name="technical_indicators_source",
    path="/data/features/technical_indicators.parquet",
    timestamp_field="event_timestamp",
)

# Alternative data source (sentiment, satellite, etc.)
alternative_data_source = FileSource(
    name="alternative_data_source",
    path="/data/features/alternative_data.parquet",
    timestamp_field="event_timestamp",
)

# Factor scores source
factor_scores_source = FileSource(
    name="factor_scores_source",
    path="/data/features/factor_scores.parquet",
    timestamp_field="event_timestamp",
)

# Order flow analytics source
order_flow_source = FileSource(
    name="order_flow_source",
    path="/data/features/order_flow.parquet",
    timestamp_field="event_timestamp",
)

# =============================================================================
# FEATURE VIEWS - Feature definitions with point-in-time correctness
# =============================================================================

# Market Data Features
market_data_fv = FeatureView(
    name="market_data_features",
    entities=[stock],
    ttl=timedelta(days=1),
    schema=[
        Field(name="open", dtype=Float64),
        Field(name="high", dtype=Float64),
        Field(name="low", dtype=Float64),
        Field(name="close", dtype=Float64),
        Field(name="volume", dtype=Float64),
        Field(name="vwap", dtype=Float64),
        Field(name="bid", dtype=Float64),
        Field(name="ask", dtype=Float64),
        Field(name="spread", dtype=Float64),
        Field(name="returns_1d", dtype=Float64),
        Field(name="returns_5d", dtype=Float64),
        Field(name="returns_21d", dtype=Float64),
        Field(name="log_volume", dtype=Float64),
        Field(name="dollar_volume", dtype=Float64),
    ],
    source=market_data_source,
    online=True,
    tags={"team": "alpha", "tier": "critical"},
)

# Technical Indicator Features
technical_indicators_fv = FeatureView(
    name="technical_indicators",
    entities=[stock],
    ttl=timedelta(days=1),
    schema=[
        # Momentum indicators
        Field(name="rsi_14", dtype=Float64),
        Field(name="rsi_7", dtype=Float64),
        Field(name="macd", dtype=Float64),
        Field(name="macd_signal", dtype=Float64),
        Field(name="macd_hist", dtype=Float64),
        Field(name="stoch_k", dtype=Float64),
        Field(name="stoch_d", dtype=Float64),
        Field(name="williams_r", dtype=Float64),
        Field(name="roc_10", dtype=Float64),
        Field(name="momentum_10", dtype=Float64),
        
        # Trend indicators
        Field(name="sma_20", dtype=Float64),
        Field(name="sma_50", dtype=Float64),
        Field(name="sma_200", dtype=Float64),
        Field(name="ema_12", dtype=Float64),
        Field(name="ema_26", dtype=Float64),
        Field(name="adx", dtype=Float64),
        Field(name="plus_di", dtype=Float64),
        Field(name="minus_di", dtype=Float64),
        
        # Volatility indicators
        Field(name="atr_14", dtype=Float64),
        Field(name="bollinger_upper", dtype=Float64),
        Field(name="bollinger_middle", dtype=Float64),
        Field(name="bollinger_lower", dtype=Float64),
        Field(name="bollinger_width", dtype=Float64),
        Field(name="keltner_upper", dtype=Float64),
        Field(name="keltner_lower", dtype=Float64),
        Field(name="realized_vol_21", dtype=Float64),
        Field(name="realized_vol_63", dtype=Float64),
        Field(name="garman_klass_vol", dtype=Float64),
        Field(name="parkinson_vol", dtype=Float64),
        
        # Volume indicators
        Field(name="obv", dtype=Float64),
        Field(name="vpt", dtype=Float64),
        Field(name="mfi_14", dtype=Float64),
        Field(name="ad_line", dtype=Float64),
        Field(name="chaikin_osc", dtype=Float64),
        
        # Price patterns
        Field(name="pivot_high", dtype=Float64),
        Field(name="pivot_low", dtype=Float64),
        Field(name="support_1", dtype=Float64),
        Field(name="resistance_1", dtype=Float64),
    ],
    source=technical_indicators_source,
    online=True,
    tags={"team": "alpha", "tier": "high"},
)

# Alternative Data Features
alternative_data_fv = FeatureView(
    name="alternative_data_features",
    entities=[stock],
    ttl=timedelta(days=1),
    schema=[
        # Sentiment features
        Field(name="news_sentiment", dtype=Float64),
        Field(name="social_sentiment", dtype=Float64),
        Field(name="earnings_sentiment", dtype=Float64),
        Field(name="analyst_sentiment", dtype=Float64),
        Field(name="sentiment_momentum", dtype=Float64),
        Field(name="sentiment_volume", dtype=Float64),
        
        # Web/social metrics
        Field(name="search_interest", dtype=Float64),
        Field(name="social_mentions", dtype=Float64),
        Field(name="social_reach", dtype=Float64),
        Field(name="reddit_wsb_mentions", dtype=Float64),
        Field(name="twitter_followers_growth", dtype=Float64),
        
        # Fundamental events
        Field(name="days_to_earnings", dtype=Float64),
        Field(name="earnings_surprise", dtype=Float64),
        Field(name="guidance_change", dtype=Float64),
        Field(name="insider_buying", dtype=Float64),
        Field(name="institutional_ownership_change", dtype=Float64),
        
        # Economic indicators
        Field(name="sector_momentum", dtype=Float64),
        Field(name="industry_momentum", dtype=Float64),
        Field(name="beta_spy", dtype=Float64),
        Field(name="correlation_spy_21d", dtype=Float64),
    ],
    source=alternative_data_source,
    online=True,
    tags={"team": "alpha", "tier": "medium"},
)

# Factor Scores Features
factor_scores_fv = FeatureView(
    name="factor_scores",
    entities=[stock],
    ttl=timedelta(days=1),
    schema=[
        # Value factors
        Field(name="factor_value", dtype=Float64),
        Field(name="factor_book_to_market", dtype=Float64),
        Field(name="factor_earnings_yield", dtype=Float64),
        Field(name="factor_fcf_yield", dtype=Float64),
        
        # Momentum factors
        Field(name="factor_momentum_12m", dtype=Float64),
        Field(name="factor_momentum_6m", dtype=Float64),
        Field(name="factor_momentum_1m", dtype=Float64),
        Field(name="factor_52w_high", dtype=Float64),
        
        # Quality factors
        Field(name="factor_quality", dtype=Float64),
        Field(name="factor_profitability", dtype=Float64),
        Field(name="factor_asset_growth", dtype=Float64),
        Field(name="factor_accruals", dtype=Float64),
        
        # Size and volatility
        Field(name="factor_size", dtype=Float64),
        Field(name="factor_low_vol", dtype=Float64),
        Field(name="factor_beta", dtype=Float64),
        Field(name="factor_idio_vol", dtype=Float64),
        
        # Composite scores
        Field(name="alpha_composite", dtype=Float64),
        Field(name="risk_score", dtype=Float64),
    ],
    source=factor_scores_source,
    online=True,
    tags={"team": "alpha", "tier": "critical"},
)

# Order Flow Analytics Features
order_flow_fv = FeatureView(
    name="order_flow_analytics",
    entities=[stock],
    ttl=timedelta(minutes=5),  # High-frequency updates
    schema=[
        # Trade imbalance
        Field(name="trade_imbalance", dtype=Float64),
        Field(name="volume_imbalance", dtype=Float64),
        Field(name="order_imbalance", dtype=Float64),
        
        # Microstructure
        Field(name="quoted_spread", dtype=Float64),
        Field(name="effective_spread", dtype=Float64),
        Field(name="realized_spread", dtype=Float64),
        Field(name="price_impact", dtype=Float64),
        Field(name="kyle_lambda", dtype=Float64),
        
        # Order book metrics
        Field(name="bid_depth_1", dtype=Float64),
        Field(name="ask_depth_1", dtype=Float64),
        Field(name="bid_depth_5", dtype=Float64),
        Field(name="ask_depth_5", dtype=Float64),
        Field(name="book_imbalance", dtype=Float64),
        
        # Flow toxicity
        Field(name="vpin", dtype=Float64),  # Volume-Synchronized PIN
        Field(name="pin", dtype=Float64),   # Probability of Informed Trading
        
        # Execution metrics
        Field(name="arrival_price", dtype=Float64),
        Field(name="twap_benchmark", dtype=Float64),
        Field(name="vwap_benchmark", dtype=Float64),
    ],
    source=order_flow_source,
    online=True,
    tags={"team": "execution", "tier": "critical"},
)

# =============================================================================
# FEATURE SERVICES - Bundles for different use cases
# =============================================================================

from feast import FeatureService

# Alpha generation service
alpha_service = FeatureService(
    name="alpha_generation",
    features=[
        market_data_fv,
        technical_indicators_fv,
        alternative_data_fv,
        factor_scores_fv,
    ],
    tags={"use_case": "alpha_model"},
)

# Execution optimization service
execution_service = FeatureService(
    name="execution_optimization",
    features=[
        market_data_fv[["close", "volume", "vwap", "spread"]],
        order_flow_fv,
    ],
    tags={"use_case": "execution"},
)

# Risk management service
risk_service = FeatureService(
    name="risk_management",
    features=[
        market_data_fv[["returns_1d", "returns_5d", "returns_21d"]],
        technical_indicators_fv[["atr_14", "realized_vol_21", "realized_vol_63"]],
        factor_scores_fv[["factor_beta", "factor_idio_vol", "risk_score"]],
    ],
    tags={"use_case": "risk"},
)


# =============================================================================
# ON-DEMAND FEATURES - Computed at request time
# =============================================================================

from feast import on_demand_feature_view
from feast.types import Float64
import pandas as pd

@on_demand_feature_view(
    sources=[market_data_fv, technical_indicators_fv],
    schema=[
        Field(name="momentum_signal", dtype=Float64),
        Field(name="mean_reversion_signal", dtype=Float64),
        Field(name="volatility_regime", dtype=Float64),
    ],
)
def compute_trading_signals(inputs: pd.DataFrame) -> pd.DataFrame:
    """Compute real-time trading signals from base features"""
    df = pd.DataFrame()
    
    # Momentum signal: RSI + MACD combination
    df["momentum_signal"] = (
        (inputs["rsi_14"] - 50) / 50 * 0.5 +
        inputs["macd_hist"].clip(-1, 1) * 0.5
    )
    
    # Mean reversion signal: Bollinger band position
    bb_position = (inputs["close"] - inputs["bollinger_middle"]) / (
        inputs["bollinger_upper"] - inputs["bollinger_lower"]
    )
    df["mean_reversion_signal"] = -bb_position.clip(-1, 1)
    
    # Volatility regime: 0=low, 0.5=normal, 1=high
    vol_ratio = inputs["realized_vol_21"] / inputs["realized_vol_63"]
    df["volatility_regime"] = (vol_ratio - 0.5).clip(0, 1)
    
    return df

