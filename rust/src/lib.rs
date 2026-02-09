//! Elite Quant System - Rust Execution Engine
//! ============================================
//! Memory-safe, low-latency trading components.
//!
//! Rust provides:
//! - Zero-cost abstractions
//! - Memory safety without garbage collection
//! - Fearless concurrency
//! - C-like performance with modern ergonomics

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam::channel::{bounded, Receiver, Sender};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error types for the trading system
#[derive(Error, Debug)]
pub enum TradingError {
    #[error("Order rejected: {reason}")]
    OrderRejected { reason: String },
    
    #[error("Risk limit exceeded: {limit}")]
    RiskLimitExceeded { limit: String },
    
    #[error("Insufficient liquidity")]
    InsufficientLiquidity,
    
    #[error("Connection error: {0}")]
    ConnectionError(String),
    
    #[error("Internal error: {0}")]
    InternalError(String),
}

pub type Result<T> = std::result::Result<T, TradingError>;

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    StopLoss,
    TakeProfit,
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    New,
    Pending,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

/// Order structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: u64,
    pub symbol: String,
    pub side: Side,
    pub order_type: OrderType,
    pub quantity: f64,
    pub price: Option<f64>,
    pub filled_quantity: f64,
    pub average_fill_price: f64,
    pub status: OrderStatus,
    pub timestamp_ns: u64,
}

impl Order {
    pub fn new_market(symbol: String, side: Side, quantity: f64) -> Self {
        Self {
            id: 0,
            symbol,
            side,
            order_type: OrderType::Market,
            quantity,
            price: None,
            filled_quantity: 0.0,
            average_fill_price: 0.0,
            status: OrderStatus::New,
            timestamp_ns: current_time_ns(),
        }
    }
    
    pub fn new_limit(symbol: String, side: Side, quantity: f64, price: f64) -> Self {
        Self {
            id: 0,
            symbol,
            side,
            order_type: OrderType::Limit,
            quantity,
            price: Some(price),
            filled_quantity: 0.0,
            average_fill_price: 0.0,
            status: OrderStatus::New,
            timestamp_ns: current_time_ns(),
        }
    }
}

/// Market tick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketTick {
    pub symbol: String,
    pub bid: f64,
    pub ask: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub last_price: f64,
    pub last_size: f64,
    pub timestamp_ns: u64,
}

impl MarketTick {
    pub fn mid_price(&self) -> f64 {
        (self.bid + self.ask) / 2.0
    }
    
    pub fn spread(&self) -> f64 {
        self.ask - self.bid
    }
    
    pub fn spread_bps(&self) -> f64 {
        self.spread() / self.mid_price() * 10000.0
    }
}

/// Position tracking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub average_price: f64,
    pub market_value: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
}

/// Risk limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    pub max_position_value: f64,
    pub max_order_value: f64,
    pub max_daily_loss: f64,
    pub max_drawdown_pct: f64,
    pub max_orders_per_second: u32,
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_position_value: 1_000_000.0,
            max_order_value: 100_000.0,
            max_daily_loss: 50_000.0,
            max_drawdown_pct: 0.05,
            max_orders_per_second: 100,
        }
    }
}

/// High-resolution time utilities
pub fn current_time_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Lock-free order ID generator
pub struct OrderIdGenerator {
    counter: AtomicU64,
}

impl OrderIdGenerator {
    pub fn new() -> Self {
        Self {
            counter: AtomicU64::new(1),
        }
    }
    
    pub fn next(&self) -> u64 {
        self.counter.fetch_add(1, Ordering::SeqCst)
    }
}

impl Default for OrderIdGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Risk manager for pre-trade checks
pub struct RiskManager {
    limits: RiskLimits,
    positions: RwLock<HashMap<String, Position>>,
    daily_pnl: AtomicU64,  // Stored as fixed-point (multiply by 100 for cents)
    orders_this_second: AtomicU64,
    last_second: AtomicU64,
}

impl RiskManager {
    pub fn new(limits: RiskLimits) -> Self {
        Self {
            limits,
            positions: RwLock::new(HashMap::new()),
            daily_pnl: AtomicU64::new(0),
            orders_this_second: AtomicU64::new(0),
            last_second: AtomicU64::new(0),
        }
    }
    
    pub fn check_order(&self, order: &Order, current_price: f64) -> Result<()> {
        // Rate limiting
        let now_seconds = current_time_ns() / 1_000_000_000;
        let last = self.last_second.load(Ordering::Relaxed);
        
        if now_seconds != last {
            self.last_second.store(now_seconds, Ordering::Relaxed);
            self.orders_this_second.store(0, Ordering::Relaxed);
        }
        
        let count = self.orders_this_second.fetch_add(1, Ordering::Relaxed);
        if count >= self.limits.max_orders_per_second as u64 {
            return Err(TradingError::RiskLimitExceeded {
                limit: "max_orders_per_second".to_string(),
            });
        }
        
        // Order size check
        let order_value = order.quantity * current_price;
        if order_value > self.limits.max_order_value {
            return Err(TradingError::RiskLimitExceeded {
                limit: "max_order_value".to_string(),
            });
        }
        
        // Position limit check
        let positions = self.positions.read();
        let current_position = positions.get(&order.symbol).map(|p| p.quantity).unwrap_or(0.0);
        
        let new_position = match order.side {
            Side::Buy => current_position + order.quantity,
            Side::Sell => current_position - order.quantity,
        };
        
        let new_position_value = new_position.abs() * current_price;
        if new_position_value > self.limits.max_position_value {
            return Err(TradingError::RiskLimitExceeded {
                limit: "max_position_value".to_string(),
            });
        }
        
        Ok(())
    }
    
    pub fn update_position(&self, symbol: &str, quantity: f64, price: f64, side: Side) {
        let mut positions = self.positions.write();
        let position = positions.entry(symbol.to_string()).or_default();
        position.symbol = symbol.to_string();
        
        match side {
            Side::Buy => {
                let new_qty = position.quantity + quantity;
                position.average_price = 
                    (position.quantity * position.average_price + quantity * price) / new_qty;
                position.quantity = new_qty;
            }
            Side::Sell => {
                let realized = quantity * (price - position.average_price);
                position.realized_pnl += realized;
                position.quantity -= quantity;
            }
        }
        
        position.market_value = position.quantity * price;
    }
    
    pub fn get_positions(&self) -> Vec<Position> {
        self.positions.read().values().cloned().collect()
    }
}

/// Order book for a single symbol
pub struct OrderBook {
    pub symbol: String,
    bids: RwLock<Vec<(f64, f64)>>,  // (price, size)
    asks: RwLock<Vec<(f64, f64)>>,
    last_update_ns: AtomicU64,
}

impl OrderBook {
    pub fn new(symbol: String) -> Self {
        Self {
            symbol,
            bids: RwLock::new(Vec::with_capacity(100)),
            asks: RwLock::new(Vec::with_capacity(100)),
            last_update_ns: AtomicU64::new(0),
        }
    }
    
    pub fn update(&self, tick: &MarketTick) {
        let mut bids = self.bids.write();
        let mut asks = self.asks.write();
        
        // Simplified: just track top of book
        bids.clear();
        bids.push((tick.bid, tick.bid_size));
        
        asks.clear();
        asks.push((tick.ask, tick.ask_size));
        
        self.last_update_ns.store(tick.timestamp_ns, Ordering::Release);
    }
    
    pub fn best_bid(&self) -> Option<(f64, f64)> {
        self.bids.read().first().copied()
    }
    
    pub fn best_ask(&self) -> Option<(f64, f64)> {
        self.asks.read().first().copied()
    }
    
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }
}

/// Execution engine
pub struct ExecutionEngine {
    risk_manager: Arc<RiskManager>,
    order_books: Arc<RwLock<HashMap<String, Arc<OrderBook>>>>,
    order_sender: Sender<Order>,
    fill_receiver: Receiver<Order>,
    id_generator: OrderIdGenerator,
    latencies_ns: RwLock<Vec<u64>>,
}

impl ExecutionEngine {
    pub fn new(risk_limits: RiskLimits) -> Self {
        let (order_tx, _order_rx) = bounded::<Order>(65536);
        let (_fill_tx, fill_rx) = bounded::<Order>(65536);
        
        Self {
            risk_manager: Arc::new(RiskManager::new(risk_limits)),
            order_books: Arc::new(RwLock::new(HashMap::new())),
            order_sender: order_tx,
            fill_receiver: fill_rx,
            id_generator: OrderIdGenerator::new(),
            latencies_ns: RwLock::new(Vec::with_capacity(10000)),
        }
    }
    
    pub fn add_symbol(&self, symbol: &str) {
        let mut books = self.order_books.write();
        books.insert(symbol.to_string(), Arc::new(OrderBook::new(symbol.to_string())));
    }
    
    pub fn update_market_data(&self, tick: MarketTick) {
        let books = self.order_books.read();
        if let Some(book) = books.get(&tick.symbol) {
            book.update(&tick);
        }
    }
    
    pub fn submit_order(&self, mut order: Order) -> Result<u64> {
        let start = Instant::now();
        
        // Assign order ID
        order.id = self.id_generator.next();
        order.timestamp_ns = current_time_ns();
        
        // Get current price
        let books = self.order_books.read();
        let current_price = books
            .get(&order.symbol)
            .and_then(|b| b.mid_price())
            .unwrap_or(100.0);  // Fallback price
        drop(books);
        
        // Risk check
        self.risk_manager.check_order(&order, current_price)?;
        
        // Submit to queue
        order.status = OrderStatus::Pending;
        if self.order_sender.try_send(order.clone()).is_err() {
            return Err(TradingError::InternalError("Order queue full".to_string()));
        }
        
        // Track latency
        let latency = start.elapsed().as_nanos() as u64;
        self.latencies_ns.write().push(latency);
        
        Ok(order.id)
    }
    
    pub fn get_latency_stats(&self) -> (f64, f64, f64) {
        let latencies = self.latencies_ns.read();
        if latencies.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        
        let mut sorted: Vec<u64> = latencies.clone();
        sorted.sort_unstable();
        
        let mean = sorted.iter().sum::<u64>() as f64 / sorted.len() as f64;
        let p50 = sorted[sorted.len() / 2] as f64;
        let p99 = sorted[sorted.len() * 99 / 100] as f64;
        
        (mean / 1000.0, p50 / 1000.0, p99 / 1000.0)  // Convert to microseconds
    }
    
    pub fn execute_rebalance(
        &self,
        symbols: &[String],
        target_weights: &[f64],
        current_weights: &[f64],
        portfolio_value: f64,
    ) -> Vec<Result<u64>> {
        let mut results = Vec::new();
        
        for (i, symbol) in symbols.iter().enumerate() {
            let weight_diff = target_weights[i] - current_weights[i];
            
            // Skip small trades
            if weight_diff.abs() < 0.001 {
                continue;
            }
            
            let trade_value = weight_diff.abs() * portfolio_value;
            
            // Get price for quantity calculation
            let books = self.order_books.read();
            let price = books
                .get(symbol)
                .and_then(|b| b.mid_price())
                .unwrap_or(100.0);
            drop(books);
            
            let quantity = trade_value / price;
            let side = if weight_diff > 0.0 { Side::Buy } else { Side::Sell };
            
            let order = Order::new_market(symbol.clone(), side, quantity);
            results.push(self.submit_order(order));
        }
        
        results
    }
    
    pub fn get_positions(&self) -> Vec<Position> {
        self.risk_manager.get_positions()
    }
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::prelude::*;
    
    #[pyclass]
    struct PyExecutionEngine {
        engine: ExecutionEngine,
    }
    
    #[pymethods]
    impl PyExecutionEngine {
        #[new]
        fn new() -> Self {
            Self {
                engine: ExecutionEngine::new(RiskLimits::default()),
            }
        }
        
        fn add_symbol(&self, symbol: &str) {
            self.engine.add_symbol(symbol);
        }
        
        fn submit_order(&self, symbol: &str, side: &str, quantity: f64) -> PyResult<u64> {
            let side = match side {
                "buy" => Side::Buy,
                "sell" => Side::Sell,
                _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid side")),
            };
            
            let order = Order::new_market(symbol.to_string(), side, quantity);
            self.engine
                .submit_order(order)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }
        
        fn get_latency_stats(&self) -> (f64, f64, f64) {
            self.engine.get_latency_stats()
        }
    }
    
    #[pymodule]
    fn elite_quant(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
        m.add_class::<PyExecutionEngine>()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_order_creation() {
        let order = Order::new_market("AAPL".to_string(), Side::Buy, 100.0);
        assert_eq!(order.symbol, "AAPL");
        assert_eq!(order.quantity, 100.0);
    }
    
    #[test]
    fn test_execution_engine() {
        let engine = ExecutionEngine::new(RiskLimits::default());
        engine.add_symbol("AAPL");
        
        // Update market data
        let tick = MarketTick {
            symbol: "AAPL".to_string(),
            bid: 150.0,
            ask: 150.05,
            bid_size: 1000.0,
            ask_size: 1000.0,
            last_price: 150.02,
            last_size: 100.0,
            timestamp_ns: current_time_ns(),
        };
        engine.update_market_data(tick);
        
        // Submit order
        let order = Order::new_market("AAPL".to_string(), Side::Buy, 100.0);
        let result = engine.submit_order(order);
        assert!(result.is_ok());
    }
}

