//! Elite Quant Execution Server
//! =============================
//! Standalone low-latency execution server.

use std::sync::Arc;
use std::time::Instant;

use elite_quant::{ExecutionEngine, MarketTick, Order, RiskLimits, Side, current_time_ns};

fn main() {
    println!("Elite Quant System - Rust Execution Server");
    println!("==========================================");
    println!();
    
    // Initialize execution engine
    let limits = RiskLimits {
        max_position_value: 1_000_000.0,
        max_order_value: 100_000.0,
        max_daily_loss: 50_000.0,
        max_drawdown_pct: 0.05,
        max_orders_per_second: 1000,
    };
    
    let engine = Arc::new(ExecutionEngine::new(limits));
    
    // Register symbols
    let symbols = vec!["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"];
    for symbol in &symbols {
        engine.add_symbol(symbol);
        println!("Registered symbol: {}", symbol);
    }
    
    // Simulate market data
    println!("\nUpdating market data...");
    for (i, symbol) in symbols.iter().enumerate() {
        let base_price = 100.0 + (i as f64 * 50.0);
        let tick = MarketTick {
            symbol: symbol.to_string(),
            bid: base_price,
            ask: base_price + 0.05,
            bid_size: 1000.0,
            ask_size: 1000.0,
            last_price: base_price + 0.02,
            last_size: 100.0,
            timestamp_ns: current_time_ns(),
        };
        engine.update_market_data(tick);
    }
    
    // Benchmark order submission
    println!("\nBenchmarking order submission...");
    let start = Instant::now();
    let n_orders = 10_000;
    let mut successful = 0;
    
    for i in 0..n_orders {
        let symbol = symbols[i % symbols.len()];
        let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
        let order = Order::new_market(symbol.to_string(), side, 10.0);
        
        if engine.submit_order(order).is_ok() {
            successful += 1;
        }
    }
    
    let elapsed = start.elapsed();
    let orders_per_sec = n_orders as f64 / elapsed.as_secs_f64();
    
    println!("Submitted {} orders in {:?}", n_orders, elapsed);
    println!("Successful: {}/{}", successful, n_orders);
    println!("Throughput: {:.0} orders/sec", orders_per_sec);
    
    // Latency stats
    let (mean, p50, p99) = engine.get_latency_stats();
    println!("\nLatency Statistics:");
    println!("  Mean: {:.2} μs", mean);
    println!("  P50:  {:.2} μs", p50);
    println!("  P99:  {:.2} μs", p99);
    
    // Positions
    println!("\nPositions:");
    for position in engine.get_positions() {
        println!(
            "  {}: qty={:.2}, avg_price={:.2}, pnl={:.2}",
            position.symbol,
            position.quantity,
            position.average_price,
            position.realized_pnl
        );
    }
    
    println!("\n✓ Execution server test complete");
}

