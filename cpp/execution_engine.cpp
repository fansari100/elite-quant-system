/**
 * Elite Quant System - C++ Execution Engine
 * ==========================================
 * Ultra-low latency order execution and market data handling.
 * 
 * This is the performance-critical layer that runs alongside the Python
 * ML pipeline. In production, this handles:
 * - Sub-microsecond order routing
 * - Market data feed handling
 * - Order book management
 * - Risk checks at execution time
 * 
 * Compile with:
 *   g++ -O3 -march=native -std=c++20 -pthread execution_engine.cpp -o execution_engine
 * 
 * For production HFT:
 *   g++ -O3 -march=native -std=c++20 -pthread -DNDEBUG -flto execution_engine.cpp -o execution_engine
 */

#include <iostream>
#include <vector>
#include <array>
#include <unordered_map>
#include <queue>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <memory>
#include <functional>
#include <fstream>
#include <sstream>

// Lock-free queue for ultra-low latency
template<typename T, size_t SIZE = 65536>
class LockFreeQueue {
private:
    std::array<T, SIZE> buffer_;
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
    
public:
    bool push(const T& item) {
        size_t tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (tail + 1) % SIZE;
        
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false;  // Queue full
        }
        
        buffer_[tail] = item;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }
    
    bool pop(T& item) {
        size_t head = head_.load(std::memory_order_relaxed);
        
        if (head == tail_.load(std::memory_order_acquire)) {
            return false;  // Queue empty
        }
        
        item = buffer_[head];
        head_.store((head + 1) % SIZE, std::memory_order_release);
        return true;
    }
    
    bool empty() const {
        return head_.load(std::memory_order_acquire) == 
               tail_.load(std::memory_order_acquire);
    }
};


// Order side enumeration
enum class Side : uint8_t { BUY = 0, SELL = 1 };

// Order type enumeration
enum class OrderType : uint8_t { 
    MARKET = 0, 
    LIMIT = 1, 
    IOC = 2,      // Immediate or Cancel
    FOK = 3       // Fill or Kill
};

// Order status
enum class OrderStatus : uint8_t {
    NEW = 0,
    PENDING = 1,
    FILLED = 2,
    PARTIAL = 3,
    CANCELLED = 4,
    REJECTED = 5
};


/**
 * Order structure - cache-line optimized (64 bytes)
 */
struct alignas(64) Order {
    uint64_t order_id;
    uint64_t timestamp_ns;      // Nanosecond timestamp
    uint32_t symbol_id;
    double price;
    double quantity;
    double filled_qty;
    Side side;
    OrderType type;
    OrderStatus status;
    uint8_t padding[5];         // Align to 64 bytes
    
    Order() : order_id(0), timestamp_ns(0), symbol_id(0), 
              price(0), quantity(0), filled_qty(0),
              side(Side::BUY), type(OrderType::MARKET), 
              status(OrderStatus::NEW) {}
};


/**
 * Market data tick - optimized for cache efficiency
 */
struct alignas(32) MarketTick {
    uint64_t timestamp_ns;
    uint32_t symbol_id;
    double bid_price;
    double ask_price;
    double bid_size;
    double ask_size;
    double last_price;
    double last_size;
};


/**
 * Position tracking per symbol
 */
struct Position {
    double quantity;
    double avg_price;
    double realized_pnl;
    double unrealized_pnl;
    double market_value;
    
    Position() : quantity(0), avg_price(0), realized_pnl(0), 
                 unrealized_pnl(0), market_value(0) {}
};


/**
 * Risk limits configuration
 */
struct RiskLimits {
    double max_position_value;
    double max_order_value;
    double max_loss_daily;
    double max_drawdown;
    int max_orders_per_second;
    
    RiskLimits() : max_position_value(1000000), max_order_value(100000),
                   max_loss_daily(50000), max_drawdown(0.05),
                   max_orders_per_second(100) {}
};


/**
 * High-precision timer for latency measurement
 */
class HighResTimer {
public:
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;
    
    static uint64_t now_ns() {
        auto now = clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
        ).count();
    }
    
    static double elapsed_us(uint64_t start_ns) {
        return (now_ns() - start_ns) / 1000.0;
    }
};


/**
 * Order Book - Level 2 market data
 */
class OrderBook {
private:
    struct PriceLevel {
        double price;
        double quantity;
        int order_count;
    };
    
    std::vector<PriceLevel> bids_;  // Sorted descending
    std::vector<PriceLevel> asks_;  // Sorted ascending
    uint32_t symbol_id_;
    uint64_t last_update_ns_;
    
public:
    OrderBook(uint32_t symbol_id) : symbol_id_(symbol_id), last_update_ns_(0) {
        bids_.reserve(100);
        asks_.reserve(100);
    }
    
    void update(const MarketTick& tick) {
        last_update_ns_ = tick.timestamp_ns;
        
        // Simplified: just track top of book
        if (bids_.empty()) {
            bids_.push_back({tick.bid_price, tick.bid_size, 1});
        } else {
            bids_[0] = {tick.bid_price, tick.bid_size, 1};
        }
        
        if (asks_.empty()) {
            asks_.push_back({tick.ask_price, tick.ask_size, 1});
        } else {
            asks_[0] = {tick.ask_price, tick.ask_size, 1};
        }
    }
    
    double get_mid_price() const {
        if (bids_.empty() || asks_.empty()) return 0;
        return (bids_[0].price + asks_[0].price) / 2.0;
    }
    
    double get_spread() const {
        if (bids_.empty() || asks_.empty()) return 0;
        return asks_[0].price - bids_[0].price;
    }
    
    double get_spread_bps() const {
        double mid = get_mid_price();
        if (mid == 0) return 0;
        return get_spread() / mid * 10000;
    }
};


/**
 * Risk Manager - Pre-trade risk checks
 */
class RiskManager {
private:
    RiskLimits limits_;
    std::unordered_map<uint32_t, Position> positions_;
    double daily_pnl_;
    double peak_equity_;
    std::atomic<int> orders_this_second_{0};
    uint64_t last_second_ns_;
    std::mutex mtx_;
    
public:
    RiskManager(const RiskLimits& limits) 
        : limits_(limits), daily_pnl_(0), peak_equity_(0), last_second_ns_(0) {}
    
    enum class RiskResult {
        APPROVED,
        REJECTED_POSITION_LIMIT,
        REJECTED_ORDER_SIZE,
        REJECTED_LOSS_LIMIT,
        REJECTED_DRAWDOWN,
        REJECTED_RATE_LIMIT
    };
    
    RiskResult check_order(const Order& order, double current_price) {
        std::lock_guard<std::mutex> lock(mtx_);
        
        // Rate limiting
        uint64_t now = HighResTimer::now_ns();
        uint64_t current_second = now / 1000000000;
        uint64_t last_second = last_second_ns_ / 1000000000;
        
        if (current_second != last_second) {
            orders_this_second_ = 0;
            last_second_ns_ = now;
        }
        
        if (orders_this_second_ >= limits_.max_orders_per_second) {
            return RiskResult::REJECTED_RATE_LIMIT;
        }
        
        // Order size check
        double order_value = order.quantity * current_price;
        if (order_value > limits_.max_order_value) {
            return RiskResult::REJECTED_ORDER_SIZE;
        }
        
        // Position limit check
        auto it = positions_.find(order.symbol_id);
        double current_position = (it != positions_.end()) ? it->second.quantity : 0;
        double new_position = current_position;
        
        if (order.side == Side::BUY) {
            new_position += order.quantity;
        } else {
            new_position -= order.quantity;
        }
        
        double new_position_value = std::abs(new_position) * current_price;
        if (new_position_value > limits_.max_position_value) {
            return RiskResult::REJECTED_POSITION_LIMIT;
        }
        
        // Daily loss check
        if (-daily_pnl_ > limits_.max_loss_daily) {
            return RiskResult::REJECTED_LOSS_LIMIT;
        }
        
        orders_this_second_++;
        return RiskResult::APPROVED;
    }
    
    void update_position(uint32_t symbol_id, double quantity, double price, Side side) {
        std::lock_guard<std::mutex> lock(mtx_);
        
        Position& pos = positions_[symbol_id];
        double trade_value = quantity * price;
        
        if (side == Side::BUY) {
            double new_qty = pos.quantity + quantity;
            pos.avg_price = (pos.quantity * pos.avg_price + trade_value) / new_qty;
            pos.quantity = new_qty;
        } else {
            double realized = quantity * (price - pos.avg_price);
            pos.realized_pnl += realized;
            daily_pnl_ += realized;
            pos.quantity -= quantity;
        }
    }
    
    double get_daily_pnl() const { return daily_pnl_; }
};


/**
 * Execution Engine - Core order management
 */
class ExecutionEngine {
private:
    LockFreeQueue<Order> order_queue_;
    LockFreeQueue<Order> fill_queue_;
    std::unordered_map<uint32_t, std::unique_ptr<OrderBook>> order_books_;
    RiskManager risk_manager_;
    
    std::atomic<uint64_t> next_order_id_{1};
    std::atomic<bool> running_{false};
    std::thread processing_thread_;
    
    // Latency tracking
    std::vector<double> latencies_us_;
    std::mutex latency_mtx_;
    
public:
    ExecutionEngine(const RiskLimits& limits = RiskLimits()) 
        : risk_manager_(limits) {
        latencies_us_.reserve(10000);
    }
    
    ~ExecutionEngine() {
        stop();
    }
    
    void add_symbol(uint32_t symbol_id) {
        order_books_[symbol_id] = std::make_unique<OrderBook>(symbol_id);
    }
    
    void start() {
        running_ = true;
        processing_thread_ = std::thread(&ExecutionEngine::process_orders, this);
    }
    
    void stop() {
        running_ = false;
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
    }
    
    uint64_t submit_order(Order order) {
        uint64_t start_ns = HighResTimer::now_ns();
        
        order.order_id = next_order_id_++;
        order.timestamp_ns = start_ns;
        order.status = OrderStatus::PENDING;
        
        // Get current price for risk check
        double current_price = 0;
        auto it = order_books_.find(order.symbol_id);
        if (it != order_books_.end()) {
            current_price = it->second->get_mid_price();
        }
        
        // Pre-trade risk check
        auto risk_result = risk_manager_.check_order(order, current_price);
        if (risk_result != RiskManager::RiskResult::APPROVED) {
            order.status = OrderStatus::REJECTED;
            return 0;  // Rejected
        }
        
        // Submit to queue
        if (!order_queue_.push(order)) {
            return 0;  // Queue full
        }
        
        // Track latency
        double latency_us = HighResTimer::elapsed_us(start_ns);
        {
            std::lock_guard<std::mutex> lock(latency_mtx_);
            latencies_us_.push_back(latency_us);
        }
        
        return order.order_id;
    }
    
    void update_market_data(const MarketTick& tick) {
        auto it = order_books_.find(tick.symbol_id);
        if (it != order_books_.end()) {
            it->second->update(tick);
        }
    }
    
    bool get_fill(Order& fill) {
        return fill_queue_.pop(fill);
    }
    
    void get_latency_stats(double& mean, double& p50, double& p99) {
        std::lock_guard<std::mutex> lock(latency_mtx_);
        
        if (latencies_us_.empty()) {
            mean = p50 = p99 = 0;
            return;
        }
        
        std::vector<double> sorted = latencies_us_;
        std::sort(sorted.begin(), sorted.end());
        
        double sum = 0;
        for (double v : sorted) sum += v;
        mean = sum / sorted.size();
        
        p50 = sorted[sorted.size() / 2];
        p99 = sorted[sorted.size() * 99 / 100];
    }
    
private:
    void process_orders() {
        Order order;
        
        while (running_) {
            if (order_queue_.pop(order)) {
                // Simulate execution (in production, send to exchange)
                execute_order(order);
            } else {
                // Busy wait with minimal overhead
                std::this_thread::yield();
            }
        }
    }
    
    void execute_order(Order& order) {
        auto it = order_books_.find(order.symbol_id);
        if (it == order_books_.end()) {
            order.status = OrderStatus::REJECTED;
            fill_queue_.push(order);
            return;
        }
        
        double mid_price = it->second->get_mid_price();
        double spread = it->second->get_spread();
        
        // Simulate fill with slippage
        double fill_price = mid_price;
        if (order.side == Side::BUY) {
            fill_price += spread / 2;  // Pay the ask
        } else {
            fill_price -= spread / 2;  // Hit the bid
        }
        
        // Update order
        order.price = fill_price;
        order.filled_qty = order.quantity;
        order.status = OrderStatus::FILLED;
        
        // Update positions
        risk_manager_.update_position(
            order.symbol_id, 
            order.quantity, 
            fill_price, 
            order.side
        );
        
        // Push to fill queue
        fill_queue_.push(order);
    }
};


/**
 * Python Integration Bridge
 * Receives signals from Python ML layer, executes trades
 */
class PythonBridge {
private:
    ExecutionEngine& engine_;
    std::unordered_map<std::string, uint32_t> symbol_map_;
    uint32_t next_symbol_id_ = 0;
    
public:
    PythonBridge(ExecutionEngine& engine) : engine_(engine) {}
    
    uint32_t register_symbol(const std::string& symbol) {
        uint32_t id = next_symbol_id_++;
        symbol_map_[symbol] = id;
        engine_.add_symbol(id);
        return id;
    }
    
    // Called from Python with portfolio weights
    void execute_rebalance(
        const std::vector<std::string>& symbols,
        const std::vector<double>& target_weights,
        const std::vector<double>& current_weights,
        double portfolio_value
    ) {
        for (size_t i = 0; i < symbols.size(); i++) {
            double weight_diff = target_weights[i] - current_weights[i];
            
            if (std::abs(weight_diff) < 0.001) continue;  // Skip small trades
            
            auto it = symbol_map_.find(symbols[i]);
            if (it == symbol_map_.end()) continue;
            
            Order order;
            order.symbol_id = it->second;
            order.quantity = std::abs(weight_diff * portfolio_value / 100.0);  // Simplified
            order.side = (weight_diff > 0) ? Side::BUY : Side::SELL;
            order.type = OrderType::MARKET;
            
            engine_.submit_order(order);
        }
    }
};


/**
 * Main entry point for testing
 */
int main() {
    std::cout << "Elite Quant System - C++ Execution Engine" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Initialize
    RiskLimits limits;
    limits.max_position_value = 1000000;
    limits.max_order_value = 100000;
    limits.max_orders_per_second = 1000;
    
    ExecutionEngine engine(limits);
    PythonBridge bridge(engine);
    
    // Register symbols
    std::vector<std::string> symbols = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"};
    for (const auto& sym : symbols) {
        uint32_t id = bridge.register_symbol(sym);
        std::cout << "Registered " << sym << " with ID " << id << std::endl;
    }
    
    // Start engine
    engine.start();
    
    // Simulate market data updates
    for (int i = 0; i < 5; i++) {
        MarketTick tick;
        tick.timestamp_ns = HighResTimer::now_ns();
        tick.symbol_id = i;
        tick.bid_price = 100.0 + i * 10;
        tick.ask_price = 100.05 + i * 10;
        tick.bid_size = 1000;
        tick.ask_size = 1000;
        engine.update_market_data(tick);
    }
    
    // Test order submission
    std::cout << "\nSubmitting test orders..." << std::endl;
    
    uint64_t start_ns = HighResTimer::now_ns();
    
    for (int i = 0; i < 1000; i++) {
        Order order;
        order.symbol_id = i % 5;
        order.quantity = 100;
        order.side = (i % 2 == 0) ? Side::BUY : Side::SELL;
        order.type = OrderType::MARKET;
        
        engine.submit_order(order);
    }
    
    double elapsed_us = HighResTimer::elapsed_us(start_ns);
    std::cout << "Submitted 1000 orders in " << elapsed_us << " us" << std::endl;
    std::cout << "Throughput: " << (1000.0 / elapsed_us * 1e6) << " orders/sec" << std::endl;
    
    // Wait for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Get latency stats
    double mean, p50, p99;
    engine.get_latency_stats(mean, p50, p99);
    
    std::cout << "\nLatency Statistics:" << std::endl;
    std::cout << "  Mean: " << mean << " us" << std::endl;
    std::cout << "  P50:  " << p50 << " us" << std::endl;
    std::cout << "  P99:  " << p99 << " us" << std::endl;
    
    // Check fills
    Order fill;
    int fill_count = 0;
    while (engine.get_fill(fill)) {
        fill_count++;
    }
    std::cout << "\nReceived " << fill_count << " fills" << std::endl;
    
    // Cleanup
    engine.stop();
    
    std::cout << "\n✓ Execution engine test complete" << std::endl;
    
    return 0;
}

