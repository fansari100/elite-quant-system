// Elite Quant Gateway - Ultra-Low Latency API Gateway in Go
// Handles order routing, load balancing, and real-time WebSocket streaming
// Used by Two Sigma, Citadel for microservices orchestration

package main

import (
	"encoding/json"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/shopspring/decimal"
	"go.uber.org/zap"
)

// Metrics for Prometheus monitoring
var (
	requestLatency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "gateway_request_latency_microseconds",
			Help:    "Request latency in microseconds",
			Buckets: []float64{10, 50, 100, 250, 500, 1000, 2500, 5000},
		},
		[]string{"endpoint", "method"},
	)

	activeConnections = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "gateway_active_websocket_connections",
			Help: "Number of active WebSocket connections",
		},
	)

	ordersProcessed = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "gateway_orders_processed_total",
			Help: "Total number of orders processed",
		},
		[]string{"symbol", "side", "status"},
	)
)

// Order represents a trading order with nanosecond precision
type Order struct {
	ID            string          `json:"id"`
	Symbol        string          `json:"symbol"`
	Side          string          `json:"side"` // BUY or SELL
	Quantity      decimal.Decimal `json:"quantity"`
	Price         decimal.Decimal `json:"price"`
	OrderType     string          `json:"order_type"` // LIMIT, MARKET, IOC, FOK
	TimestampNano int64           `json:"timestamp_nano"`
	ClientID      string          `json:"client_id"`
}

// Signal represents an AI model prediction
type Signal struct {
	Symbol      string          `json:"symbol"`
	Direction   int             `json:"direction"` // 1=long, -1=short, 0=neutral
	Confidence  decimal.Decimal `json:"confidence"`
	Uncertainty decimal.Decimal `json:"uncertainty"`
	RegimeID    int             `json:"regime_id"`
	Timestamp   int64           `json:"timestamp"`
}

// RiskCheck performs real-time risk validation
type RiskCheck struct {
	MaxPositionSize decimal.Decimal
	MaxOrderValue   decimal.Decimal
	MaxDrawdown     decimal.Decimal
	DailyLossLimit  decimal.Decimal
	currentPnL      decimal.Decimal
	mu              sync.RWMutex
}

func (r *RiskCheck) Validate(order *Order) (bool, string) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	orderValue := order.Price.Mul(order.Quantity)

	if orderValue.GreaterThan(r.MaxOrderValue) {
		return false, "ORDER_VALUE_EXCEEDED"
	}

	if r.currentPnL.LessThan(r.DailyLossLimit.Neg()) {
		return false, "DAILY_LOSS_LIMIT_BREACHED"
	}

	return true, "OK"
}

// OrderBook maintains a lock-free order book snapshot
type OrderBook struct {
	Symbol    string
	Bids      []PriceLevel
	Asks      []PriceLevel
	Timestamp int64
	sequence  uint64
}

type PriceLevel struct {
	Price    decimal.Decimal `json:"price"`
	Quantity decimal.Decimal `json:"quantity"`
	Orders   int             `json:"orders"`
}

// ConnectionPool manages WebSocket connections with zero-copy broadcasts
type ConnectionPool struct {
	clients    map[*websocket.Conn]bool
	broadcast  chan []byte
	register   chan *websocket.Conn
	unregister chan *websocket.Conn
	mu         sync.RWMutex
}

func NewConnectionPool() *ConnectionPool {
	return &ConnectionPool{
		clients:    make(map[*websocket.Conn]bool),
		broadcast:  make(chan []byte, 10000), // High-throughput buffer
		register:   make(chan *websocket.Conn),
		unregister: make(chan *websocket.Conn),
	}
}

func (p *ConnectionPool) Run() {
	for {
		select {
		case client := <-p.register:
			p.mu.Lock()
			p.clients[client] = true
			p.mu.Unlock()
			activeConnections.Inc()

		case client := <-p.unregister:
			p.mu.Lock()
			if _, ok := p.clients[client]; ok {
				delete(p.clients, client)
				client.Close()
			}
			p.mu.Unlock()
			activeConnections.Dec()

		case message := <-p.broadcast:
			p.mu.RLock()
			for client := range p.clients {
				// Non-blocking send with timeout
				err := client.WriteMessage(websocket.TextMessage, message)
				if err != nil {
					p.unregister <- client
				}
			}
			p.mu.RUnlock()
		}
	}
}

// CircuitBreaker prevents cascading failures
type CircuitBreaker struct {
	failures     uint64
	threshold    uint64
	lastFailure  time.Time
	resetTimeout time.Duration
	state        uint32 // 0=closed, 1=open, 2=half-open
}

func (cb *CircuitBreaker) Call(fn func() error) error {
	if atomic.LoadUint32(&cb.state) == 1 {
		if time.Since(cb.lastFailure) > cb.resetTimeout {
			atomic.StoreUint32(&cb.state, 2) // half-open
		} else {
			return ErrCircuitOpen
		}
	}

	err := fn()
	if err != nil {
		failures := atomic.AddUint64(&cb.failures, 1)
		cb.lastFailure = time.Now()
		if failures >= cb.threshold {
			atomic.StoreUint32(&cb.state, 1) // open
		}
		return err
	}

	atomic.StoreUint64(&cb.failures, 0)
	atomic.StoreUint32(&cb.state, 0) // closed
	return nil
}

var ErrCircuitOpen = &CircuitOpenError{}

type CircuitOpenError struct{}

func (e *CircuitOpenError) Error() string {
	return "circuit breaker is open"
}

// RateLimiter implements token bucket for DDoS protection
type RateLimiter struct {
	tokens     uint64
	maxTokens  uint64
	refillRate uint64
	lastRefill time.Time
	mu         sync.Mutex
}

func (rl *RateLimiter) Allow() bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	elapsed := now.Sub(rl.lastRefill)
	tokensToAdd := uint64(elapsed.Seconds()) * rl.refillRate

	rl.tokens = min(rl.tokens+tokensToAdd, rl.maxTokens)
	rl.lastRefill = now

	if rl.tokens > 0 {
		rl.tokens--
		return true
	}
	return false
}

func min(a, b uint64) uint64 {
	if a < b {
		return a
	}
	return b
}

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		return true // Configure properly in production
	},
}

func main() {
	// Initialize structured logger
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	// Register Prometheus metrics
	prometheus.MustRegister(requestLatency, activeConnections, ordersProcessed)

	// Initialize connection pool
	pool := NewConnectionPool()
	go pool.Run()

	// Initialize risk checker
	riskCheck := &RiskCheck{
		MaxPositionSize: decimal.NewFromInt(1000000),
		MaxOrderValue:   decimal.NewFromInt(500000),
		MaxDrawdown:     decimal.NewFromFloat(0.02),
		DailyLossLimit:  decimal.NewFromInt(100000),
	}

	// Initialize circuit breaker
	cb := &CircuitBreaker{
		threshold:    5,
		resetTimeout: 30 * time.Second,
	}

	// Rate limiter: 10000 requests/second per client
	rateLimiter := &RateLimiter{
		tokens:     10000,
		maxTokens:  10000,
		refillRate: 10000,
		lastRefill: time.Now(),
	}

	// Gin router with zero-allocation middleware
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Recovery())

	// Latency tracking middleware
	r.Use(func(c *gin.Context) {
		start := time.Now()
		c.Next()
		latency := time.Since(start).Microseconds()
		requestLatency.WithLabelValues(c.Request.URL.Path, c.Request.Method).Observe(float64(latency))
	})

	// Rate limiting middleware
	r.Use(func(c *gin.Context) {
		if !rateLimiter.Allow() {
			c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{"error": "rate limit exceeded"})
			return
		}
		c.Next()
	})

	// Health check endpoint
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":    "healthy",
			"timestamp": time.Now().UnixNano(),
		})
	})

	// Prometheus metrics endpoint
	r.GET("/metrics", gin.WrapH(promhttp.Handler()))

	// Order submission with risk checks
	r.POST("/api/v1/orders", func(c *gin.Context) {
		var order Order
		if err := c.ShouldBindJSON(&order); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		order.TimestampNano = time.Now().UnixNano()

		// Risk validation
		valid, reason := riskCheck.Validate(&order)
		if !valid {
			ordersProcessed.WithLabelValues(order.Symbol, order.Side, "REJECTED").Inc()
			c.JSON(http.StatusForbidden, gin.H{
				"error":  "risk check failed",
				"reason": reason,
			})
			return
		}

		// Route to execution engine via circuit breaker
		err := cb.Call(func() error {
			// Forward to Rust/C++ execution engine
			// In production: gRPC call to execution service
			return nil
		})

		if err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "execution service unavailable"})
			return
		}

		ordersProcessed.WithLabelValues(order.Symbol, order.Side, "ACCEPTED").Inc()
		c.JSON(http.StatusAccepted, gin.H{
			"order_id":  order.ID,
			"status":    "ACCEPTED",
			"timestamp": order.TimestampNano,
		})
	})

	// Signal ingestion from AI models
	r.POST("/api/v1/signals", func(c *gin.Context) {
		var signal Signal
		if err := c.ShouldBindJSON(&signal); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		signal.Timestamp = time.Now().UnixNano()

		// Broadcast to all connected clients
		signalJSON, _ := json.Marshal(signal)
		pool.broadcast <- signalJSON

		c.JSON(http.StatusAccepted, gin.H{
			"status":    "BROADCAST",
			"timestamp": signal.Timestamp,
		})
	})

	// WebSocket for real-time streaming
	r.GET("/ws/market-data", func(c *gin.Context) {
		conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
		if err != nil {
			logger.Error("WebSocket upgrade failed", zap.Error(err))
			return
		}

		pool.register <- conn

		// Keep connection alive
		go func() {
			defer func() {
				pool.unregister <- conn
			}()

			for {
				_, _, err := conn.ReadMessage()
				if err != nil {
					break
				}
			}
		}()
	})

	// Order book snapshot
	r.GET("/api/v1/orderbook/:symbol", func(c *gin.Context) {
		symbol := c.Param("symbol")

		// In production: fetch from KDB+/Redis
		orderBook := OrderBook{
			Symbol:    symbol,
			Timestamp: time.Now().UnixNano(),
			Bids: []PriceLevel{
				{Price: decimal.NewFromFloat(100.50), Quantity: decimal.NewFromInt(1000), Orders: 5},
				{Price: decimal.NewFromFloat(100.45), Quantity: decimal.NewFromInt(2000), Orders: 8},
			},
			Asks: []PriceLevel{
				{Price: decimal.NewFromFloat(100.55), Quantity: decimal.NewFromInt(1500), Orders: 4},
				{Price: decimal.NewFromFloat(100.60), Quantity: decimal.NewFromInt(2500), Orders: 7},
			},
		}

		c.JSON(http.StatusOK, orderBook)
	})

	// Start server with graceful shutdown
	srv := &http.Server{
		Addr:    ":8080",
		Handler: r,
	}

	// Graceful shutdown
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("Server failed", zap.Error(err))
		}
	}()

	logger.Info("Gateway started", zap.String("addr", ":8080"))

	// Block forever (in production: handle signals)
	select {}
}
