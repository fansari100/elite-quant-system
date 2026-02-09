# Go API Gateway

Ultra-low latency API gateway for the Elite Quant System.

## Features

- **High-throughput WebSocket streaming** for real-time market data
- **Circuit breaker pattern** for fault tolerance
- **Rate limiting** with token bucket algorithm
- **Prometheus metrics** for observability
- **gRPC backend communication** with execution engines

## Build & Run

```bash
cd go
go mod download
go build -o gateway ./cmd/gateway
./gateway
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |
| `/api/v1/orders` | POST | Submit order |
| `/api/v1/signals` | POST | Receive AI signals |
| `/api/v1/orderbook/:symbol` | GET | Order book snapshot |
| `/ws/market-data` | WS | Real-time streaming |

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Clients   │────▶│   Gateway   │────▶│  Execution  │
│  (WebSocket)│     │    (Go)     │     │  (Rust/C++) │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ AI Inference│
                    │  (Python)   │
                    └─────────────┘
```

## Performance

- Target latency: < 100μs p99
- Throughput: 100,000+ req/sec
- Concurrent WebSocket connections: 10,000+

