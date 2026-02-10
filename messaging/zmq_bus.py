"""
ZeroMQ Message Bus — ultra-low-latency inter-process communication.

Implements pub/sub and request/reply patterns for connecting
the Python strategy layer, Rust execution engine, and Go API gateway.
"""

from __future__ import annotations

import zmq
import json
import time
import numpy as np
from typing import Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class MarketTick:
    symbol: str
    timestamp: float
    bid: float
    ask: float
    bid_size: int
    ask_size: int


@dataclass
class OrderSignal:
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    signal_strength: float
    timestamp: float


class ZMQPublisher:
    """Publishes market data and signals to subscribers via ZeroMQ PUB socket."""

    def __init__(self, endpoint: str = "tcp://*:5555"):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 100_000)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(endpoint)

    def publish_tick(self, tick: MarketTick) -> None:
        topic = f"tick.{tick.symbol}".encode()
        payload = json.dumps(asdict(tick)).encode()
        self.socket.send_multipart([topic, payload])

    def publish_signal(self, signal: OrderSignal) -> None:
        topic = f"signal.{signal.symbol}".encode()
        payload = json.dumps(asdict(signal)).encode()
        self.socket.send_multipart([topic, payload])

    def close(self):
        self.socket.close()
        self.ctx.term()


class ZMQSubscriber:
    """Subscribes to market data and signal topics via ZeroMQ SUB socket."""

    def __init__(self, endpoint: str = "tcp://localhost:5555", topics: Optional[list[str]] = None):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.setsockopt(zmq.RCVHWM, 100_000)
        self.socket.connect(endpoint)

        for topic in (topics or [""]):
            self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)

    def receive(self, timeout_ms: int = 1000) -> Optional[tuple[str, dict]]:
        if self.socket.poll(timeout_ms):
            topic, payload = self.socket.recv_multipart()
            return topic.decode(), json.loads(payload.decode())
        return None

    def stream(self):
        """Infinite generator yielding (topic, data) tuples."""
        while True:
            result = self.receive(timeout_ms=100)
            if result:
                yield result

    def close(self):
        self.socket.close()
        self.ctx.term()


class ZMQRequestReply:
    """Request/reply pattern for synchronous RPC between components."""

    @staticmethod
    def server(endpoint: str = "tcp://*:5556"):
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REP)
        socket.bind(endpoint)
        return ctx, socket

    @staticmethod
    def client(endpoint: str = "tcp://localhost:5556"):
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REQ)
        socket.connect(endpoint)
        return ctx, socket
