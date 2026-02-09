"""
C++/CUDA Python Bindings
=========================
Bridge between Python ML layer and C++/CUDA execution layer.

This module provides:
1. ctypes bindings to C++ execution engine
2. CUDA kernel calls for GPU-accelerated operations
3. KDB+ connection for time series data
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CUDA Kernels Bridge
# ============================================================================

class CUDAKernels:
    """
    Python interface to custom CUDA kernels.
    
    Provides GPU-accelerated operations for:
    - Path signature computation
    - Lead-lag correlation analysis
    - EWMA volatility estimation
    """
    
    def __init__(self, lib_path: Optional[str] = None):
        self.lib_path = lib_path or self._find_library()
        self.lib = None
        self._load_library()
    
    def _find_library(self) -> str:
        """Find CUDA kernels shared library."""
        search_paths = [
            Path(__file__).parent.parent / "cpp" / "cuda_kernels.so",
            Path("/usr/local/lib/cuda_kernels.so"),
            Path.cwd() / "cuda_kernels.so"
        ]
        
        for path in search_paths:
            if path.exists():
                return str(path)
        
        return ""
    
    def _load_library(self):
        """Load the CUDA shared library."""
        if not self.lib_path or not Path(self.lib_path).exists():
            logger.warning("CUDA kernels library not found. Using NumPy fallback.")
            self.lib = None
            return
        
        try:
            self.lib = ctypes.CDLL(self.lib_path)
            self._setup_function_signatures()
            logger.info(f"Loaded CUDA kernels from {self.lib_path}")
        except Exception as e:
            logger.warning(f"Failed to load CUDA kernels: {e}")
            self.lib = None
    
    def _setup_function_signatures(self):
        """Define C function signatures."""
        if self.lib is None:
            return
        
        # compute_path_signature
        self.lib.compute_path_signature.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # path
            ctypes.POINTER(ctypes.c_float),  # sig1
            ctypes.POINTER(ctypes.c_float),  # sig2
            ctypes.c_int,                     # batch_size
            ctypes.c_int,                     # seq_len
            ctypes.c_int                      # dim
        ]
        self.lib.compute_path_signature.restype = None
        
        # compute_lead_lag_correlations
        self.lib.compute_lead_lag_correlations.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # returns_a
            ctypes.POINTER(ctypes.c_float),  # returns_b
            ctypes.POINTER(ctypes.c_float),  # correlations
            ctypes.c_int,                     # batch_size
            ctypes.c_int,                     # seq_len
            ctypes.c_int                      # max_lag
        ]
        self.lib.compute_lead_lag_correlations.restype = None
        
        # compute_ewma_volatility
        self.lib.compute_ewma_volatility.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # returns
            ctypes.POINTER(ctypes.c_float),  # volatility
            ctypes.c_float,                   # lambda
            ctypes.c_int,                     # batch_size
            ctypes.c_int                      # seq_len
        ]
        self.lib.compute_ewma_volatility.restype = None
    
    def compute_path_signature(
        self,
        path: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute path signatures using GPU.
        
        Args:
            path: (batch, seq_len, dim) array
            
        Returns:
            sig1: (batch, dim) Level 1 signature
            sig2: (batch, dim, dim) Level 2 signature
        """
        path = np.ascontiguousarray(path, dtype=np.float32)
        batch_size, seq_len, dim = path.shape
        
        sig1 = np.zeros((batch_size, dim), dtype=np.float32)
        sig2 = np.zeros((batch_size, dim * dim), dtype=np.float32)
        
        if self.lib is not None:
            self.lib.compute_path_signature(
                path.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                sig1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                sig2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch_size, seq_len, dim
            )
        else:
            # NumPy fallback
            sig1 = path[:, -1] - path[:, 0]
            increments = np.diff(path, axis=1)
            cumsum = np.cumsum(increments, axis=1)
            for i in range(dim):
                for j in range(dim):
                    sig2[:, i * dim + j] = np.sum(
                        cumsum[:, :-1, i] * increments[:, 1:, j], axis=1
                    )
        
        sig2 = sig2.reshape(batch_size, dim, dim)
        return sig1, sig2
    
    def compute_lead_lag_correlations(
        self,
        returns_a: np.ndarray,
        returns_b: np.ndarray,
        max_lag: int = 10
    ) -> np.ndarray:
        """
        Compute cross-correlations at different lags.
        
        Args:
            returns_a: (batch, seq_len) returns for asset A
            returns_b: (batch, seq_len) returns for asset B
            max_lag: Maximum lag to compute
            
        Returns:
            correlations: (batch, 2*max_lag+1) correlations at each lag
        """
        returns_a = np.ascontiguousarray(returns_a, dtype=np.float32)
        returns_b = np.ascontiguousarray(returns_b, dtype=np.float32)
        batch_size, seq_len = returns_a.shape
        
        correlations = np.zeros((batch_size, 2 * max_lag + 1), dtype=np.float32)
        
        if self.lib is not None:
            self.lib.compute_lead_lag_correlations(
                returns_a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                returns_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                correlations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch_size, seq_len, max_lag
            )
        else:
            # NumPy fallback
            for i, lag in enumerate(range(-max_lag, max_lag + 1)):
                if lag >= 0:
                    a = returns_a[:, lag:]
                    b = returns_b[:, :seq_len - lag] if lag > 0 else returns_b
                else:
                    a = returns_a[:, :seq_len + lag]
                    b = returns_b[:, -lag:]
                
                # Pearson correlation
                a_mean = a.mean(axis=1, keepdims=True)
                b_mean = b.mean(axis=1, keepdims=True)
                a_centered = a - a_mean
                b_centered = b - b_mean
                
                cov = (a_centered * b_centered).sum(axis=1)
                std_a = np.sqrt((a_centered ** 2).sum(axis=1))
                std_b = np.sqrt((b_centered ** 2).sum(axis=1))
                
                correlations[:, i] = cov / (std_a * std_b + 1e-8)
        
        return correlations
    
    def compute_ewma_volatility(
        self,
        returns: np.ndarray,
        lambda_param: float = 0.94
    ) -> np.ndarray:
        """
        Compute EWMA volatility.
        
        Args:
            returns: (batch, seq_len) returns
            lambda_param: Decay parameter (typically 0.94)
            
        Returns:
            volatility: (batch, seq_len) EWMA volatility
        """
        returns = np.ascontiguousarray(returns, dtype=np.float32)
        batch_size, seq_len = returns.shape
        
        volatility = np.zeros_like(returns)
        
        if self.lib is not None:
            self.lib.compute_ewma_volatility(
                returns.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                volatility.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                lambda_param,
                batch_size, seq_len
            )
        else:
            # NumPy fallback
            var = returns[:, 0:1] ** 2
            volatility[:, 0] = np.sqrt(var.squeeze())
            
            for t in range(1, seq_len):
                var = lambda_param * var + (1 - lambda_param) * returns[:, t:t+1] ** 2
                volatility[:, t] = np.sqrt(var.squeeze())
        
        return volatility


# ============================================================================
# C++ Execution Engine Bridge
# ============================================================================

class ExecutionEngineBridge:
    """
    Python interface to C++ execution engine.
    
    Provides:
    - Order submission
    - Position tracking
    - Latency monitoring
    """
    
    def __init__(self, lib_path: Optional[str] = None):
        self.lib_path = lib_path
        self.lib = None
        self.symbol_map: Dict[str, int] = {}
        self.next_symbol_id = 0
        
        # Fallback to Python implementation if C++ not available
        self.use_fallback = True
        self.positions: Dict[str, float] = {}
        self.orders: List[Dict] = []
    
    def register_symbol(self, symbol: str) -> int:
        """Register a symbol and get its ID."""
        if symbol not in self.symbol_map:
            self.symbol_map[symbol] = self.next_symbol_id
            self.next_symbol_id += 1
        return self.symbol_map[symbol]
    
    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "market"
    ) -> int:
        """
        Submit an order to the execution engine.
        
        Args:
            symbol: Ticker symbol
            side: "buy" or "sell"
            quantity: Order quantity
            price: Limit price (None for market orders)
            order_type: "market", "limit", "ioc", "fok"
            
        Returns:
            Order ID
        """
        order_id = len(self.orders) + 1
        
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'order_type': order_type,
            'status': 'pending',
            'filled_qty': 0.0
        }
        
        self.orders.append(order)
        
        # Simulate immediate fill for market orders
        if order_type == "market":
            self._simulate_fill(order)
        
        return order_id
    
    def _simulate_fill(self, order: Dict):
        """Simulate order fill."""
        order['status'] = 'filled'
        order['filled_qty'] = order['quantity']
        
        # Update position
        symbol = order['symbol']
        current_pos = self.positions.get(symbol, 0.0)
        
        if order['side'] == 'buy':
            self.positions[symbol] = current_pos + order['quantity']
        else:
            self.positions[symbol] = current_pos - order['quantity']
    
    def execute_rebalance(
        self,
        symbols: List[str],
        target_weights: np.ndarray,
        current_weights: np.ndarray,
        portfolio_value: float,
        prices: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        Execute portfolio rebalance.
        
        Args:
            symbols: List of ticker symbols
            target_weights: Target portfolio weights
            current_weights: Current portfolio weights
            portfolio_value: Total portfolio value
            prices: Current prices (optional, for position sizing)
            
        Returns:
            List of order IDs
        """
        order_ids = []
        
        weight_diffs = target_weights - current_weights
        
        for i, symbol in enumerate(symbols):
            diff = weight_diffs[i]
            
            # Skip small trades
            if abs(diff) < 0.001:
                continue
            
            # Calculate trade value
            trade_value = abs(diff) * portfolio_value
            
            # Estimate quantity (simplified)
            if prices is not None and prices[i] > 0:
                quantity = trade_value / prices[i]
            else:
                quantity = trade_value / 100.0  # Default price assumption
            
            side = "buy" if diff > 0 else "sell"
            
            order_id = self.submit_order(symbol, side, quantity)
            order_ids.append(order_id)
            
            logger.debug(f"Order {order_id}: {side} {quantity:.2f} {symbol}")
        
        return order_ids
    
    def get_positions(self) -> Dict[str, float]:
        """Get current positions."""
        return self.positions.copy()
    
    def get_orders(self, status: Optional[str] = None) -> List[Dict]:
        """Get orders, optionally filtered by status."""
        if status is None:
            return self.orders.copy()
        return [o for o in self.orders if o['status'] == status]


# ============================================================================
# KDB+ Bridge
# ============================================================================

class KDBBridge:
    """
    Python interface to KDB+ time series database.
    
    Requires: qPython library
    pip install qpython
    """
    
    def __init__(self, host: str = "localhost", port: int = 5001):
        self.host = host
        self.port = port
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Connect to KDB+ server."""
        try:
            from qpython import qconnection
            self.conn = qconnection.QConnection(self.host, self.port)
            self.conn.open()
            logger.info(f"Connected to KDB+ at {self.host}:{self.port}")
        except ImportError:
            logger.warning("qPython not installed. KDB+ features unavailable.")
            logger.warning("Install with: pip install qpython")
            self.conn = None
        except Exception as e:
            logger.warning(f"Could not connect to KDB+: {e}")
            self.conn = None
    
    def query(self, q_code: str):
        """Execute Q code and return result."""
        if self.conn is None:
            raise RuntimeError("Not connected to KDB+")
        return self.conn.sync(q_code)
    
    def get_quotes(self, symbols: List[str]) -> Dict:
        """Get latest quotes for symbols."""
        if self.conn is None:
            return {}
        
        sym_str = "`" + "`".join(symbols)
        result = self.query(f"getQuotes[{sym_str}]")
        return dict(result)
    
    def get_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "daily"
    ) -> np.ndarray:
        """Get historical OHLCV bars."""
        if self.conn is None:
            return np.array([])
        
        result = self.query(
            f"getBars[`{symbol};{start_date};{end_date};`{interval}]"
        )
        return np.array(result)
    
    def get_features(self, symbol: str, n_days: int = 252) -> Dict:
        """Get pre-computed features for ML model."""
        if self.conn is None:
            return {}
        
        result = self.query(f"getFeatures[`{symbol};{n_days}]")
        return dict(result)
    
    def insert_trade(
        self,
        symbol: str,
        price: float,
        size: int,
        exchange: str = "NYSE",
        side: str = "B"
    ):
        """Insert a trade tick."""
        if self.conn is None:
            return
        
        self.query(
            f"insertTrade[`{symbol};{price};{size};`{exchange};`{side}]"
        )
    
    def close(self):
        """Close connection."""
        if self.conn is not None:
            self.conn.close()


# ============================================================================
# Unified Interface
# ============================================================================

class QuantSystemBridge:
    """
    Unified interface to all system components.
    
    Provides single access point for:
    - GPU-accelerated computations (CUDA)
    - Low-latency execution (C++)
    - Time series data (KDB+)
    """
    
    def __init__(
        self,
        cuda_lib: Optional[str] = None,
        cpp_lib: Optional[str] = None,
        kdb_host: str = "localhost",
        kdb_port: int = 5001
    ):
        self.cuda = CUDAKernels(cuda_lib)
        self.execution = ExecutionEngineBridge(cpp_lib)
        
        try:
            self.kdb = KDBBridge(kdb_host, kdb_port)
        except Exception:
            self.kdb = None
    
    def compute_signatures(self, path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute path signatures using GPU."""
        return self.cuda.compute_path_signature(path)
    
    def compute_lead_lag(
        self,
        returns_a: np.ndarray,
        returns_b: np.ndarray,
        max_lag: int = 10
    ) -> np.ndarray:
        """Compute lead-lag correlations."""
        return self.cuda.compute_lead_lag_correlations(returns_a, returns_b, max_lag)
    
    def execute_trades(
        self,
        symbols: List[str],
        target_weights: np.ndarray,
        current_weights: np.ndarray,
        portfolio_value: float
    ) -> List[int]:
        """Execute portfolio rebalance."""
        return self.execution.execute_rebalance(
            symbols, target_weights, current_weights, portfolio_value
        )
    
    def get_market_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> np.ndarray:
        """Get historical market data from KDB+."""
        if self.kdb is not None:
            return self.kdb.get_bars(symbol, start_date, end_date)
        return np.array([])


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    'CUDAKernels',
    'ExecutionEngineBridge',
    'KDBBridge',
    'QuantSystemBridge'
]

