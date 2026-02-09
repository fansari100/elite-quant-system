# Elite Quant System - Docker Image
# ==================================
# Multi-stage build for production deployment
# Optimized for NVIDIA H200 GPU

# ============================================================================
# Stage 1: Python dependencies
# ============================================================================
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 AS python-base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ============================================================================
# Stage 2: Rust build
# ============================================================================
FROM rust:1.75 AS rust-builder

WORKDIR /build

# Copy Rust project
COPY rust/Cargo.toml rust/Cargo.lock* ./
COPY rust/src ./src

# Build release binary
RUN cargo build --release

# ============================================================================
# Stage 3: C++ build
# ============================================================================
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 AS cpp-builder

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy C++ source
COPY cpp/ ./cpp/

# Build execution engine
WORKDIR /build/cpp
RUN g++ -O3 -march=native -std=c++20 -pthread execution_engine.cpp -o execution_engine

# Build CUDA kernels (if NVCC available)
RUN nvcc -O3 -arch=sm_90 --shared -Xcompiler -fPIC cuda_kernels.cu -o cuda_kernels.so 2>/dev/null || echo "CUDA build skipped"

# ============================================================================
# Stage 4: Final image
# ============================================================================
FROM python-base AS final

LABEL maintainer="Elite Quant Team"
LABEL description="Elite Quant System - Production Image"
LABEL version="1.0.0"

WORKDIR /app

# Copy Python requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy built binaries from previous stages
COPY --from=rust-builder /build/target/release/execution_server /usr/local/bin/
COPY --from=cpp-builder /build/cpp/execution_engine /usr/local/bin/
COPY --from=cpp-builder /build/cpp/cuda_kernels.so /usr/local/lib/ 2>/dev/null || true

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -s /bin/bash quant && chown -R quant:quant /app
USER quant

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "main.py", "demo"]

# ============================================================================
# Alternative targets
# ============================================================================

# Training target
FROM final AS training
CMD ["python", "main.py", "train"]

# API server target
FROM final AS api
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api:create_app", "--host", "0.0.0.0", "--port", "8000"]

# Backtest target
FROM final AS backtest
CMD ["python", "main.py", "backtest"]

