# Elite Quant System - C++/CUDA Build Configuration
# ==================================================

# Compiler settings
CXX = g++
NVCC = nvcc

# C++ flags (optimized for production)
CXXFLAGS = -O3 -march=native -std=c++20 -pthread -Wall -Wextra
CXXFLAGS_DEBUG = -g -O0 -std=c++20 -pthread -Wall -Wextra -fsanitize=address

# CUDA flags
# For H200: sm_90a (Hopper architecture with async features)
# For H100: sm_90
# For A100: sm_80
# For V100: sm_70
CUDA_ARCH ?= sm_90
NVCCFLAGS = -O3 -arch=$(CUDA_ARCH) -std=c++17 --expt-relaxed-constexpr
NVCCFLAGS += -Xcompiler -fPIC  # Position independent code for shared library

# Shared library flags
SHARED_FLAGS = --shared -Xcompiler -fPIC

# Include paths
INCLUDES = -I/usr/local/cuda/include

# Library paths
LDFLAGS = -L/usr/local/cuda/lib64
LIBS = -lcudart -lcublas

# Targets
.PHONY: all clean debug cuda python

all: execution_engine cuda_kernels.so

# C++ Execution Engine
execution_engine: execution_engine.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
	@echo "✓ Built execution_engine"

# Debug build
execution_engine_debug: execution_engine.cpp
	$(CXX) $(CXXFLAGS_DEBUG) $< -o $@
	@echo "✓ Built execution_engine_debug"

# CUDA Kernels (shared library for Python)
cuda_kernels.so: cuda_kernels.cu
	$(NVCC) $(NVCCFLAGS) $(SHARED_FLAGS) $< -o $@ $(LDFLAGS) $(LIBS)
	@echo "✓ Built cuda_kernels.so"

# CUDA test executable
cuda_test: cuda_kernels.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)
	@echo "✓ Built cuda_test"

# Python bindings (requires pybind11)
python: cuda_kernels.so
	@echo "CUDA kernels ready for Python import"

# Clean
clean:
	rm -f execution_engine execution_engine_debug cuda_kernels.so cuda_test *.o

# Debug target
debug: execution_engine_debug

# Test targets
test_cpp: execution_engine
	./execution_engine

test_cuda: cuda_test
	./cuda_test

# Help
help:
	@echo "Elite Quant System - Build Targets"
	@echo "==================================="
	@echo ""
	@echo "make all            - Build all targets"
	@echo "make execution_engine - Build C++ execution engine"
	@echo "make cuda_kernels.so  - Build CUDA kernels (shared lib)"
	@echo "make cuda_test      - Build CUDA test executable"
	@echo "make debug          - Build with debug symbols"
	@echo "make clean          - Remove built files"
	@echo ""
	@echo "Environment variables:"
	@echo "  CUDA_ARCH=sm_90   - Target GPU architecture"
	@echo "                      sm_90a for H200/H100"
	@echo "                      sm_80 for A100"
	@echo "                      sm_70 for V100"

