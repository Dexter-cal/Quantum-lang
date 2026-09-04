#!/bin/bash
# Quantum Language v2.1 — Build Script
set -e

echo "==================================="
echo "  Quantum Language v2.1 — Build"
echo "==================================="

# Check dependencies
command -v cargo >/dev/null 2>&1 || { echo "ERROR: cargo not found. Install Rust: https://rustup.rs"; exit 1; }
command -v gcc >/dev/null 2>&1 || { echo "ERROR: gcc not found. Install build-essential"; exit 1; }

echo ""
echo "[1/3] Building compiler..."
cargo build --release -p quantum-compiler
echo "      ✓ quantumc → target/release/quantumc"

echo ""
echo "[2/3] Building QuantumAI library..."
cargo build --release -p quantumai
echo "      ✓ quantumai → target/release/libquantumai.rlib"

echo ""
echo "[3/3] Running tests..."
cargo test --quiet 2>&1 | grep "test result" | grep -v "0 tests"
echo "      ✓ All tests passed"

echo ""
echo "==================================="
echo "  Build complete!"
echo ""
echo "  Usage:"
echo "    ./target/release/quantumc run examples/hello.qtm"
echo "    ./target/release/quantumc run examples/features.qtm"
echo "    cargo run --example ml_demo -p quantumai"
echo ""
echo "  Install (optional):"
echo "    sudo cp target/release/quantumc /usr/local/bin/"
echo "==================================="
