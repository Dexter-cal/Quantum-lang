# Quantum Language Framework

**A compiled programming language with built-in AI/ML capabilities.**

This is the modular implementation of the Quantum framework, built incrementally from the ground up. See [`docs/QUANTUM_FULL_DESIGN.md`](docs/QUANTUM_FULL_DESIGN.md) for the complete 115-section design specification.

## Phase 1: Core Tensor + Regression (Current)

### What's Implemented

✅ **Tensor** — N-dimensional arrays with:
- Creation (zeros, ones, random, Xavier init)
- Operations (add, sub, mul, matmul, transpose)
- Activations (ReLU, sigmoid, softmax, tanh)
- Reductions (sum, mean, std, min, max)

✅ **Open Technique Interface** — Every ML algorithm must implement:
```rust
fn forward(&mut self, input: &Tensor) -> Result<Tensor>
fn objective(&self, output: &Tensor, target: &Tensor) -> Result<Signal>
fn update(&mut self, signal: &Signal, learning_rate: f64) -> Result<()>
```

✅ **Regression Technique** — First implementation:
- Linear model: y = Xw + b
- MSE loss function
- Gradient-based updates
- Full training pipeline

✅ **Data Utilities** — Dataset wrapper + synthetic data generation

✅ **CLI** — Simple end-to-end training demo

### Quick Start

```bash
# Build the project
cargo build --release

# Run the example
cargo run --bin qai

# Run tests
cargo test
```

### Project Structure

```
src/
├── lib.rs              # Library root
├── error.rs            # Error types
├── qai/
│   ├── mod.rs
│   ├── tensor.rs       # Tensor implementation (core)
│   ├── technique.rs    # Open interface
│   ├── regression.rs   # Linear regression (first algorithm)
│   ├── optimizer.rs    # SGD, Adam, etc.
│   ├── loss.rs         # Loss functions
│   └── data.rs         # Dataset utilities
└── bin/
    └── cli.rs          # Command-line interface
```

## Next Steps (Phase 2)

- [ ] CNN as second Technique (prove modularity)
- [ ] Logic blocks (basic composition)
- [ ] Custom Bridge implementations
- [ ] Experiment tracking
- [ ] Skills system (attachable adapters)

## Design Reference

Every component maps directly to the 115-section design spec:

- **Tensor** → Section 1 (core computational unit)
- **Technique** → Section 3 (open interface pattern)
- **Regression** → Sections 13, 57 (worked example)
- **Data** → Section 12 (pipeline)

See [`docs/DESIGN_REFERENCE/`](docs/DESIGN_REFERENCE/) for cross-references.

## Philosophy

**Nothing is hardcoded.** Every technique, layer, optimizer, and visualization is defined via an open interface. This prevents special cases and ensures the language scales cleanly as we add more features.

## Building the Framework

We're implementing this **step-by-step**, each step adding ONE small, testable component:

1. ✅ **Step 1: Tensor** — Basic N-D arrays (DONE)
2. ✅ **Step 2: Technique + Regression** — Prove the interface works (DONE)
3. **Step 3: CNN** — Second technique, test modularity
4. **Step 4: Logic blocks** — Control flow inside models
5. **Step 5: Skills** — Attachable capabilities
6. ... (continue per the 115-section design)

Each step:
- Lives in its own module
- Has clear tests
- Maps to the design spec
- Doesn't touch earlier code

## Contributing

New techniques and algorithms should:

1. Implement the `Technique` trait
2. Go in `src/qai/<technique_name>.rs`
3. Include unit tests
4. Update the README

## License

MIT
