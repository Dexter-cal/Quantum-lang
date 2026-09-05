# Design Reference Guide

This folder maps each implementation component to the Quantum language 115-section design specification.

Every file in `src/qai/` references one or more sections below.

---

## Architecture Components

### Core Computational Unit
- **File:** `src/qai/tensor.rs`
- **Reference Sections:**
  - Section 1: Layers of AI Development (Tensor is the foundational layer)
  - Section 18: Hardware Awareness (auto-adapt batch sizes)
  - Section 79: Weight Table View (tensors as readable structures)
- **What it does:** N-dimensional arrays with 40+ operations (add, matmul, activations, etc.)
- **Why it matters:** Every ML operation flows through tensors

### Open Interface Pattern
- **File:** `src/qai/technique.rs`
- **Reference Sections:**
  - Section 3: Core Language/Framework Philosophy ("nothing hardcoded")
  - Section 56: Type/Technique Authoring System (open trait)
  - Section 70: Open Logic Authoring (no hardcoded menus)
- **What it does:** Trait that every ML algorithm must implement
- **Why it matters:** Enables extensibility without special cases

```rust
pub trait Technique {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor>;
    fn objective(&self, output: &Tensor, target: &Tensor) -> Result<Signal>;
    fn update(&mut self, signal: &Signal, learning_rate: f64) -> Result<()>;
}
```

Built-ins (Regression, CNN, Transformer) are just pre-shipped implementations.
Users write custom techniques the exact same way.

### Regression Technique (First Technique)
- **File:** `src/qai/regression.rs`
- **Reference Sections:**
  - Section 13: Worked Example — Sentiment Classifier (but simplified for MVP)
  - Section 57: Formula Reference Library (y = Xw + b)
  - Section 63: Optimization (pruning, quantization)
- **What it does:** Linear regression with MSE loss and SGD update
- **Why it matters:** First proof-of-concept that the Technique interface works
- **Formula:** `y = X·w + b`

### Optimizers
- **File:** `src/qai/optimizer.rs`
- **Reference Sections:**
  - Section 99: Numerical Correctness (automatic differentiation foundation)
  - Section 103: Additional Training Strategies (SGD, Adam, SAM, etc.)
- **What it does:** Gradient-based optimizers (SGD, Adam)
- **Why it matters:** How models improve during training
- **Next:** Implement full backprop, not just simplified updates

### Loss Functions
- **File:** `src/qai/loss.rs`
- **Reference Sections:**
  - Section 5: Prediction Tracking & Fine-Tuning (loss curves)
  - Section 99: Numerical Correctness (gradient computation)
- **What it does:** MSE, MAE, CrossEntropy (implementations vary by use case)
- **Why it matters:** Different losses for different problems

### Data Utilities
- **File:** `src/qai/data.rs`
- **Reference Sections:**
  - Section 12: Data Pipeline & Deployment Gaps (streaming, versioning)
  - Section 111: Reproducibility, Splitting & Sharing Details
- **What it does:** Dataset wrapper, synthetic data generation
- **Why it matters:** Clean data handling prevents silent bugs

---

## Roadmap (Phases)

### Phase 1: Foundation ✅ DONE
- [x] Tensor implementation
- [x] Technique interface
- [x] Regression (first technique)
- [x] MSE loss
- [x] SGD optimizer
- [x] Simple CLI

### Phase 2: Extensibility (Next)
- [ ] CNN as second Technique (prove modularity)
- [ ] Transformer as third Technique
- [ ] Logic blocks (basic control flow)
- [ ] Bridge interface (model-to-model communication)
- [ ] Experiment tracking

### Phase 3: Intelligence
- [ ] MoE (Mixture of Experts)
- [ ] Diffusion models
- [ ] Self-play system (multi-role training)
- [ ] Mutable weights via logic
- [ ] Skills system (attachable capabilities)

### Phase 4: Production
- [ ] Serving infrastructure
- [ ] Export formats (.qmodel, GGUF, WASM)
- [ ] Marketplace/registry
- [ ] Federated learning
- [ ] Chip/ASIC deployment

---

## Design Principles

### Principle 1: Open Interface Everywhere
**Quote from Section 31:** "The framework is not hardcoded — techniques, layers, bridges, optimizers, dashboards, and guardrails all route through the same open interface pattern."

✅ Implemented so far:
- Technique trait — any algorithm can implement it
- Optimizer trait — any gradient-based method can plug in
- Loss trait — any loss function fits the interface

🔄 Coming next:
- Bridge trait — any model-to-model communication
- Environment trait — any execution context (VM, robot, game engine)
- Tool trait — any external action

### Principle 2: Nothing is Hardcoded
**Quote from Section 3:** "Nothing is hardcoded — every technique, layer, bridge, optimizer, and visualization is defined via an open interface, not baked into the compiler."

✅ In Regression:
- No special treatment in compiler
- Uses same `forward()`, `objective()`, `update()` as any technique
- New algorithms don't require compiler changes

### Principle 3: Interpretability by Default
**Quote from Section 4:** "Interpretability by default — every model type supports inspection/explanation out of the box."

🔄 Planned for Phase 2:
- `.explain()` on every prediction
- Visualization of decision traces
- Weight-to-concept translation
- Confidence-annotated outputs

---

## How to Add a New Technique

1. Create `src/qai/myalgorithm.rs`
2. Implement `Technique` trait:
   ```rust
   impl Technique for MyAlgorithm {
       fn forward(&mut self, input: &Tensor) -> Result<Tensor> { ... }
       fn objective(&self, output: &Tensor, target: &Tensor) -> Result<Signal> { ... }
       fn update(&mut self, signal: &Signal, lr: f64) -> Result<()> { ... }
   }
   ```
3. Add to `src/qai/mod.rs`:
   ```rust
   pub mod myalgorithm;
   pub use myalgorithm::MyAlgorithm;
   ```
4. Write tests (same as Regression)
5. Update this README
6. That's it! No compiler changes needed.

---

## Testing

Every module includes unit tests:

```bash
# Run all tests
cargo test

# Run specific module tests
cargo test qai::tensor
cargo test qai::regression

# Run with output
cargo test -- --nocapture
```

---

## Current Limitations (MVP)

❌ Not yet implemented:
- Full backpropagation (currently simplified)
- GPU support
- Distributed training
- Logic blocks
- Skills system
- Serving infrastructure
- Dashboard/visualization

✅ Proven to work:
- Modular structure
- Open interfaces
- Training end-to-end
- Loss convergence
- Multiple algorithms (ready for CNN next)

---

## References

- **Full Design:** See `docs/QUANTUM_FULL_DESIGN.md` (115 sections)
- **Language Reference:** See `docs/LANGUAGE_REFERENCE.md`
- **ML Framework Reference:** See `docs/QUANTUMAI_REFERENCE.md`

---

**Next step:** Add CNN as second Technique, then Logic blocks for control flow.
