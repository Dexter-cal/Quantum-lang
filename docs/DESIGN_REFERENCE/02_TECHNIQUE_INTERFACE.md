# Technique Interface (Sections 3, 56, 70)

The core abstraction enabling the open-interface pattern.

## The Pattern

Instead of hardcoding separate paths for Regression, CNN, Transformer, MoE, etc., we define ONE interface:

```rust
pub trait Technique: Send + Sync {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor>;
    fn objective(&self, output: &Tensor, target: &Tensor) -> Result<Signal>;
    fn update(&mut self, signal: &Signal, learning_rate: f64) -> Result<()>;
    fn name(&self) -> &str;
    fn description(&self) -> &str { ... }
    fn save_weights(&self, path: &str) -> Result<()> { ... }
    fn load_weights(&mut self, path: &str) -> Result<()> { ... }
}
```

## Signal Type

Generic feedback from an objective function:

```rust
pub struct Signal {
    pub loss: f64,              // Could be MSE, cross-entropy, reward, etc.
    pub gradients: Tensor,      // For gradient-based optimization
}
```

Note: "loss" is a misnomer for some techniques:
- Regression: actual loss (lower is better)
- RL: reward signal (higher is better)
- Diffusion: denoising error
- GAN: competing losses

## Built-in Implementations

### Regression (Section 57)
```rust
impl Technique for Regression {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // y = X·w + b
    }
    
    fn objective(&self, output: &Tensor, target: &Tensor) -> Result<Signal> {
        // MSE: (1/n) * sum((y_pred - y_true)^2)
    }
    
    fn update(&mut self, signal: &Signal, lr: f64) -> Result<()> {
        // w = w - lr * gradient
    }
}
```

## How to Implement Your Own

### Step 1: Define the Algorithm
```rust
pub struct MyCNN {
    conv_weights: Tensor,
    fc_weights: Tensor,
    // ... other parameters
}
```

### Step 2: Implement Technique
```rust
impl Technique for MyCNN {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // Convolution + ReLU + pooling + dense layer
    }
    
    fn objective(&self, output: &Tensor, target: &Tensor) -> Result<Signal> {
        // Compute loss (MSE or CrossEntropy)
    }
    
    fn update(&mut self, signal: &Signal, lr: f64) -> Result<()> {
        // Update conv_weights and fc_weights
    }
    
    fn name(&self) -> &str { "MyCNN" }
    fn description(&self) -> &str { "Convolutional neural network" }
}
```

### Step 3: No Compiler Changes Needed
Just add it to `src/qai/` and export from `src/qai/mod.rs`. The framework treats it exactly like Regression.

## Why This Works

**Problem (before open interfaces):**
- Add Regression → compiler knows about it
- Add CNN → compiler updated again
- Add Transformer → compiler updated again
- Add MoE → compiler updated again
- Framework becomes bloated

**Solution (with open interfaces):**
- Define the trait ONCE
- Every algorithm implements it
- Compiler doesn't need to know about any specific one
- New algorithms don't require compiler changes

## Limitations of Current MVP

❌ Simplified `update()` — doesn't compute full backprop
❌ No automatic differentiation — gradients are hand-coded
❌ No weight parameter shapes — Tensor handles everything

✅ Proof-of-concept complete
✅ Ready for CNN next
✅ Architecture is solid
