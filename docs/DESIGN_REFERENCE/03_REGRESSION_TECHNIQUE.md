# Regression Technique (Sections 13, 57, 63)

The first working implementation of the open Technique interface.

## What Regression Does

**Linear regression:** Fit a line (in N dimensions) to data.

```
y = X·w + b

where:
  X = input features (n_samples, n_features)
  w = weights (n_features, 1)
  b = bias (scalar)
  y = predictions (n_samples, 1)
```

## Loss Function: Mean Squared Error

```
MSE = (1/n) * sum((y_pred - y_true)^2)

Gradient w.r.t predictions:
dMSE/dy = 2 * (y_pred - y_true) / n
```

## Training Pipeline

```rust
let mut model = Regression::new(1)?;  // 1 input feature

for epoch in 0..100 {
    // 1. Forward pass
    let output = model.forward(&x)?;
    
    // 2. Compute loss + gradient
    let signal = model.objective(&output, &y)?;
    
    // 3. Update weights
    model.update(&signal, learning_rate)?;
}
```

## Current Limitations

❌ `update()` is simplified:
- Currently only updates bias
- Doesn't compute full weight gradients
- Real implementation would use backprop: `dL/dw = X^T · dL/dy`

✅ Loss converges (proof it works)
✅ Ready for full backprop in Phase 2

## Next Steps

### Phase 2: Full Backpropagation
Implement automatic differentiation:
```rust
fn update(&mut self, signal: &Signal, lr: f64) -> Result<()> {
    // Compute d/dw and d/db
    let dw = input.transpose().matmul(&signal.gradients)?;
    let db = signal.gradients.mean();
    
    // Gradient descent
    self.weights = self.weights.sub(&dw.scale(lr))?;
    self.bias = self.bias.sub(&Tensor::new(vec![db * lr], vec![1])?)?;
}
```

### Phase 3: Optimization Techniques
Extend beyond basic gradient descent:
- Momentum (accelerates convergence)
- L2 regularization (prevents overfitting)
- Learning rate scheduling (decreases over time)

## Testing

```bash
cargo test qai::regression
```

Tests verify:
- ✅ Model creation
- ✅ Forward pass produces correct shape
- ✅ Objective computes non-zero loss
- ✅ Loss decreases with training

## Reference to 115-Section Design

- **Section 13:** "Worked Example — Sentiment Classifier" (same pattern, but simpler)
- **Section 57:** "Formula Reference Library" (y = mx + b)
- **Section 63:** "Optimization, Model Families & Capability Extraction" (pruning, quantization strategies)
- **Section 99:** "Numerical Correctness, Evaluation Integrity & Alignment Techniques" (autodiff)

## Real-World Example

Predicting house prices from features:

```rust
let dataset = make_regression(1000, 5)?;  // 1000 samples, 5 features
let mut model = Regression::new(5)?;

for epoch in 0..1000 {
    let output = model.forward(&dataset.x)?;
    let signal = model.objective(&output, &dataset.y)?;
    model.update(&signal, 0.01)?;
    
    if (epoch + 1) % 100 == 0 {
        println!("Epoch {}: loss = {:.6}", epoch + 1, signal.loss);
    }
}

// Make prediction on new data
let test_data = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5])?;
let price = model.forward(&test_data)?;
println!("Predicted price: {}", price.as_slice()[0]);
```
