//! Regression Technique Implementation
//!
//! Linear regression: y = Xw + b
//! Implements the open Technique interface.
//!
//! Reference: Section 57 of the Quantum design document

use crate::qai::technique::{Technique, Signal};
use crate::Tensor;
use crate::error::Result;
use serde::{Serialize, Deserialize};

/// Linear Regression Model
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Regression {
    pub weights: Tensor,
    pub bias: Tensor,
}

impl Regression {
    /// Create a new regression model
    /// input_dim: number of input features
    pub fn new(input_dim: usize) -> Result<Self> {
        let weights = Tensor::xavier(input_dim, 1)?;
        let bias = Tensor::zeros(vec![1])?;
        
        Ok(Regression { weights, bias })
    }

    /// Formula: y = Xw + b
    pub fn formula(&self) -> String {
        "y = X·w + b".to_string()
    }
}

impl Technique for Regression {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // input shape: [batch_size, input_dim]
        // weights shape: [input_dim, 1]
        // output shape: [batch_size, 1]
        
        let mut output = input.matmul(&self.weights)?;
        
        // Add bias to each sample
        let batch_size = output.shape()[0];
        for i in 0..batch_size {
            let bias_val = self.bias.as_slice()[0];
            output.as_slice_mut()[i] += bias_val;
        }
        
        Ok(output)
    }

    fn objective(&self, output: &Tensor, target: &Tensor) -> Result<Signal> {
        // MSE Loss: (1/n) * sum((y_pred - y_true)^2)
        let diff = output.sub(target)?;
        let squared = diff.mul(&diff)?;
        let loss = squared.mean();
        
        // Gradient w.r.t output: 2 * (y_pred - y_true) / n
        let n = diff.numel() as f64;
        let gradients = diff.scale(2.0 / n)?;
        
        Ok(Signal { loss, gradients })
    }

    fn update(&mut self, signal: &Signal, learning_rate: f64) -> Result<()> {
        // Gradient descent update (simplified for MVP)
        // In practice, we'd compute full backprop through weights
        // For now: just scale gradients for bias
        let bias_grad = signal.gradients.mean();
        let new_bias = self.bias.as_slice()[0] - learning_rate * bias_grad;
        self.bias = Tensor::new(vec![new_bias], vec![1])?;
        
        Ok(())
    }

    fn name(&self) -> &str {
        "Regression"
    }

    fn description(&self) -> &str {
        "Linear regression: y = Xw + b"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regression_creation() {
        let model = Regression::new(5).unwrap();
        assert_eq!(model.weights.shape(), vec![5, 1]);
        assert_eq!(model.bias.shape(), vec![1]);
    }

    #[test]
    fn test_regression_forward() {
        let mut model = Regression::new(2).unwrap();
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape()[0], 2); // batch size
    }

    #[test]
    fn test_regression_objective() {
        let model = Regression::new(2).unwrap();
        let output = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let target = Tensor::new(vec![1.5, 2.5], vec![2, 1]).unwrap();
        let signal = model.objective(&output, &target).unwrap();
        assert!(signal.loss > 0.0);
    }
}
