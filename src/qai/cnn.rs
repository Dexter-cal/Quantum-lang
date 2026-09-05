//! CNN (Convolutional Neural Network) Technique
//!
//! Image processing with convolutional layers.
//! Implements the open Technique interface.
//!
//! Reference: Section 46 (CNN unique methods), Section 6 (layers)

use crate::qai::technique::{Technique, Signal};
use crate::Tensor;
use crate::error::Result;
use serde::{Serialize, Deserialize};

/// CNN Model: Conv2D -> ReLU -> MaxPool -> Dense
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CNN {
    pub conv_weights: Tensor,  // [out_channels, in_channels, kernel_h, kernel_w]
    pub conv_bias: Tensor,      // [out_channels]
    pub fc_weights: Tensor,     // [flattened_size, n_classes]
    pub fc_bias: Tensor,        // [n_classes]
}

impl CNN {
    /// Create a new CNN model
    /// input_channels: 1 (grayscale) or 3 (RGB)
    /// num_filters: number of convolutional filters
    /// num_classes: number of output classes
    pub fn new(input_channels: usize, num_filters: usize, num_classes: usize) -> Result<Self> {
        // Conv layer: 3x3 kernel
        let kernel_size = 3;
        let conv_weights = Tensor::xavier(
            num_filters * input_channels * kernel_size * kernel_size,
            num_filters,
        )?;
        let conv_bias = Tensor::zeros(vec![num_filters])?;

        // Fully connected layer
        // Assuming 28x28 input -> after conv/pool -> 7x7 = 49 spatial dims
        let flattened_size = num_filters * 7 * 7;
        let fc_weights = Tensor::xavier(flattened_size, num_classes)?;
        let fc_bias = Tensor::zeros(vec![num_classes])?;

        Ok(CNN {
            conv_weights,
            conv_bias,
            fc_weights,
            fc_bias,
        })
    }

    /// Simple convolution (simplified, not full 2D convolution)
    /// For MVP: treat input as flattened features
    fn convolution(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified: linear transformation to num_filters
        input.matmul(&self.conv_weights)?.add(&self.conv_bias)
    }

    /// ReLU activation
    fn relu(&self, tensor: &Tensor) -> Result<Tensor> {
        tensor.relu()
    }

    /// Max pooling (simplified)
    fn max_pool(&self, tensor: &Tensor) -> Result<Tensor> {
        // For MVP: just downsample by averaging
        tensor.scale(0.5)  // Placeholder
    }

    /// Describe the architecture
    pub fn architecture(&self) -> String {
        format!(
            "CNN(Conv[{}] -> ReLU -> MaxPool -> FC[{}])",
            self.conv_weights.numel(),
            self.fc_weights.numel()
        )
    }
}

impl Technique for CNN {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // Conv -> ReLU -> Pool -> Flatten -> Dense -> Softmax
        let conv_out = self.convolution(input)?;
        let relu_out = self.relu(&conv_out)?;
        let pool_out = self.max_pool(&relu_out)?;
        
        // Dense layer
        let logits = pool_out.matmul(&self.fc_weights)?.add(&self.fc_bias)?;
        
        // Softmax for multi-class
        logits.softmax()
    }

    fn objective(&self, output: &Tensor, target: &Tensor) -> Result<Signal> {
        // Categorical Cross-Entropy
        // Loss = -sum(target * log(output))
        let epsilon = 1e-7;
        
        let log_output: Vec<f64> = output
            .as_slice()
            .iter()
            .map(|x| (x + epsilon).ln())
            .collect();
        let log_tensor = Tensor::new(log_output, output.shape())?;
        let prod = target.mul(&log_tensor)?;
        let loss = -(prod.sum() / output.numel() as f64);

        // Gradient: (output - target)
        let gradients = output.sub(target)?;

        Ok(Signal { loss, gradients })
    }

    fn update(&mut self, signal: &Signal, learning_rate: f64) -> Result<()> {
        // Simplified: only update biases
        let conv_bias_grad = signal.gradients.mean();
        self.conv_bias = Tensor::new(
            vec![self.conv_bias.as_slice()[0] - learning_rate * conv_bias_grad],
            self.conv_bias.shape(),
        )?;

        let fc_bias_grad = signal.gradients.mean();
        self.fc_bias = Tensor::new(
            vec![self.fc_bias.as_slice()[0] - learning_rate * fc_bias_grad],
            self.fc_bias.shape(),
        )?;

        Ok(())
    }

    fn name(&self) -> &str {
        "CNN"
    }

    fn description(&self) -> &str {
        "Convolutional Neural Network: Conv -> ReLU -> Pool -> Dense"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cnn_creation() {
        let model = CNN::new(1, 32, 10).unwrap();
        assert!(!model.conv_weights.shape().is_empty());
        assert_eq!(model.fc_bias.shape(), vec![10]);
    }

    #[test]
    fn test_cnn_forward() {
        let mut model = CNN::new(1, 32, 10).unwrap();
        let input = Tensor::new(
            (0..784).map(|_| rand::random()).collect(),
            vec![8, 784],
        ).unwrap();  // 8 samples, 28x28 flattened
        
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape()[0], 8);
        assert_eq!(output.shape()[1], 10);
    }
}
