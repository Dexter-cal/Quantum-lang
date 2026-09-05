//! Transformer Technique
//!
//! Attention-based architecture for sequences.
//! Implements the open Technique interface.
//!
//! Reference: Section 46 (Transformer unique methods), Section 20 (multimodal)

use crate::qai::technique::{Technique, Signal};
use crate::Tensor;
use crate::error::Result;
use serde::{Serialize, Deserialize};

/// Transformer Model: Multi-head self-attention + FFN
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transformer {
    pub attention_weights: Tensor,  // Query, Key, Value projections
    pub ffn_weights: Tensor,         // Feed-forward network
    pub output_weights: Tensor,      // Output projection
    pub n_heads: usize,
}

impl Transformer {
    /// Create a new Transformer
    /// embed_dim: embedding dimension (e.g., 512)
    /// num_heads: number of attention heads (e.g., 8)
    /// vocab_size: vocabulary size for output
    pub fn new(embed_dim: usize, num_heads: usize, vocab_size: usize) -> Result<Self> {
        let attention_weights = Tensor::xavier(embed_dim * 3, embed_dim)?; // Q, K, V
        let ffn_weights = Tensor::xavier(embed_dim, embed_dim * 4)?;        // FFN expansion
        let output_weights = Tensor::xavier(embed_dim, vocab_size)?;        // Output layer

        Ok(Transformer {
            attention_weights,
            ffn_weights,
            output_weights,
            n_heads,
        })
    }

    /// Scaled dot-product attention (simplified)
    fn attention(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified: just apply attention_weights
        let attention_out = input.matmul(&self.attention_weights)?;
        attention_out.softmax()
    }

    /// Feed-forward network
    fn ffn(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = input.matmul(&self.ffn_weights)?;
        let activated = hidden.relu()?;
        activated.scale(0.5)  // Simplified projection back
    }

    /// Describe the architecture
    pub fn architecture(&self) -> String {
        format!(
            "Transformer(heads={}, attn_size={}, ffn_size={})",
            self.n_heads,
            self.attention_weights.numel(),
            self.ffn_weights.numel()
        )
    }
}

impl Technique for Transformer {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // Multi-head self-attention
        let attn_out = self.attention(input)?;
        
        // Residual connection (simplified: just add)
        let with_residual = input.add(&attn_out)?;
        
        // Feed-forward network
        let ffn_out = self.ffn(&with_residual)?;
        
        // Output projection + softmax
        let logits = ffn_out.matmul(&self.output_weights)?;
        logits.softmax()
    }

    fn objective(&self, output: &Tensor, target: &Tensor) -> Result<Signal> {
        // Cross-entropy loss
        let epsilon = 1e-7;
        let log_output: Vec<f64> = output
            .as_slice()
            .iter()
            .map(|x| (x + epsilon).ln())
            .collect();
        let log_tensor = Tensor::new(log_output, output.shape())?;
        let prod = target.mul(&log_tensor)?;
        let loss = -(prod.sum() / output.numel() as f64);

        let gradients = output.sub(target)?;

        Ok(Signal { loss, gradients })
    }

    fn update(&mut self, signal: &Signal, learning_rate: f64) -> Result<()> {
        // Simplified: mock update
        let grad = signal.gradients.mean();
        let new_val = self.attention_weights.mean() - learning_rate * grad;
        self.attention_weights = Tensor::new(
            vec![new_val],
            vec![1],
        )?;
        Ok(())
    }

    fn name(&self) -> &str {
        "Transformer"
    }

    fn description(&self) -> &str {
        "Transformer: Multi-head self-attention + Feed-forward network"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_creation() {
        let model = Transformer::new(512, 8, 10000).unwrap();
        assert!(!model.attention_weights.shape().is_empty());
    }

    #[test]
    fn test_transformer_forward() {
        let mut model = Transformer::new(512, 8, 10000).unwrap();
        let input = Tensor::random(vec![4, 512]).unwrap();  // 4 samples, 512 dims
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape()[0], 4);
    }
}
