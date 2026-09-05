//! RNN/LSTM Technique
//!
//! Recurrent networks for sequential data.
//! Implements the open Technique interface.
//!
//! Reference: Section 46 (RNN unique methods), Section 2 (deep learning types)

use crate::qai::technique::{Technique, Signal};
use crate::Tensor;
use crate::error::Result;
use serde::{Serialize, Deserialize};

/// LSTM (Long Short-Term Memory) Cell
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LSTM {
    pub input_weights: Tensor,   // Weights for input
    pub hidden_weights: Tensor,  // Weights for hidden state
    pub bias: Tensor,             // Bias
    pub hidden_state: Tensor,     // Current hidden state
    pub cell_state: Tensor,       // Current cell state
}

impl LSTM {
    /// Create a new LSTM
    /// input_size: size of input features
    /// hidden_size: size of hidden state
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self> {
        let input_weights = Tensor::xavier(input_size, hidden_size * 4)?;     // 4 gates
        let hidden_weights = Tensor::xavier(hidden_size, hidden_size * 4)?;   // 4 gates
        let bias = Tensor::zeros(vec![hidden_size * 4])?;
        let hidden_state = Tensor::zeros(vec![hidden_size])?;
        let cell_state = Tensor::zeros(vec![hidden_size])?;

        Ok(LSTM {
            input_weights,
            hidden_weights,
            bias,
            hidden_state,
            cell_state,
        })
    }

    /// LSTM forward pass (simplified)
    fn lstm_cell(&self, input: &Tensor, h: &Tensor, c: &Tensor) -> Result<(Tensor, Tensor)> {
        // Simplified: compute gates (forget, input, candidate, output)
        let input_contribution = input.matmul(&self.input_weights)?;
        let hidden_contribution = h.matmul(&self.hidden_weights)?;
        let combined = input_contribution.add(&hidden_contribution)?;
        
        // Apply sigmoid + tanh for gates
        let gates = combined.sigmoid()?;
        let candidates = combined.tanh()?;
        
        // Simplified state update
        let new_c = c.mul(&gates)?;  // Forget gate
        let new_h = gates.mul(&candidates)?;  // Output gate
        
        Ok((new_h, new_c))
    }

    /// Describe the architecture
    pub fn architecture(&self) -> String {
        format!(
            "LSTM(input_size={}, hidden_size={})",
            self.input_weights.numel(),
            self.hidden_weights.numel()
        )
    }
}

impl Technique for LSTM {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // Process sequence
        let (new_h, new_c) = self.lstm_cell(input, &self.hidden_state, &self.cell_state)?;
        
        // Update state for next step
        self.hidden_state = new_h.clone();
        self.cell_state = new_c;
        
        // Return hidden state as output
        new_h
    }

    fn objective(&self, output: &Tensor, target: &Tensor) -> Result<Signal> {
        // MSE loss for sequence tasks
        let diff = output.sub(target)?;
        let squared = diff.mul(&diff)?;
        let loss = squared.mean();
        
        let gradients = diff.scale(2.0 / diff.numel() as f64)?;
        
        Ok(Signal { loss, gradients })
    }

    fn update(&mut self, signal: &Signal, learning_rate: f64) -> Result<()> {
        // Simplified update
        let grad = signal.gradients.mean();
        let h_val = self.hidden_state.as_slice()[0] - learning_rate * grad;
        self.hidden_state = Tensor::new(vec![h_val], vec![1])?;
        Ok(())
    }

    fn name(&self) -> &str {
        "LSTM"
    }

    fn description(&self) -> &str {
        "LSTM: Long Short-Term Memory recurrent network"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lstm_creation() {
        let model = LSTM::new(64, 128).unwrap();
        assert_eq!(model.hidden_state.shape(), vec![128]);
    }

    #[test]
    fn test_lstm_forward() {
        let mut model = LSTM::new(64, 128).unwrap();
        let input = Tensor::random(vec![1, 64]).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape(), vec![128]);
    }
}
