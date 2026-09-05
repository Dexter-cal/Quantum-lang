//! Technique Trait: Open Interface for ML Algorithms
//!
//! This is the core abstraction that enables the open-interface pattern.
//! Every ML algorithm (Regression, CNN, Transformer, MoE, etc.) implements this trait.
//! This prevents hardcoding and ensures all techniques are treated equally.
//!
//! Reference: Section 3 of the Quantum design document

use crate::Tensor;
use crate::error::Result;

/// Signal: Generic feedback from an objective function
/// Different techniques interpret this differently:
/// - Regression/Classification: loss + gradients
/// - RL: reward signal
/// - Diffusion: denoising error
#[derive(Clone, Debug)]
pub struct Signal {
    pub loss: f64,
    pub gradients: Tensor,
}

/// Open Technique Interface
///
/// Every ML technique must implement this trait.
/// Built-ins (Regression, CNN, Transformer) are just pre-shipped implementations.
/// Users can implement their own techniques the exact same way.
pub trait Technique: Send + Sync {
    /// Forward pass: transform input to output
    fn forward(&mut self, input: &Tensor) -> Result<Tensor>;

    /// Objective: compute the signal (loss, reward, etc.)
    /// Not limited to "loss" — different techniques use this differently
    fn objective(&self, output: &Tensor, target: &Tensor) -> Result<Signal>;

    /// Update: apply the signal to improve weights
    fn update(&mut self, signal: &Signal, learning_rate: f64) -> Result<()>;

    /// Name of the technique (for logging, debugging)
    fn name(&self) -> &str;

    /// Get a description of the technique
    fn description(&self) -> &str {
        "A machine learning technique"
    }

    /// Save weights to a format (JSON, binary, etc.)
    fn save_weights(&self, path: &str) -> Result<()> {
        Err(crate::error::QuantumError::TrainingError(
            format!("{} does not implement save_weights", self.name()),
        ))
    }

    /// Load weights from a format
    fn load_weights(&mut self, path: &str) -> Result<()> {
        Err(crate::error::QuantumError::TrainingError(
            format!("{} does not implement load_weights", self.name()),
        ))
    }
}
