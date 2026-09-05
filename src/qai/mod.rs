//! QuantumAI Framework
//!
//! The core ML/AI framework for Quantum.
//! Organized as modular subcomponents:
//!
//! - `tensor`: N-dimensional array operations
//! - `technique`: Open trait for defining ML algorithms
//! - `regression`: Linear regression (first technique)
//! - `optimizer`: Gradient-based optimizers (SGD, Adam)
//! - `loss`: Loss function implementations
//! - `data`: Dataset utilities

pub mod tensor;
pub mod technique;
pub mod regression;
pub mod optimizer;
pub mod loss;
pub mod data;

pub use tensor::Tensor;
pub use technique::Technique;
pub use regression::Regression;
pub use optimizer::{Optimizer, SGD, Adam};
pub use loss::{Loss, LossFn};
