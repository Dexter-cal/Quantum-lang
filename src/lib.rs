//! Quantum Language Framework
//! A compiled programming language with built-in AI/ML capabilities
//!
//! This library provides the core components of the Quantum framework:
//! - Tensor operations (core computational unit)
//! - Technique trait (open interface for ML algorithms)
//! - Built-in techniques (Regression, will add CNN, Transformer, etc.)
//! - Optimizers and loss functions

pub mod qai;
pub mod error;

pub use qai::tensor::Tensor;
pub use qai::technique::Technique;
pub use qai::regression::Regression;
pub use error::{QuantumError, Result};
