//! Error types for the Quantum framework

use thiserror::Error;

#[derive(Debug, Error)]
pub enum QuantumError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },

    #[error("Invalid tensor dimension: {0}")]
    InvalidDimension(String),

    #[error("Division by zero in loss calculation")]
    DivisionByZero,

    #[error("Training error: {0}")]
    TrainingError(String),

    #[error("Data loading error: {0}")]
    DataError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

pub type Result<T> = std::result::Result<T, QuantumError>;
