//! # QuantumMind v1.0
//! The AI coding assistant built INTO the Quantum Programming Language.
//!
//! Features:
//! - Generates full projects from a single prompt
//! - Trains AI models, creates datasets, builds libraries
//! - Accesses the internet for datasets and documentation
//! - Creates its own tools when it needs them
//! - Learns from every interaction
//! - Reasons from first principles, not just pattern matching
//! - Fully offline capable for Quantum-specific tasks

pub mod brain;
pub mod codegen;
pub mod knowledge;
pub mod memory;
pub mod tools;
pub mod reasoner;
pub mod internet;
pub mod trainer;
pub mod project_builder;
pub mod conversation;
pub mod self_improver;
pub mod real_trainer;

pub use brain::QuantumMind;
pub use conversation::Message;
