//! Reinforcement Learning (Q-Learning) Technique
//!
//! Value-based RL agent learning from environment interaction.
//! Implements the open Technique interface.
//!
//! Reference: Section 2 (deep learning types), Section 34 (execution-grounded learning)

use crate::qai::technique::{Technique, Signal};
use crate::Tensor;
use crate::error::Result;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Q-Learning Agent
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QLearning {
    pub q_table: HashMap<String, Vec<f64>>,  // state -> Q-values
    pub learning_rate: f64,
    pub discount_factor: f64,  // gamma
    pub epsilon: f64,           // exploration rate
}

impl QLearning {
    /// Create a new Q-Learning agent
    /// learning_rate: how fast to update Q-values
    /// discount_factor: importance of future rewards (gamma)
    pub fn new(learning_rate: f64, discount_factor: f64) -> Result<Self> {
        Ok(QLearning {
            q_table: HashMap::new(),
            learning_rate,
            discount_factor,
            epsilon: 0.1,
        })
    }

    /// Get Q-values for a state
    fn get_q_values(&self, state: &str, n_actions: usize) -> Vec<f64> {
        self.q_table
            .get(state)
            .cloned()
            .unwrap_or_else(|| vec![0.0; n_actions])
    }

    /// Select action using epsilon-greedy strategy
    fn select_action(&self, state: &str, n_actions: usize) -> usize {
        if rand::random::<f64>() < self.epsilon {
            (rand::random::<f64>() * n_actions as f64) as usize
        } else {
            let q_values = self.get_q_values(state, n_actions);
            q_values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }
    }

    /// Describe the learning algorithm
    pub fn algorithm_info(&self) -> String {
        format!(
            "QLearning(lr={}, gamma={}, epsilon={})",
            self.learning_rate, self.discount_factor, self.epsilon
        )
    }
}

impl Technique for QLearning {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // For RL: forward is action selection
        // Input interpreted as state features
        let state_str = format!("{:?}", input.as_slice());
        let q_values = self.get_q_values(&state_str, 4);  // Assume 4 actions
        
        Tensor::new(q_values, vec![1, 4])
    }

    fn objective(&self, output: &Tensor, target: &Tensor) -> Result<Signal> {
        // TD error: loss = (Q-target - Q-current)^2
        let diff = target.sub(output)?;
        let squared = diff.mul(&diff)?;
        let loss = squared.mean();
        
        // Gradient is the TD error
        let gradients = diff.scale(2.0)?;
        
        Ok(Signal { loss, gradients })
    }

    fn update(&mut self, signal: &Signal, _learning_rate: f64) -> Result<()> {
        // Q-learning uses Bellman equation, not simple gradient descent
        // For MVP: simplified mock update
        let _ = signal;
        Ok(())
    }

    fn name(&self) -> &str {
        "QLearning"
    }

    fn description(&self) -> &str {
        "Q-Learning: Value-based reinforcement learning"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qlearning_creation() {
        let model = QLearning::new(0.1, 0.99).unwrap();
        assert_eq!(model.learning_rate, 0.1);
    }

    #[test]
    fn test_qlearning_forward() {
        let mut model = QLearning::new(0.1, 0.99).unwrap();
        let input = Tensor::random(vec![1, 4]).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape()[1], 4);  // 4 actions
    }
}
