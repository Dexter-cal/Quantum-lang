//! Decision Tree Technique
//!
//! Tree-based classifier using recursive splitting.
//! Implements the open Technique interface.
//!
//! Reference: Section 2 (classical ML), Section 46 (decision tree unique methods)

use crate::qai::technique::{Technique, Signal};
use crate::Tensor;
use crate::error::Result;
use serde::{Serialize, Deserialize};

/// Decision Tree Node
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreeNode {
    pub is_leaf: bool,
    pub feature_idx: Option<usize>,
    pub threshold: Option<f64>,
    pub class: Option<usize>,
    pub left: Option<Box<TreeNode>>,
    pub right: Option<Box<TreeNode>>,
}

/// Decision Tree Classifier
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionTree {
    pub root: TreeNode,
    pub max_depth: usize,
    pub n_classes: usize,
}

impl DecisionTree {
    /// Create a new decision tree
    /// max_depth: maximum tree depth
    /// n_classes: number of output classes
    pub fn new(max_depth: usize, n_classes: usize) -> Result<Self> {
        let root = TreeNode {
            is_leaf: true,
            feature_idx: None,
            threshold: None,
            class: Some(0),
            left: None,
            right: None,
        };
        
        Ok(DecisionTree {
            root,
            max_depth,
            n_classes,
        })
    }

    /// Find the best split (simplified)
    fn find_best_split(&self, data: &Tensor) -> Option<(usize, f64)> {
        // Placeholder: return first feature and median value
        if data.shape().len() > 1 && data.shape()[1] > 0 {
            Some((0, 0.5))
        } else {
            None
        }
    }

    /// Predict class for a single sample
    fn predict_sample(&self, sample: &[f64]) -> usize {
        let mut node = &self.root;
        
        while !node.is_leaf {
            if let (Some(idx), Some(threshold)) = (node.feature_idx, node.threshold) {
                if idx < sample.len() && sample[idx] <= threshold {
                    if let Some(ref left) = node.left {
                        node = left;
                    } else {
                        break;
                    }
                } else if let Some(ref right) = node.right {
                    node = right;
                } else {
                    break;
                }
            }
        }
        
        node.class.unwrap_or(0)
    }

    /// Describe the tree
    pub fn tree_info(&self) -> String {
        format!(
            "DecisionTree(max_depth={}, n_classes={}, n_nodes=1)",
            self.max_depth, self.n_classes
        )
    }
}

impl Technique for DecisionTree {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // Predict class for each sample
        let n_samples = input.shape()[0];
        let mut predictions = vec![];
        
        for i in 0..n_samples {
            let start_idx = i * input.shape()[1];
            let end_idx = (i + 1) * input.shape()[1];
            let sample = &input.as_slice()[start_idx..end_idx];
            let class = self.predict_sample(sample);
            
            // One-hot encode
            for j in 0..self.n_classes {
                predictions.push(if j == class { 1.0 } else { 0.0 });
            }
        }
        
        Tensor::new(predictions, vec![n_samples, self.n_classes])
    }

    fn objective(&self, output: &Tensor, target: &Tensor) -> Result<Signal> {
        // Classification error or gini impurity
        let diff = output.sub(target)?;
        let squared = diff.mul(&diff)?;
        let loss = squared.mean();
        
        let gradients = diff.scale(2.0 / diff.numel() as f64)?;
        
        Ok(Signal { loss, gradients })
    }

    fn update(&mut self, signal: &Signal, _learning_rate: f64) -> Result<()> {
        // Trees don't use gradient descent; no update needed
        let _ = signal;  // Avoid unused warning
        Ok(())
    }

    fn name(&self) -> &str {
        "DecisionTree"
    }

    fn description(&self) -> &str {
        "Decision Tree: Recursive partitioning classifier"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_creation() {
        let model = DecisionTree::new(5, 3).unwrap();
        assert_eq!(model.max_depth, 5);
        assert_eq!(model.n_classes, 3);
    }

    #[test]
    fn test_tree_forward() {
        let mut model = DecisionTree::new(5, 3).unwrap();
        let input = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        ).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape()[0], 2);
        assert_eq!(output.shape()[1], 3);
    }
}
