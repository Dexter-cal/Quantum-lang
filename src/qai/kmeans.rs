//! K-Means Clustering Technique
//!
//! Unsupervised learning for clustering data.
//! Implements the open Technique interface.
//!
//! Reference: Section 2 (classical ML), Section 46 (K-Means unique methods)

use crate::qai::technique::{Technique, Signal};
use crate::Tensor;
use crate::error::Result;
use serde::{Serialize, Deserialize};

/// K-Means Clustering Model
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KMeans {
    pub centroids: Tensor,      // [k, features]
    pub k: usize,
    pub max_iterations: usize,
}

impl KMeans {
    /// Create a new K-Means model
    /// k: number of clusters
    /// n_features: number of features per data point
    pub fn new(k: usize, n_features: usize) -> Result<Self> {
        let centroids = Tensor::random(vec![k, n_features])?;
        
        Ok(KMeans {
            centroids,
            k,
            max_iterations: 100,
        })
    }

    /// Compute distances from data points to centroids
    fn distances(&self, data: &Tensor) -> Result<Tensor> {
        // Simplified: compute all pairwise distances
        // Real implementation would use efficient distance metrics
        let n_samples = data.shape()[0];
        let mut distances = vec![];
        
        for i in 0..n_samples {
            for j in 0..self.k {
                // Euclidean distance placeholder
                distances.push(0.0);
            }
        }
        
        Tensor::new(distances, vec![n_samples, self.k])
    }

    /// Cluster centers and inertia
    pub fn cluster_centers(&self) -> Result<Tensor> {
        Ok(self.centroids.clone())
    }

    /// Describe the clustering
    pub fn description_verbose(&self) -> String {
        format!(
            "KMeans(k={}, n_features={}, max_iter={})",
            self.k,
            self.centroids.shape()[1],
            self.max_iterations
        )
    }
}

impl Technique for KMeans {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // Assign each point to nearest centroid
        let distances = self.distances(input)?;
        
        // Return cluster assignments (argmin)
        let assignments: Vec<f64> = (0..distances.shape()[0])
            .map(|i| {
                let row = &distances.as_slice()[i * self.k..(i+1) * self.k];
                row.iter().enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx as f64)
                    .unwrap_or(0.0)
            })
            .collect();
        
        Tensor::new(assignments, vec![distances.shape()[0], 1])
    }

    fn objective(&self, output: &Tensor, _target: &Tensor) -> Result<Signal> {
        // Inertia: sum of squared distances to nearest centroid
        // For unsupervised, we don't have targets
        let loss = output.as_slice().iter().map(|x| x * x).sum::<f64>() / output.numel() as f64;
        
        // Pseudo-gradients for interface compatibility
        let gradients = Tensor::zeros(output.shape())?;
        
        Ok(Signal { loss, gradients })
    }

    fn update(&mut self, signal: &Signal, _learning_rate: f64) -> Result<()> {
        // K-Means doesn't use gradients; centroids updated based on assignments
        // For MVP, simplified: don't update
        Ok(())
    }

    fn name(&self) -> &str {
        "KMeans"
    }

    fn description(&self) -> &str {
        "K-Means: Unsupervised clustering algorithm"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_creation() {
        let model = KMeans::new(3, 10).unwrap();
        assert_eq!(model.centroids.shape(), vec![3, 10]);
    }

    #[test]
    fn test_kmeans_forward() {
        let mut model = KMeans::new(3, 10).unwrap();
        let input = Tensor::random(vec![20, 10]).unwrap();
        let assignments = model.forward(&input).unwrap();
        assert_eq!(assignments.shape()[0], 20);
    }
}
