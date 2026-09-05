//! Data utilities: datasets, loaders, preprocessing

use crate::Tensor;
use crate::error::Result;

/// Simple dataset wrapper
pub struct Dataset {
    pub x: Tensor,      // features
    pub y: Tensor,      // labels/targets
}

impl Dataset {
    pub fn new(x: Tensor, y: Tensor) -> Result<Self> {
        let x_shape = x.shape();
        let y_shape = y.shape();
        
        if x_shape[0] != y_shape[0] {
            return Err(crate::error::QuantumError::ShapeMismatch {
                expected: vec![x_shape[0]],
                actual: vec![y_shape[0]],
            });
        }
        
        Ok(Dataset { x, y })
    }

    pub fn n_samples(&self) -> usize {
        self.x.shape()[0]
    }

    pub fn n_features(&self) -> usize {
        if self.x.ndim() > 1 {
            self.x.shape()[1]
        } else {
            1
        }
    }
}

/// Generate simple regression data: y = 2*x + 3 + noise
pub fn make_regression(n_samples: usize, n_features: usize) -> Result<Dataset> {
    let mut x_data = vec![];
    let mut y_data = vec![];
    
    for _ in 0..n_samples {
        for _ in 0..n_features {
            x_data.push(rand::random::<f64>() * 10.0);
        }
        
        // Compute y = 2*x + 3 + noise
        let y = 2.0 * x_data[x_data.len() - 1] + 3.0 + (rand::random::<f64>() - 0.5) * 2.0;
        y_data.push(y);
    }
    
    let x = Tensor::new(x_data, vec![n_samples, n_features])?;
    let y = Tensor::new(y_data, vec![n_samples, 1])?;
    
    Dataset::new(x, y)
}
