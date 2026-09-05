//! Tensor: N-dimensional array operations
//!
//! The core computational unit of the Quantum framework.
//! Built on top of ndarray for performance.

use ndarray::{Array, ArrayD, IxDyn, s};
use crate::error::{QuantumError, Result};
use std::fmt;

/// A multi-dimensional array (tensor) for ML computations
#[derive(Clone, Debug)]
pub struct Tensor {
    data: ArrayD<f64>,
}

impl Tensor {
    /// Create a tensor from raw data and shape
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Result<Self> {
        if shape.is_empty() {
            return Err(QuantumError::InvalidDimension("Shape cannot be empty".to_string()));
        }
        
        let total_elements: usize = shape.iter().product();
        if data.len() != total_elements {
            return Err(QuantumError::ShapeMismatch {
                expected: shape,
                actual: vec![data.len()],
            });
        }
        
        let arr = Array::from_shape_vec(IxDyn(&shape), data)
            .map_err(|_| QuantumError::InvalidDimension("Failed to reshape tensor".to_string()))?;
        
        Ok(Tensor { data: arr })
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: Vec<usize>) -> Result<Self> {
        if shape.is_empty() {
            return Err(QuantumError::InvalidDimension("Shape cannot be empty".to_string()));
        }
        let arr = Array::zeros(IxDyn(&shape));
        Ok(Tensor { data: arr })
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: Vec<usize>) -> Result<Self> {
        if shape.is_empty() {
            return Err(QuantumError::InvalidDimension("Shape cannot be empty".to_string()));
        }
        let arr = Array::ones(IxDyn(&shape));
        Ok(Tensor { data: arr })
    }

    /// Create a tensor with random values (0, 1)
    pub fn random(shape: Vec<usize>) -> Result<Self> {
        if shape.is_empty() {
            return Err(QuantumError::InvalidDimension("Shape cannot be empty".to_string()));
        }
        let total: usize = shape.iter().product();
        let data: Vec<f64> = (0..total).map(|_| rand::random()).collect();
        Self::new(data, shape)
    }

    /// Create a tensor with Xavier initialization (for weights)
    pub fn xavier(in_size: usize, out_size: usize) -> Result<Self> {
        let limit = (6.0 / (in_size + out_size) as f64).sqrt();
        let total = in_size * out_size;
        let data: Vec<f64> = (0..total)
            .map(|_| (rand::random::<f64>() * 2.0 - 1.0) * limit)
            .collect();
        Self::new(data, vec![in_size, out_size])
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Reshape tensor to new shape
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor> {
        let total_elements: usize = new_shape.iter().product();
        if total_elements != self.numel() {
            return Err(QuantumError::ShapeMismatch {
                expected: new_shape,
                actual: self.shape(),
            });
        }
        let data = self.data.iter().copied().collect();
        Tensor::new(data, new_shape)
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape() != other.shape() {
            return Err(QuantumError::ShapeMismatch {
                expected: self.shape(),
                actual: other.shape(),
            });
        }
        let result = self.data.clone() + &other.data;
        let data = result.iter().copied().collect();
        Tensor::new(data, self.shape())
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape() != other.shape() {
            return Err(QuantumError::ShapeMismatch {
                expected: self.shape(),
                actual: other.shape(),
            });
        }
        let result = self.data.clone() - &other.data;
        let data = result.iter().copied().collect();
        Tensor::new(data, self.shape())
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape() != other.shape() {
            return Err(QuantumError::ShapeMismatch {
                expected: self.shape(),
                actual: other.shape(),
            });
        }
        let result = self.data.clone() * &other.data;
        let data = result.iter().copied().collect();
        Tensor::new(data, self.shape())
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: f64) -> Result<Tensor> {
        let result = self.data.clone() * scalar;
        let data = result.iter().copied().collect();
        Tensor::new(data, self.shape())
    }

    /// Matrix multiplication (for 2D tensors)
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(QuantumError::InvalidDimension(
                "matmul requires 2D tensors".to_string(),
            ));
        }
        
        let self_shape = self.shape();
        let other_shape = other.shape();
        
        if self_shape[1] != other_shape[0] {
            return Err(QuantumError::ShapeMismatch {
                expected: vec![self_shape[0], other_shape[1]],
                actual: vec![self_shape[1], other_shape[0]],
            });
        }

        let self_2d = self.data.clone().into_shape((self_shape[0], self_shape[1]))?;
        let other_2d = other.data.clone().into_shape((other_shape[0], other_shape[1]))?;
        let result = self_2d.dot(&other_2d);
        
        let data = result.iter().copied().collect();
        Tensor::new(data, vec![result.shape()[0], result.shape()[1]])
    }

    /// Transpose (for 2D tensors)
    pub fn transpose(&self) -> Result<Tensor> {
        if self.ndim() != 2 {
            return Err(QuantumError::InvalidDimension(
                "transpose requires 2D tensor".to_string(),
            ));
        }
        let shape = self.shape();
        let data_2d = self.data.clone().into_shape((shape[0], shape[1]))?;
        let transposed = data_2d.t().to_owned();
        let data = transposed.iter().copied().collect();
        Tensor::new(data, vec![shape[1], shape[0]])
    }

    /// Sum all elements
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Mean of all elements
    pub fn mean(&self) -> f64 {
        self.sum() / self.numel() as f64
    }

    /// Standard deviation
    pub fn std(&self) -> f64 {
        let mean = self.mean();
        let variance: f64 = self.data.iter().map(|x| (x - mean).powi(2)).sum();
        (variance / self.numel() as f64).sqrt()
    }

    /// Min value
    pub fn min(&self) -> f64 {
        self.data.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Max value
    pub fn max(&self) -> f64 {
        self.data.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    /// ReLU activation
    pub fn relu(&self) -> Result<Tensor> {
        let data: Vec<f64> = self.data.iter().map(|x| x.max(0.0)).collect();
        Tensor::new(data, self.shape())
    }

    /// Sigmoid activation
    pub fn sigmoid(&self) -> Result<Tensor> {
        let data: Vec<f64> = self.data.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        Tensor::new(data, self.shape())
    }

    /// Softmax (for 1D or last dimension)
    pub fn softmax(&self) -> Result<Tensor> {
        let shape = self.shape();
        if shape.is_empty() {
            return Err(QuantumError::InvalidDimension("Cannot softmax empty tensor".to_string()));
        }
        
        let max_val = self.max();
        let shifted: Vec<f64> = self.data.iter().map(|x| x - max_val).collect();
        let exp_vals: Vec<f64> = shifted.iter().map(|x| x.exp()).collect();
        let sum_exp: f64 = exp_vals.iter().sum();
        
        let data: Vec<f64> = exp_vals.iter().map(|x| x / sum_exp).collect();
        Tensor::new(data, shape)
    }

    /// Tanh activation
    pub fn tanh(&self) -> Result<Tensor> {
        let data: Vec<f64> = self.data.iter().map(|x| x.tanh()).collect();
        Tensor::new(data, self.shape())
    }

    /// Get raw data as slice
    pub fn as_slice(&self) -> &[f64] {
        self.data.as_slice().unwrap_or(&[])
    }

    /// Get mutable raw data
    pub fn as_slice_mut(&mut self) -> &mut [f64] {
        self.data.as_slice_mut().unwrap_or(&mut [])
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Tensor(shape: {:?}, min: {:.4}, max: {:.4}, mean: {:.4})",
            self.shape(),
            self.min(),
            self.max(),
            self.mean()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(t.shape(), vec![2, 2]);
        assert_eq!(t.numel(), 4);
    }

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::zeros(vec![3, 3]).unwrap();
        assert_eq!(t.sum(), 0.0);
    }

    #[test]
    fn test_tensor_add() {
        let t1 = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let t2 = Tensor::new(vec![3.0, 4.0], vec![2]).unwrap();
        let result = t1.add(&t2).unwrap();
        assert_eq!(result.sum(), 10.0);
    }

    #[test]
    fn test_tensor_matmul() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let t2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let result = t1.matmul(&t2).unwrap();
        assert_eq!(result.shape(), vec![2, 2]);
    }
}
