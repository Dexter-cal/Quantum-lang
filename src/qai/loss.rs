//! Loss Function Implementations
//!
//! Common loss functions: MSE, CrossEntropy, MAE, etc.

use crate::Tensor;
use crate::error::Result;
use serde::{Serialize, Deserialize};

/// Loss function types
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum LossFn {
    MSE,
    MAE,
    CrossEntropy,
    BCE, // Binary Cross Entropy
    Huber,
}

impl LossFn {
    pub fn name(&self) -> &str {
        match self {
            LossFn::MSE => "MSE",
            LossFn::MAE => "MAE",
            LossFn::CrossEntropy => "CrossEntropy",
            LossFn::BCE => "BCE",
            LossFn::Huber => "Huber",
        }
    }
}

/// Loss function trait
pub trait Loss {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<f64>;
    fn gradient(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor>;
}

/// Mean Squared Error Loss
pub struct MSELoss;

impl Loss for MSELoss {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<f64> {
        let diff = predictions.sub(targets)?;
        let squared = diff.mul(&diff)?;
        Ok(squared.mean())
    }

    fn gradient(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let diff = predictions.sub(targets)?;
        let n = diff.numel() as f64;
        diff.scale(2.0 / n)
    }
}

/// Mean Absolute Error Loss
pub struct MAELoss;

impl Loss for MAELoss {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<f64> {
        let diff = predictions.sub(targets)?;
        let abs_diff: Vec<f64> = diff.as_slice().iter().map(|x| x.abs()).collect();
        Ok(abs_diff.iter().sum::<f64>() / abs_diff.len() as f64)
    }

    fn gradient(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let diff = predictions.sub(targets)?;
        let grad: Vec<f64> = diff.as_slice().iter().map(|x| x.signum()).collect();
        Tensor::new(grad, diff.shape())
    }
}
