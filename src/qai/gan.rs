//! GAN (Generative Adversarial Network) Technique
//!
//! Generator and discriminator for generative tasks.
//! Implements the open Technique interface.
//!
//! Reference: Section 2 (deep learning types), Section 46 (GAN unique methods)

use crate::qai::technique::{Technique, Signal};
use crate::Tensor;
use crate::error::Result;
use serde::{Serialize, Deserialize};

/// GAN Model: Generator + Discriminator
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GAN {
    pub generator_weights: Tensor,      // Generator network
    pub discriminator_weights: Tensor,  // Discriminator network
    pub latent_dim: usize,
}

impl GAN {
    /// Create a new GAN
    /// latent_dim: dimension of latent space (typically 100)
    /// output_dim: dimension of generated output
    pub fn new(latent_dim: usize, output_dim: usize) -> Result<Self> {
        let generator_weights = Tensor::xavier(latent_dim, output_dim)?;
        let discriminator_weights = Tensor::xavier(output_dim, 1)?;  // Binary classification

        Ok(GAN {
            generator_weights,
            discriminator_weights,
            latent_dim,
        })
    }

    /// Generator forward pass
    fn generate(&self, noise: &Tensor) -> Result<Tensor> {
        noise.matmul(&self.generator_weights)
    }

    /// Discriminator forward pass
    fn discriminate(&self, data: &Tensor) -> Result<Tensor> {
        let logits = data.matmul(&self.discriminator_weights)?;
        logits.sigmoid()  // Output probability [0, 1]
    }

    /// Describe the architecture
    pub fn architecture(&self) -> String {
        format!(
            "GAN(latent_dim={}, gen_size={}, disc_size={})",
            self.latent_dim,
            self.generator_weights.numel(),
            self.discriminator_weights.numel()
        )
    }
}

impl Technique for GAN {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // For GAN, forward is generator: noise -> fake data
        self.generate(input)
    }

    fn objective(&self, output: &Tensor, target: &Tensor) -> Result<Signal> {
        // Binary cross-entropy for discriminator loss
        // loss = -[y*log(D) + (1-y)*log(1-D)]
        
        let epsilon = 1e-7;
        let one_minus_output: Vec<f64> = output
            .as_slice()
            .iter()
            .map(|x| (1.0 - x).max(epsilon))
            .collect();
        let one_minus = Tensor::new(one_minus_output, output.shape())?;
        
        let output_safe: Vec<f64> = output
            .as_slice()
            .iter()
            .map(|x| x.max(epsilon))
            .collect();
        let output_safe_t = Tensor::new(output_safe, output.shape())?;
        
        let first_term = target.mul(&output_safe_t)?;
        let one_minus_target: Vec<f64> = target
            .as_slice()
            .iter()
            .map(|x| 1.0 - x)
            .collect();
        let one_minus_target_t = Tensor::new(one_minus_target, target.shape())?;
        let second_term = one_minus_target_t.mul(&one_minus)?;
        let sum_terms = first_term.add(&second_term)?;
        let loss = -sum_terms.mean();

        // Gradient for discriminator
        let gradients = output.sub(target)?;

        Ok(Signal { loss, gradients })
    }

    fn update(&mut self, signal: &Signal, learning_rate: f64) -> Result<()> {
        // Simplified: mock update
        let grad = signal.gradients.mean();
        let new_val = self.generator_weights.mean() - learning_rate * grad;
        self.generator_weights = Tensor::new(vec![new_val], vec![1])?;
        Ok(())
    }

    fn name(&self) -> &str {
        "GAN"
    }

    fn description(&self) -> &str {
        "GAN: Generative Adversarial Network"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gan_creation() {
        let model = GAN::new(100, 784).unwrap();  // 100-dim latent, 28x28 images
        assert!(!model.generator_weights.shape().is_empty());
    }

    #[test]
    fn test_gan_forward() {
        let mut model = GAN::new(100, 784).unwrap();
        let noise = Tensor::random(vec![8, 100]).unwrap();
        let generated = model.forward(&noise).unwrap();
        assert_eq!(generated.shape()[0], 8);
        assert_eq!(generated.shape()[1], 784);
    }
}
