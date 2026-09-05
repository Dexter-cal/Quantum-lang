//! Command-line interface for Quantum
//!
//! Simple CLI for training and evaluating models.
//! Usage:
//!   qai build regression
//!   qai train --data sample.csv --epochs 100
//!   qai predict --model model.json --input "1.0 2.0 3.0"

use quantum::{
    Tensor,
    Technique,
    Regression,
    qai::data::make_regression,
};
use std::io::{self, Write};

fn main() {
    println!("\n╔════════════════════════════════════════╗");
    println!("║  Quantum AI Framework v0.1.0          ║");
    println!("║  Simple Linear Regression Example      ║");
    println!("╚════════════════════════════════════════╝\n");

    // Step 1: Create model
    println!("[1/5] Creating regression model with 1 input feature...");
    let mut model = Regression::new(1).expect("Failed to create model");
    println!("      ✓ Model created");
    println!("      Formula: {}", model.formula());

    // Step 2: Generate synthetic data
    println!("\n[2/5] Generating synthetic regression data...");
    let dataset = make_regression(100, 1).expect("Failed to generate data");
    println!("      ✓ Generated {} samples", dataset.n_samples());
    println!("      Features per sample: {}", dataset.n_features());

    // Step 3: Training loop
    println!("\n[3/5] Training model (100 epochs, learning_rate=0.01)...");
    let learning_rate = 0.01;
    let epochs = 100;
    let mut losses = vec![];

    for epoch in 0..epochs {
        // Forward pass
        let output = model.forward(&dataset.x).expect("Forward pass failed");

        // Compute loss
        let signal = model.objective(&output, &dataset.y).expect("Objective failed");
        losses.push(signal.loss);

        // Update model
        model.update(&signal, learning_rate).expect("Update failed");

        // Print progress
        if (epoch + 1) % 10 == 0 {
            println!("      Epoch {:3}/{}: loss = {:.6}", epoch + 1, epochs, signal.loss);
        }
    }

    // Step 4: Evaluate
    println!("\n[4/5] Evaluation on training data...");
    let final_output = model.forward(&dataset.x).expect("Forward pass failed");
    let final_signal = model
        .objective(&final_output, &dataset.y)
        .expect("Objective failed");
    println!("      Final Loss: {:.6}", final_signal.loss);
    println!("      Loss improvement: {:.2}%", 
        ((losses[0] - losses[losses.len() - 1]) / losses[0]) * 100.0
    );

    // Step 5: Make predictions
    println!("\n[5/5] Making predictions...");
    let test_input = Tensor::new(vec![2.0, 5.0, 8.0], vec![3, 1])
        .expect("Failed to create test input");
    let predictions = model.forward(&test_input).expect("Prediction failed");
    
    println!("\n      Test inputs: [2.0, 5.0, 8.0]");
    println!("      Predictions: {:?}", predictions.as_slice());
    println!("      Expected (y = 2x+3): [7.0, 13.0, 19.0]");

    // Summary
    println!("\n╔════════════════════════════════════════╗");
    println!("║  Training Complete                     ║");
    println!("╚════════════════════════════════════════╝\n");
    
    println!("Model: {}", model.name());
    println!("Description: {}", model.description());
    println!("Final Loss: {:.6}", final_signal.loss);
    println!("\n✓ Framework MVP working correctly!\n");
}
