//! Code Generator — generates full Quantum projects from English descriptions
//! Can build classifiers, full ML pipelines, entire applications from one prompt.

use crate::knowledge::KnowledgeBase;
use crate::reasoner::ReasoningEngine;

pub struct CodeGenerator {
    pub knowledge: KnowledgeBase,
    pub reasoner: ReasoningEngine,
}

#[derive(Debug, Clone)]
pub struct GeneratedProject {
    pub name: String,
    pub description: String,
    pub files: Vec<GeneratedFile>,
    pub instructions: Vec<String>,
    pub estimated_accuracy: String,
}

#[derive(Debug, Clone)]
pub struct GeneratedFile {
    pub path: String,
    pub content: String,
    pub description: String,
}

impl CodeGenerator {
    pub fn new() -> Self {
        Self {
            knowledge: KnowledgeBase::new(),
            reasoner: ReasoningEngine::new(),
        }
    }

    /// Generate a complete project from a single English prompt
    pub fn generate_project(&self, prompt: &str) -> GeneratedProject {
        let (task_type, confidence) = self.reasoner.identify_task(prompt);
        let project_name = self.extract_project_name(prompt);
        let steps = self.reasoner.decompose_problem(prompt);

        let mut files = Vec::new();

        // Generate main source file.
        // NOTE: this is Rust source using the `quantumai` crate (the same
        // pattern as quantumai/examples/*.rs), NOT Quantum-language (.qtm)
        // source — quantumc does not yet support the full QuantumAI API
        // surface used here (closures, generics, Vec<&str>, etc.). It is
        // built and run via `cargo`, hence src/main.rs + Cargo.toml below.
        let main_code = self.generate_main_file(prompt, task_type);
        files.push(GeneratedFile {
            path: "src/main.rs".to_string(),
            content: main_code,
            description: "Main application entry point (Rust + QuantumAI)".to_string(),
        });

        // Generate Cargo.toml — points at the local quantumai crate, mirroring
        // how quantumai/examples are built within this workspace.
        files.push(GeneratedFile {
            path: "Cargo.toml".to_string(),
            content: format!(r#"[package]
name = "{}"
version = "0.1.0"
edition = "2021"
description = "{}"

[dependencies]
quantumai = {{ path = "../quantumai" }}
"#, project_name, prompt.replace('"', "'")),
            description: "Cargo project configuration".to_string(),
        });

        // Generate README
        files.push(GeneratedFile {
            path: "README.md".to_string(),
            content: self.generate_readme(&project_name, prompt, task_type, &steps),
            description: "Project documentation".to_string(),
        });

        // Generate test file
        files.push(GeneratedFile {
            path: "tests/test_model.rs".to_string(),
            content: self.generate_tests(task_type),
            description: "Unit tests for the model".to_string(),
        });

        let acc_estimate = match task_type {
            "classifier" | "nlp" | "image_classification" => "85-99% accuracy (depends on data quality)",
            "regressor" => "R² > 0.85 (depends on data complexity)",
            "anomaly" => "Detects 90%+ of anomalies with low false positives",
            "clustering" => "Silhouette score > 0.5 for well-separated clusters",
            _ => "Performance depends on data quality and quantity",
        }.to_string();

        GeneratedProject {
            name: project_name,
            description: prompt.to_string(),
            files,
            instructions: steps,
            estimated_accuracy: acc_estimate,
        }
    }

    fn extract_project_name(&self, prompt: &str) -> String {
        // Strip leading filler/command words so e.g. "build full project
        // fraud detection system" yields "fraud_detection_system_using"
        // instead of "build_full_project_fraud".
        let mut cleaned = prompt.to_lowercase();
        for filler in ["build full project ", "build a full project ", "create full project ",
                       "build complete project ", "full project ", "build ", "create ", "generate "] {
            if cleaned.starts_with(filler) {
                cleaned = cleaned[filler.len()..].to_string();
                break;
            }
        }
        let words: Vec<&str> = cleaned.split_whitespace().take(4).collect();
        words.join("_")
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .collect::<String>()
            .trim_matches('_')
            .to_string()
    }

    pub fn generate_main_file(&self, prompt: &str, task_type: &str) -> String {
        let p = prompt.to_lowercase();
        // Sanitize for embedding in generated string literals: strip quotes
        // and backslashes so the generated .qtm file's println!("...") and
        // header comment remain valid regardless of what the user typed.
        let prompt: String = prompt.chars()
            .filter(|&c| c != '"' && c != '\\')
            .collect();
        let prompt = prompt.as_str();

        // Extract dataset hint from prompt
        let dataset = if p.contains("iris") { "iris" }
            else if p.contains("mnist") { "mnist" }
            else if p.contains("csv") || p.contains("file") { "your_data.csv" }
            else { "iris" };  // default to iris as demo

        let epochs = if p.contains("quick") || p.contains("fast") { 50 }
            else if p.contains("accurate") || p.contains("best") { 300 }
            else { 150 };

        let lr = if task_type == "regressor" { 0.001 }
            else { 0.01 };

        match task_type {
            "classifier" | "nlp" | "image_classification" => format!(r#"// {prompt}
// Generated by QuantumMind v1.0
// Task: {task_type}

use quantumai::*;

fn main() {{
    println!("=== {prompt} ===");

    // ── Step 1: Load and analyse data ─────────────────────────────
    println!("\n[1/6] Loading data...");
    let pipeline = DataPipeline::load("{dataset}")
        .clean()
        .normalize()
        .shuffle();
    let ds = pipeline.dataset();

    let feat_names: Vec<&str> = (0..ds.n_features)
        .map(|i| Box::leak(format!("feature_{{}}", i+1).into_boxed_str()) as &str)
        .collect();
    let cls_names: Vec<&str> = (0..ds.n_classes)
        .map(|i| Box::leak(format!("class_{{}}", i+1).into_boxed_str()) as &str)
        .collect();

    let qr = DataQualityReport::analyze(&ds.x, &ds.y, &feat_names, &cls_names);
    qr.print(&cls_names);

    // ── Step 2: Split ──────────────────────────────────────────────
    println!("\n[2/6] Splitting train/test...");
    let (train, test) = ds.train_test_split(0.2);
    println!("  Train: {{}} samples  Test: {{}} samples",
        train.n_samples, test.n_samples);

    // ── Step 3: Hyperparameter search ──────────────────────────────
    println!("\n[3/6] Searching for best hyperparameters...");
    let best = tune(&train, 8);
    println!("  Best: optimizer={{}} lr={{}} accuracy={{:.3}}",
        best.best_optimizer, best.best_lr, best.best_accuracy);

    // ── Step 4: Train with best config ────────────────────────────
    println!("\n[4/6] Training model...");
    let mut model = Model::create("classifier");
    model.lr = best.best_lr;
    model.opt_name = best.best_optimizer.clone();
    model.train(&train.x, &train.y, {epochs}, true);

    // ── Step 5: Full benchmark ─────────────────────────────────────
    println!("\n[5/6] Benchmarking...");
    let report = BenchmarkReport::run(
        &mut model,
        &train.x, &train.y,
        &test.x,  &test.y,
        ds.n_classes, &cls_names,
    );
    report.print(&cls_names);

    // ── Step 6: AI analysis + deployment ──────────────────────────
    println!("\n[6/6] AI analysis...");
    AIAssistant::analyze(&report);
    ModelCard::from_benchmark(&model, &report).print();
    model.hardware_info();

    // Save model
    model.save("model.json");
    println!("\n✅ Project complete!");
    println!("   Model saved to: model.json");
    println!("   Deploy with: model.serve(8080)");
    println!("   Accuracy: {{:.1}}%", report.test_accuracy * 100.0);
}}"#, prompt=prompt, dataset=dataset, epochs=epochs),

            "regressor" => format!(r#"// {prompt}
// Generated by QuantumMind v1.0
// Task: regression

use quantumai::*;
use quantumai::ml::*;

fn main() {{
    println!("=== Regression: {prompt} ===");

    let pipeline = DataPipeline::load("{dataset}")
        .normalize()
        .shuffle();
    let ds = pipeline.dataset();
    let (train, test) = ds.train_test_split(0.2);

    // Try multiple regression approaches
    println!("Training Linear Regression...");
    let mut lr = LinearRegression::new();
    lr.lr = 0.01; lr.epochs = 500;
    lr.fit(&train.x, &train.y);
    println!("LinearRegression R²: {{:.4}}", lr.r2_score(&test.x, &test.y));

    println!("Training Neural Regressor...");
    let mut model = Model::create("regressor");
    model.lr = {lr};
    model.train(&train.x, &train.y, {epochs}, true);

    let ev = model.evaluate(&test.x, &test.y);
    println!("\nNeural Regressor MSE: {{:.4}}", ev["loss"]);

    model.save("regressor.json");
    println!("\n✅ Regression model complete!");
}}"#, prompt=prompt, dataset=dataset, epochs=epochs, lr=lr),

            "anomaly" => format!(r#"// {prompt}
// Generated by QuantumMind v1.0
// Task: anomaly detection

use quantumai::*;

fn main() {{
    println!("=== Anomaly Detection: {prompt} ===");

    // Load normal data (the model learns what "normal" looks like)
    let normal_data = DataPipeline::load("{dataset}").normalize().dataset();
    println!("Loaded {{}} normal samples", normal_data.n_samples);

    // Train anomaly detector
    println!("Training AnomalyDetector...");
    let mut detector = AnomalyDetector::new(normal_data.n_features);
    detector.train_on_normal(&normal_data.x, 150);

    // Test on samples
    println!("\nScoring samples:");
    let n = normal_data.n_samples.min(5);
    let nf = normal_data.n_features;
    for i in 0..n {{
        let sample = Tensor::new(
            normal_data.x.data[i*nf..(i+1)*nf].to_vec(),
            vec![1, nf]
        );
        println!("  Sample {{}}: {{}}", i+1, detector.describe(&sample));
    }}

    // Simulate an anomaly
    let fake_anomaly = Tensor::new(vec![999.0; nf], vec![1, nf]);
    println!("\n  Injected anomaly: {{}}", detector.describe(&fake_anomaly));

    println!("\n✅ Anomaly detector ready!");
}}"#, prompt=prompt, dataset=dataset),

            "timeseries" => format!(r#"// {prompt}
// Generated by QuantumMind v1.0
// Task: time series

use quantumai::*;

fn main() {{
    println!("=== Time Series: {prompt} ===");

    let seq_len = 10;
    let input_size = 1;
    let hidden_size = 64;
    let batch_size = 16;

    println!("Building LSTM model...");
    println!("  Input:  sequence length = {{}} steps", seq_len);
    println!("  Hidden: {{}} neurons", hidden_size);
    println!("  Layers: 2");

    let lstm = LSTM::new(input_size, hidden_size, 2);

    // Simulate time series input
    let input = Tensor::randn(vec![batch_size, seq_len, input_size]);
    let (output, cell) = lstm.forward(&input);

    println!("  Output shape: {{:?}}", output.shape);
    println!("  Cell shape:   {{:?}}", cell.shape);

    println!("\nFor real forecasting:");
    println!("  1. Prepare windowed data (past N steps → next value)");
    println!("  2. Normalize the time series");
    println!("  3. Train LSTM on historical data");
    println!("  4. Predict future values");
    println!("\n✅ Time series model built!");
}}"#, prompt=prompt),

            _ => format!(r#"// {prompt}
// Generated by QuantumMind v1.0

use quantumai::*;

fn main() {{
    // Auto-generated project
    let auto = AutoMLPipeline::run(
        "iris",
        "classifier",
        &["class1", "class2", "class3"],
        &["feature1", "feature2", "feature3", "feature4"],
    );
    println!("Done in {{:.1}}s", auto.total_time_s);
}}"#, prompt=prompt),
        }
    }

    fn generate_readme(&self, name: &str, prompt: &str, task: &str, steps: &[String]) -> String {
        format!(r#"# {}

> {}

## What This Does
This is a **{}** project built with Rust and the QuantumAI ML framework.

## How to Run
```bash
# From this project's directory:
cargo run --release

# Run tests:
cargo test
```

Note: this project depends on the local `quantumai` crate (see Cargo.toml),
so it must live alongside the `quantumai/` directory in the quantum-lang
workspace (or update the path dependency to point at your QuantumAI checkout).

## Project Steps
{}

## What the Model Does
{}

## Files
- `src/main.rs` — main application (Rust + QuantumAI)
- `tests/test_model.rs` — unit tests
- `Cargo.toml` — project config

## Built With
- **QuantumAI v2.1** — the ML framework
- **QuantumMind v1.0** — generated this project
"#,
            name, prompt, task,
            steps.iter().map(|s| format!("- {}", s)).collect::<Vec<_>>().join("\n"),
            match task {
                "classifier" => "Takes input features and predicts which category they belong to.",
                "regressor" => "Takes input features and predicts a continuous numerical value.",
                "anomaly" => "Learns what normal looks like, then flags anything unusual.",
                "timeseries" => "Processes sequences of data to find temporal patterns.",
                _ => "Processes input data to produce predictions.",
            })
    }

    fn generate_tests(&self, task: &str) -> String {
        match task {
            "classifier" => r#"// Unit tests for classifier
use quantumai::*;

#[test]
fn test_model_loads() {
    let model = Model::create("classifier");
    assert!(model.inner.n_params > 0, "Model should have parameters");
}

#[test]
fn test_prediction_shape() {
    let ds = DataPipeline::load("iris").normalize().dataset();
    let mut model = Model::create("classifier");
    model.train(&ds.x, &ds.y, 10, false);
    let sample = Tensor::new(ds.x.data[..4].to_vec(), vec![1, 4]);
    let pred = model.predict(&sample);
    assert_eq!(pred.shape[0], 1, "Prediction should have batch=1");
}

#[test]
fn test_accuracy_reasonable() {
    let ds = DataPipeline::load("iris").normalize().dataset();
    let (train, test) = ds.train_test_split(0.2);
    let mut model = Model::create("classifier");
    model.lr = 0.01;
    model.train(&train.x, &train.y, 100, false);
    let ev = model.evaluate(&test.x, &test.y);
    assert!(ev["accuracy"] > 0.5, "Accuracy should be > 50%");
}
"#.to_string(),
            "anomaly" => r#"// Unit tests for anomaly detector
use quantumai::*;

#[test]
fn test_detector_trains() {
    let normal = DataPipeline::load("iris").normalize().dataset();
    let mut detector = AnomalyDetector::new(normal.n_features);
    detector.train_on_normal(&normal.x, 20);
    assert!(detector.threshold > 0.0, "Threshold should be positive after training");
}

#[test]
fn test_anomaly_scores_higher_than_normal() {
    let normal = DataPipeline::load("iris").normalize().dataset();
    let mut detector = AnomalyDetector::new(normal.n_features);
    detector.train_on_normal(&normal.x, 40);

    let nf = normal.n_features;
    let normal_sample = Tensor::new(normal.x.data[..nf].to_vec(), vec![1, nf]);
    let anomaly_sample = Tensor::new(vec![999.0; nf], vec![1, nf]);

    let normal_score = detector.score(&normal_sample);
    let anomaly_score = detector.score(&anomaly_sample);
    assert!(anomaly_score > normal_score, "Injected outlier should score higher than a normal sample");
}
"#.to_string(),
            _ => r#"use quantumai::*;

#[test]
fn test_basic() {
    let ds = DataPipeline::load("iris").normalize().dataset();
    assert!(ds.n_samples > 0, "Dataset should not be empty");
}
"#.to_string(),
        }
    }

    /// Generate a single function from description
    pub fn generate_function(&self, description: &str) -> String {
        let d = description.to_lowercase();
        if d.contains("fibonacci") {
            return r#"fn fibonacci(n: int) -> int {
    if n <= 1 {
        return n
    }
    return fibonacci(n - 1) + fibonacci(n - 2)
}"#.to_string();
        }
        if d.contains("factorial") {
            return r#"fn factorial(n: int) -> int {
    if n <= 1 { return 1 }
    return n * factorial(n - 1)
}"#.to_string();
        }
        if d.contains("sort") || d.contains("bubble sort") {
            return r#"fn bubble_sort(arr: [int]) -> [int] {
    let n = arr.len()
    for i in 0..n {
        for j in 0..(n - i - 1) {
            if arr[j] > arr[j + 1] {
                let temp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = temp
            }
        }
    }
    return arr
}"#.to_string();
        }
        if d.contains("is prime") || d.contains("prime check") {
            return r#"fn is_prime(n: int) -> bool {
    if n < 2 { return false }
    if n == 2 { return true }
    if n % 2 == 0 { return false }
    let mut i = 3
    while i * i <= n {
        if n % i == 0 { return false }
        i = i + 2
    }
    return true
}"#.to_string();
        }
        if d.contains("train") || d.contains("classifier") {
            return r#"fn train_classifier(data: Dataset, epochs: int) -> Model {
    let (train, test) = data.train_test_split(0.2)
    let mut model = Model::create("classifier")
    model.lr = 0.01
    model.train(train.x, train.y, epochs, true)
    let ev = model.evaluate(test.x, test.y)
    println("Accuracy: " + ev["accuracy"])
    return model
}"#.to_string();
        }
        // Generic function template
        format!(r#"// Function: {}
fn my_function(input: int) -> int {{
    // TODO: implement {}
    return input
}}"#, description, description)
    }

    /// Explain what a piece of code does in plain English
    pub fn explain_code(&self, code: &str) -> String {
        let c = code.to_lowercase();
        let mut explanation = Vec::new();

        if c.contains("fn ") { explanation.push("• Defines a function (reusable block of code)"); }
        if c.contains("let mut") { explanation.push("• Creates a mutable variable (can change its value)"); }
        if c.contains("let ") { explanation.push("• Creates a variable to store a value"); }
        if c.contains("for ") { explanation.push("• Loops through a range or collection"); }
        if c.contains("while ") { explanation.push("• Keeps repeating while a condition is true"); }
        if c.contains("if ") { explanation.push("• Makes a decision based on a condition"); }
        if c.contains("return") { explanation.push("• Returns a result from the function"); }
        if c.contains("model::create") { explanation.push("• Creates a new AI model"); }
        if c.contains("model.train") || c.contains(".train(") { explanation.push("• Trains the AI model on data"); }
        if c.contains("model.predict") || c.contains(".predict(") { explanation.push("• Makes predictions using the trained model"); }
        if c.contains("datapipeline") { explanation.push("• Loads and prepares data for training"); }
        if c.contains("benchmarkreport") { explanation.push("• Measures and reports model performance"); }
        if c.contains("sequential") { explanation.push("• Builds a neural network layer by layer"); }
        if c.contains("println") { explanation.push("• Prints output to the screen"); }

        if explanation.is_empty() {
            return "This code performs computations or operations on data.".to_string();
        }

        format!("This code:\n{}", explanation.join("\n"))
    }

    /// Fix a code error
    pub fn fix_code(&self, code: &str, error: &str) -> String {
        if let Some(fix) = self.knowledge.fix_error(error) {
            return format!("Error: {}\n\nFix: {}\n\nYour code needs: {}", error, fix,
                if error.contains("undefined") { "declare the variable with 'let'" }
                else if error.contains("type") { "convert the type or use the correct type" }
                else if error.contains("borrow") { "add 'mut' keyword" }
                else { "check the syntax" });
        }
        format!("Could not automatically fix: {}\nTry: check syntax, variable names, and types.", error)
    }
}
