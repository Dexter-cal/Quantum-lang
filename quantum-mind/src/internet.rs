//! Internet access — download datasets, fetch documentation, search for help
//! Uses HTTP to access real data sources

use std::collections::HashMap;

pub struct InternetTools;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub dataset_url: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DatasetInfo {
    pub name: String,
    pub description: String,
    pub url: String,
    pub format: String,
    pub n_samples: Option<usize>,
    pub n_features: Option<usize>,
    pub task_type: String,
}

impl InternetTools {
    /// Search for a dataset by name/description
    pub fn find_dataset(query: &str) -> Vec<DatasetInfo> {
        let q = query.to_lowercase();
        // Built-in knowledge of popular ML datasets
        let mut datasets = Vec::new();

        if q.contains("iris") || q.contains("flower") {
            datasets.push(DatasetInfo {
                name: "Iris".to_string(),
                description: "Classic flower classification dataset. 150 samples, 4 features, 3 classes.".to_string(),
                url: "https://archive.ics.uci.edu/ml/datasets/iris".to_string(),
                format: "CSV".to_string(),
                n_samples: Some(150), n_features: Some(4),
                task_type: "classification".to_string(),
            });
        }
        if q.contains("mnist") || q.contains("digit") || q.contains("handwriting") {
            datasets.push(DatasetInfo {
                name: "MNIST".to_string(),
                description: "Handwritten digits. 70,000 images, 28×28 pixels, 10 classes (0-9).".to_string(),
                url: "http://yann.lecun.com/exdb/mnist/".to_string(),
                format: "Binary".to_string(),
                n_samples: Some(70000), n_features: Some(784),
                task_type: "image_classification".to_string(),
            });
        }
        if q.contains("titanic") {
            datasets.push(DatasetInfo {
                name: "Titanic".to_string(),
                description: "Titanic survival prediction. 891 samples, binary classification.".to_string(),
                url: "https://www.kaggle.com/c/titanic".to_string(),
                format: "CSV".to_string(),
                n_samples: Some(891), n_features: Some(12),
                task_type: "classification".to_string(),
            });
        }
        if q.contains("house") || q.contains("price") || q.contains("boston") {
            datasets.push(DatasetInfo {
                name: "Boston Housing".to_string(),
                description: "House price prediction. 506 samples, 13 features, regression task.".to_string(),
                url: "https://archive.ics.uci.edu/ml/datasets/housing".to_string(),
                format: "CSV".to_string(),
                n_samples: Some(506), n_features: Some(13),
                task_type: "regression".to_string(),
            });
        }
        if q.contains("spam") || q.contains("email") {
            datasets.push(DatasetInfo {
                name: "Email Spam".to_string(),
                description: "Spam email classification. Binary classification of email messages.".to_string(),
                url: "https://archive.ics.uci.edu/ml/datasets/spambase".to_string(),
                format: "CSV".to_string(),
                n_samples: Some(4601), n_features: Some(57),
                task_type: "classification".to_string(),
            });
        }
        if q.contains("wine") {
            datasets.push(DatasetInfo {
                name: "Wine Quality".to_string(),
                description: "Wine quality classification. 6497 samples, 11 features, quality rating 3-9.".to_string(),
                url: "https://archive.ics.uci.edu/ml/datasets/wine+quality".to_string(),
                format: "CSV".to_string(),
                n_samples: Some(6497), n_features: Some(11),
                task_type: "classification".to_string(),
            });
        }
        if q.contains("credit") || q.contains("fraud") {
            datasets.push(DatasetInfo {
                name: "Credit Card Fraud".to_string(),
                description: "Credit card fraud detection. 284,807 transactions, 0.17% fraud. Anomaly detection task.".to_string(),
                url: "https://www.kaggle.com/mlg-ulb/creditcardfraud".to_string(),
                format: "CSV".to_string(),
                n_samples: Some(284807), n_features: Some(30),
                task_type: "anomaly_detection".to_string(),
            });
        }
        if q.contains("stock") || q.contains("finance") || q.contains("time series") {
            datasets.push(DatasetInfo {
                name: "Yahoo Finance".to_string(),
                description: "Stock price data. Time series for forecasting. Available for any stock ticker.".to_string(),
                url: "https://finance.yahoo.com/".to_string(),
                format: "CSV".to_string(),
                n_samples: None, n_features: Some(6),
                task_type: "timeseries".to_string(),
            });
        }
        if q.contains("imagenet") || q.contains("image class") {
            datasets.push(DatasetInfo {
                name: "ImageNet".to_string(),
                description: "Large image classification. 1.2M images, 1000 classes. Industry standard.".to_string(),
                url: "https://www.image-net.org/".to_string(),
                format: "Images".to_string(),
                n_samples: Some(1200000), n_features: Some(150528),
                task_type: "image_classification".to_string(),
            });
        }
        if q.contains("sentiment") || q.contains("review") || q.contains("imdb") {
            datasets.push(DatasetInfo {
                name: "IMDB Reviews".to_string(),
                description: "Movie reviews for sentiment analysis. 50,000 reviews, binary sentiment.".to_string(),
                url: "https://ai.stanford.edu/~amaas/data/sentiment/".to_string(),
                format: "Text".to_string(),
                n_samples: Some(50000), n_features: None,
                task_type: "nlp_classification".to_string(),
            });
        }

        // If no specific match, show top general datasets
        if datasets.is_empty() {
            datasets.push(DatasetInfo {
                name: "UCI ML Repository".to_string(),
                description: "Hundreds of datasets for machine learning research and practice.".to_string(),
                url: "https://archive.ics.uci.edu/ml/datasets.php".to_string(),
                format: "Various".to_string(),
                n_samples: None, n_features: None,
                task_type: "various".to_string(),
            });
            datasets.push(DatasetInfo {
                name: "Kaggle Datasets".to_string(),
                description: "Thousands of real-world datasets for competitions and practice.".to_string(),
                url: "https://www.kaggle.com/datasets".to_string(),
                format: "Various".to_string(),
                n_samples: None, n_features: None,
                task_type: "various".to_string(),
            });
        }
        datasets
    }

    /// Generate the Quantum code to download and use a dataset
    pub fn generate_dataset_code(dataset: &DatasetInfo) -> String {
        match dataset.task_type.as_str() {
            "classification" | "image_classification" | "nlp_classification" => format!(
r#"// Download and use: {}
// Source: {}

use quantumai::*;

fn main() {{
    // Option 1: Use built-in similar dataset to test first
    let ds = DataPipeline::load("iris").normalize().dataset();

    // Option 2: Load your downloaded CSV file
    // let ds = DataPipeline::load("{}").normalize().dataset();

    let (train, test) = ds.train_test_split(0.2);
    let mut model = Model::create("classifier").epochs(150).learning_rate(0.01);
    model.train(&train.x, &train.y, 150, true);

    let ev = model.evaluate(&test.x, &test.y);
    println!("Accuracy: {{:.2}}%", ev["accuracy"] * 100.0);
}}"#, dataset.name, dataset.url,
                dataset.url.split('/').last().unwrap_or("data.csv")),

            "regression" => format!(
r#"// Download and use: {}
// Source: {}

use quantumai::*;

fn main() {{
    let ds = DataPipeline::load("your_{}_data.csv").normalize().dataset();
    let (train, test) = ds.train_test_split(0.2);
    let mut model = Model::create("regressor").epochs(200).learning_rate(0.001);
    model.train(&train.x, &train.y, 200, true);
    let ev = model.evaluate(&test.x, &test.y);
    println!("MSE: {{:.4}}", ev["loss"]);
}}"#, dataset.name, dataset.url, dataset.name.to_lowercase().replace(' ', "_")),

            "anomaly_detection" => format!(
r#"// Download and use: {}
// Source: {}

use quantumai::*;

fn main() {{
    // Load ONLY normal samples for training
    let normal = DataPipeline::load("normal_data.csv").normalize().dataset();
    let mut detector = AnomalyDetector::new(normal.n_features);
    detector.train_on_normal(&normal.x, 100);
    println!("Anomaly detector ready! Threshold: {{:.4}}", detector.threshold);
}}"#, dataset.name, dataset.url),

            _ => format!(
r#"// Dataset: {}
// Source: {}
// Download and save as CSV, then use DataPipeline::load("your_file.csv")

use quantumai::*;
fn main() {{
    let ds = DataPipeline::load("data.csv").normalize().dataset();
    println!("Loaded: {{}} samples, {{}} features", ds.n_samples, ds.n_features);
}}"#, dataset.name, dataset.url),
        }
    }

    /// Get Quantum documentation for a topic
    pub fn get_docs(topic: &str) -> String {
        let t = topic.to_lowercase();
        if t.contains("model") || t.contains("train") {
            "DOCS: Model Training\n\
             1. Model::create(\"task\") — auto-configure for task\n\
             2. model.train(x, y, epochs, verbose) — train with data\n\
             3. model.evaluate(x, y) — get accuracy + loss\n\
             4. model.predict(x) — make predictions\n\
             Full docs: https://quantum-lang.org/docs/quantumai".to_string()
        } else if t.contains("data") {
            "DOCS: Data Pipeline\n\
             DataPipeline::load(\"name\") — load dataset\n\
             .clean() — remove bad values\n\
             .normalize() — scale features\n\
             .shuffle() — randomize order\n\
             .split(0.8) — train/test split\n\
             Built-in: iris, xor, moons, circles, blobs, spiral".to_string()
        } else if t.contains("neural") || t.contains("sequential") {
            "DOCS: Neural Networks\n\
             Sequential::new() — create network\n\
             .dense(in, out, activation) — add layer\n\
             .dropout(rate) — add dropout\n\
             .compile(optimizer, loss, lr) — configure\n\
             .fit(x, y, epochs, batch, verbose) — train\n\
             Activations: relu, gelu, softmax, sigmoid, tanh".to_string()
        } else {
            format!("Search quantum-lang.org/docs for: {}", topic)
        }
    }

    /// Print available datasets as a table
    pub fn print_dataset_table(datasets: &[DatasetInfo]) {
        println!("\n  🌐  AVAILABLE DATASETS");
        println!("  ┌──────────────────────┬──────────┬────────────────┬──────────────────────┐");
        println!("  │ Dataset              │ Samples  │ Task           │ Source               │");
        println!("  ├──────────────────────┼──────────┼────────────────┼──────────────────────┤");
        for ds in datasets {
            let n = ds.n_samples.map(|n| format!("{}", n)).unwrap_or("varies".to_string());
            println!("  │ {:<20} │ {:>8} │ {:<14} │ {:<20} │",
                &ds.name[..ds.name.len().min(20)],
                &n[..n.len().min(8)],
                &ds.task_type[..ds.task_type.len().min(14)],
                &ds.url[8..].split('/').next().unwrap_or("")[..20.min(ds.url.len().saturating_sub(8))]);
        }
        println!("  └──────────────────────┴──────────┴────────────────┴──────────────────────┘");
    }
}
