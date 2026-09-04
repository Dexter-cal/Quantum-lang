//! Reasoning Engine — thinks from first principles, not just pattern matching
//! This is what makes QuantumMind different from typical LLMs.
//! Instead of predicting the next token, it applies logical rules to derive answers.

use std::collections::HashMap;

/// A logical rule: IF conditions THEN conclusion
#[derive(Debug, Clone)]
pub struct Rule {
    pub id: String,
    pub conditions: Vec<String>,
    pub conclusion: String,
    pub action: Option<String>,  // code to generate or action to take
    pub confidence: f64,
}

/// Result of reasoning
#[derive(Debug, Clone)]
pub struct ReasoningResult {
    pub answer: String,
    pub confidence: f64,
    pub reasoning_steps: Vec<String>,
    pub suggested_code: Option<String>,
    pub follow_up: Vec<String>,
}

pub struct ReasoningEngine {
    pub rules: Vec<Rule>,
    pub working_memory: HashMap<String, String>,
}

impl ReasoningEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            rules: Vec::new(),
            working_memory: HashMap::new(),
        };
        engine.load_rules();
        engine
    }

    fn add_rule(&mut self, id: &str, conditions: Vec<&str>, conclusion: &str, action: Option<&str>) {
        self.rules.push(Rule {
            id: id.to_string(),
            conditions: conditions.iter().map(|s| s.to_string()).collect(),
            conclusion: conclusion.to_string(),
            action: action.map(|s| s.to_string()),
            confidence: 1.0,
        });
    }

    fn load_rules(&mut self) {
        // ── Problem diagnosis rules ──────────────────────────────────
        self.add_rule("R001",
            vec!["accuracy < 0.6", "loss > 1.0"],
            "Model is not learning. Data may not be normalized or learning rate is wrong.",
            Some("Try: DataPipeline::load(data).normalize() and learning_rate(0.01)"));

        self.add_rule("R002",
            vec!["train_accuracy > 0.95", "test_accuracy < 0.7"],
            "Model is overfitting — memorizing training data but failing on new data.",
            Some("Add Dropout(0.3) between layers and collect more training data"));

        self.add_rule("R003",
            vec!["loss not decreasing", "epochs > 50"],
            "Model is stuck in a local minimum. Try a different optimizer or learning rate.",
            Some("Change optimizer to 'adamw' or increase learning rate to 0.01"));

        self.add_rule("R004",
            vec!["all predictions same class"],
            "Model output collapsed — all samples predict same class.",
            Some("Remove BatchNorm layers, use lr=0.01, train longer"));

        self.add_rule("R005",
            vec!["dataset < 100 samples"],
            "Dataset is very small. Use cross-validation for reliable estimates.",
            Some("CrossValidation::run(&dataset, \"classifier\", 100, 5)"));

        // ── Architecture selection rules ─────────────────────────────
        self.add_rule("R010",
            vec!["task = image classification"],
            "Use pretrained EfficientNet or ResNet for image classification.",
            Some("Model::load(\"efficientnet\", true)"));

        self.add_rule("R011",
            vec!["task = text classification", "data = text"],
            "Use BERT-tiny or create NLP model for text classification.",
            Some("Model::create(\"nlp\") or Model::load(\"bert-tiny\", true)"));

        self.add_rule("R012",
            vec!["task = time series", "data = sequential"],
            "Use LSTM for time series — it captures temporal dependencies.",
            Some("LSTM::new(input_size, 64, 2)"));

        self.add_rule("R013",
            vec!["task = anomaly detection"],
            "Use AnomalyDetector (autoencoder) trained only on normal data.",
            Some("AnomalyDetector::new(n_features).train_on_normal(normal_data, 100)"));

        self.add_rule("R014",
            vec!["task = clustering", "no labels"],
            "Use KMeans for known number of clusters, DBSCAN for unknown.",
            Some("KMeans::new(k) or DBSCAN::new(0.5, 5)"));

        self.add_rule("R015",
            vec!["task = regression", "output = continuous"],
            "Use Model::create(\"regressor\") with MSE loss.",
            Some("Model::create(\"regressor\").epochs(200).learning_rate(0.01)"));

        self.add_rule("R016",
            vec!["task = generation", "need new samples"],
            "Use GAN for generating new data samples.",
            Some("GAN::new(latent_dim, output_dim, hidden_dim)"));

        // ── Data rules ───────────────────────────────────────────────
        self.add_rule("R020",
            vec!["data not normalized", "features different scales"],
            "Always normalize data before training. Features on different scales confuse the model.",
            Some("DataPipeline::load(data).normalize()"));

        self.add_rule("R021",
            vec!["class imbalance", "accuracy misleading"],
            "Use F1 score not accuracy when classes are imbalanced.",
            Some("Check BenchmarkReport F1 scores per class"));

        self.add_rule("R022",
            vec!["need more data", "small dataset"],
            "Try data augmentation or synthetic data generation with make_classification.",
            Some("data::make_classification(500, n_features, n_classes)"));

        // ── Performance rules ────────────────────────────────────────
        self.add_rule("R030",
            vec!["model too slow", "inference > 10ms"],
            "Optimize model: reduce layers/neurons or use quantization.",
            Some("model.optimize(&[\"quantize_int8\", \"prune_0.3\"])"));

        self.add_rule("R031",
            vec!["model too large", "size > 100MB"],
            "Model is too large for edge deployment. Use knowledge distillation.",
            Some("Use a smaller architecture: Model::create with fewer layers"));

        // ── Best practice rules ──────────────────────────────────────
        self.add_rule("R040",
            vec!["new project", "starting from scratch"],
            "Always start with AutoMLPipeline to find best settings automatically.",
            Some("AutoMLPipeline::run(dataset, task, classes, features)"));

        self.add_rule("R041",
            vec!["hyperparameters unknown"],
            "Use tune() function to automatically search for best hyperparameters.",
            Some("let best = tune(&dataset, 10)"));

        self.add_rule("R042",
            vec!["model trained", "before deployment"],
            "Always run BenchmarkReport and CrossValidation before deploying.",
            Some("BenchmarkReport::run(...) then CrossValidation::run(...)"));
    }

    /// Reason about a situation and return conclusions
    pub fn reason(&self, facts: &[String]) -> Vec<ReasoningResult> {
        let mut results = Vec::new();
        let facts_lower: Vec<String> = facts.iter().map(|f| f.to_lowercase()).collect();

        for rule in &self.rules {
            let matches = rule.conditions.iter().all(|cond| {
                facts_lower.iter().any(|fact| fact.contains(cond.as_str()))
            });

            if matches {
                let steps = vec![
                    format!("Observed: {}", facts.join(", ")),
                    format!("Applied rule {}: IF {} THEN {}", rule.id,
                        rule.conditions.join(" AND "), rule.conclusion),
                ];
                results.push(ReasoningResult {
                    answer: rule.conclusion.clone(),
                    confidence: rule.confidence,
                    reasoning_steps: steps,
                    suggested_code: rule.action.clone(),
                    follow_up: vec![],
                });
            }
        }
        results
    }

    /// Reason about a task type from description
    pub fn identify_task(&self, description: &str) -> (&str, f64) {
        let d = description.to_lowercase();
        // Check anomaly/fraud BEFORE classifier: "fraud detection" and
        // "anomaly detection" both contain "detect", which would otherwise
        // be caught by the classifier check below.
        if d.contains("anomaly") || d.contains("fraud") || d.contains("unusual") || d.contains("outlier") { return ("anomaly", 0.9); }
        if d.contains("classif") || d.contains("categor") || d.contains("detect") || d.contains("spam") || d.contains("sentiment") { return ("classifier", 0.9); }
        if d.contains("regress") || d.contains("predict.*number") || d.contains("price") || d.contains("forecast.*number") { return ("regressor", 0.9); }
        if d.contains("cluster") || d.contains("group") || d.contains("segment") { return ("clustering", 0.9); }
        if d.contains("generat") || d.contains("create.*image") || d.contains("synthetic") { return ("generative", 0.85); }
        if d.contains("time series") || d.contains("sequence") || d.contains("forecast") { return ("timeseries", 0.9); }
        if d.contains("image") || d.contains("photo") || d.contains("picture") { return ("image_classification", 0.9); }
        if d.contains("text") || d.contains("nlp") || d.contains("language") || d.contains("review") { return ("nlp", 0.9); }
        ("classifier", 0.5)  // default
    }

    /// Break a complex problem into sub-problems (divide and conquer)
    pub fn decompose_problem(&self, problem: &str) -> Vec<String> {
        let d = problem.to_lowercase();
        let mut steps = Vec::new();

        if d.contains("train") || d.contains("model") || d.contains("predict") {
            steps.push("1. Load and analyze your data (DataQualityReport)".to_string());
            steps.push("2. Clean and normalize the data (DataPipeline)".to_string());
            steps.push("3. Split into train/test sets (80/20)".to_string());
            steps.push("4. Search for best hyperparameters (tune())".to_string());
            steps.push("5. Train the model (Model::create or Sequential)".to_string());
            steps.push("6. Evaluate with full benchmark (BenchmarkReport)".to_string());
            steps.push("7. Get AI advice (AIAssistant::analyze)".to_string());
            steps.push("8. Save and deploy (model.save + model.serve)".to_string());
        } else if d.contains("project") || d.contains("build") || d.contains("create.*app") {
            steps.push("1. Define the problem clearly (what input? what output?)".to_string());
            steps.push("2. Collect or generate training data".to_string());
            steps.push("3. Run AutoMLPipeline for automatic setup".to_string());
            steps.push("4. Refine the best model further".to_string());
            steps.push("5. Build the application around the model".to_string());
            steps.push("6. Test with real data".to_string());
            steps.push("7. Deploy with model.serve()".to_string());
        } else {
            steps.push("1. Understand the problem requirements".to_string());
            steps.push("2. Identify the right approach".to_string());
            steps.push("3. Implement step by step".to_string());
            steps.push("4. Test and verify".to_string());
        }
        steps
    }

    /// Store a fact in working memory for multi-turn reasoning
    pub fn remember(&mut self, key: &str, value: &str) {
        self.working_memory.insert(key.to_string(), value.to_string());
    }

    /// Recall a fact from working memory
    pub fn recall(&self, key: &str) -> Option<&String> {
        self.working_memory.get(key)
    }
}
