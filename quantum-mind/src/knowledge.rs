//! Knowledge base — everything QuantumMind knows about Quantum Language
//! This is what makes it smart at Quantum specifically.
//! Unlike LLMs that pattern-match, this stores FACTS and RULES.

use std::collections::HashMap;

/// A fact in the knowledge base
#[derive(Debug, Clone)]
pub struct Fact {
    pub id: String,
    pub category: FactCategory,
    pub content: String,
    pub example: Option<String>,
    pub related: Vec<String>,
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FactCategory {
    Syntax,           // Language syntax rules
    QuantumAI,        // QuantumAI framework
    ErrorPattern,     // Common errors + fixes
    BestPractice,     // How to write good Quantum code
    Algorithm,        // ML algorithms
    ProjectPattern,   // Full project templates
    Tool,             // Available tools
    Dataset,          // Known datasets
    ConceptDefinition,// What things mean
}

/// The complete knowledge base for QuantumMind
pub struct KnowledgeBase {
    pub facts: Vec<Fact>,
    pub index: HashMap<String, Vec<usize>>,  // keyword → fact indices
    pub error_fixes: HashMap<String, String>, // error message → fix
    pub code_templates: HashMap<String, String>, // task → code template
    pub concept_definitions: HashMap<String, String>, // concept → plain English definition
}

impl KnowledgeBase {
    pub fn new() -> Self {
        let mut kb = Self {
            facts: Vec::new(),
            index: HashMap::new(),
            error_fixes: HashMap::new(),
            code_templates: HashMap::new(),
            concept_definitions: HashMap::new(),
        };
        kb.load_quantum_syntax();
        kb.load_quantumai_knowledge();
        kb.load_error_patterns();
        kb.load_code_templates();
        kb.load_concept_definitions();
        kb.load_project_patterns();
        kb
    }

    fn add_fact(&mut self, id: &str, cat: FactCategory, content: &str, example: Option<&str>, related: Vec<&str>) {
        let idx = self.facts.len();
        // Index by words in content
        for word in content.split_whitespace() {
            let key = word.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string();
            if key.len() > 2 {
                self.index.entry(key).or_insert_with(Vec::new).push(idx);
            }
        }
        self.facts.push(Fact {
            id: id.to_string(),
            category: cat,
            content: content.to_string(),
            example: example.map(|s| s.to_string()),
            related: related.iter().map(|s| s.to_string()).collect(),
            confidence: 1.0,
        });
    }

    fn load_quantum_syntax(&mut self) {
        self.add_fact("syn001", FactCategory::Syntax,
            "Declare a function with fn keyword followed by name, parameters, return type",
            Some("fn add(a: int, b: int) -> int { return a + b }"),
            vec!["syn002", "syn003"]);

        self.add_fact("syn002", FactCategory::Syntax,
            "Declare a variable with let keyword. Use mut for mutable variables",
            Some("let x = 42\nlet mut count = 0"),
            vec!["syn001"]);

        self.add_fact("syn003", FactCategory::Syntax,
            "If-else conditional uses if condition { } else { } syntax",
            Some("if x > 0 { println(\"positive\") } else { println(\"negative\") }"),
            vec!["syn001"]);

        self.add_fact("syn004", FactCategory::Syntax,
            "For loop iterates with for variable in range syntax",
            Some("for i in 0..10 { println(i) }"),
            vec!["syn005"]);

        self.add_fact("syn005", FactCategory::Syntax,
            "While loop repeats while condition is true",
            Some("let mut i = 0\nwhile i < 10 { i = i + 1 }"),
            vec!["syn004"]);

        self.add_fact("syn006", FactCategory::Syntax,
            "Import modules with import keyword. Use as for aliases",
            Some("import quantumai as qai\nimport std::collections::HashMap"),
            vec!["syn001"]);

        self.add_fact("syn007", FactCategory::Syntax,
            "String concatenation uses + operator. F-strings use f\"text {variable}\"",
            Some("let name = \"World\"\nprintln(\"Hello \" + name)\nprintln(f\"Hello {name}\")"),
            vec!["syn002"]);

        self.add_fact("syn008", FactCategory::Syntax,
            "Struct defines a custom data type with named fields",
            Some("struct Point { x: float, y: float }\nlet p = Point { x: 1.0, y: 2.0 }"),
            vec!["syn009"]);

        self.add_fact("syn009", FactCategory::Syntax,
            "Enum defines a type with multiple variants",
            Some("enum Color { Red, Green, Blue }\nlet c = Color::Red"),
            vec!["syn008"]);

        self.add_fact("syn010", FactCategory::Syntax,
            "Match expression pattern matches on values like a switch statement",
            Some("match x {\n    0 => println(\"zero\"),\n    1..10 => println(\"small\"),\n    _ => println(\"large\")\n}"),
            vec!["syn003"]);

        self.add_fact("syn011", FactCategory::Syntax,
            "Return statement exits a function with a value",
            Some("fn square(n: int) -> int { return n * n }"),
            vec!["syn001"]);

        self.add_fact("syn012", FactCategory::Syntax,
            "Types: int, float, bool, string, char. Arrays: [T]. Tuples: (A, B)",
            Some("let n: int = 42\nlet f: float = 3.14\nlet b: bool = true\nlet s: string = \"hello\""),
            vec!["syn002"]);
    }

    fn load_quantumai_knowledge(&mut self) {
        self.add_fact("qai001", FactCategory::QuantumAI,
            "Create a model for any task with Model::create(task)",
            Some("let mut model = Model::create(\"classifier\")"),
            vec!["qai002"]);

        self.add_fact("qai002", FactCategory::QuantumAI,
            "Train a model with model.train(x, y, epochs, verbose)",
            Some("model.train(&data.x, &data.y, 100, true)"),
            vec!["qai001", "qai003"]);

        self.add_fact("qai003", FactCategory::QuantumAI,
            "Load built-in datasets with DataPipeline::load(name). Options: iris, xor, moons, circles, blobs, spiral",
            Some("let ds = DataPipeline::load(\"iris\").normalize().dataset()"),
            vec!["qai002"]);

        self.add_fact("qai004", FactCategory::QuantumAI,
            "Evaluate model performance with model.evaluate(x, y) returns accuracy and loss",
            Some("let ev = model.evaluate(&test.x, &test.y);\nprintln(ev[\"accuracy\"])"),
            vec!["qai002"]);

        self.add_fact("qai005", FactCategory::QuantumAI,
            "Make predictions with model.predict(x) or model.predict_class(x) for class index",
            Some("let pred = model.predict(&sample)\nlet cls = model.predict_class(&sample)"),
            vec!["qai004"]);

        self.add_fact("qai006", FactCategory::QuantumAI,
            "Full benchmark report with BenchmarkReport::run shows accuracy, speed, confusion matrix",
            Some("let report = BenchmarkReport::run(&mut model, &tr.x, &tr.y, &te.x, &te.y, 3, &[\"a\",\"b\",\"c\"]);\nreport.print(&[\"a\",\"b\",\"c\"])"),
            vec!["qai004"]);

        self.add_fact("qai007", FactCategory::QuantumAI,
            "AutoML pipeline: one call does everything — load, clean, tune, train, report",
            Some("let auto = AutoMLPipeline::run(\"iris\", \"classifier\", &classes, &features)"),
            vec!["qai001", "qai002"]);

        self.add_fact("qai008", FactCategory::QuantumAI,
            "Sequential neural network: add layers then compile and fit",
            Some("let mut m = Sequential::new();\nm.dense(4, 32, \"relu\");\nm.dense(32, 3, \"softmax\");\nm.compile(\"adam\", \"cross_entropy\", 0.01);\nm.fit(&x, &y, 100, 32, true)"),
            vec!["qai009"]);

        self.add_fact("qai009", FactCategory::QuantumAI,
            "Optimizers available: adam, adamw, sgd, rmsprop, adagrad, nadam",
            Some("m.compile(\"adamw\", \"cross_entropy\", 0.001)"),
            vec!["qai008"]);

        self.add_fact("qai010", FactCategory::QuantumAI,
            "Loss functions: cross_entropy, mse, mae, binary_crossentropy, huber, focal",
            Some("m.compile(\"adam\", \"mse\", 0.01) // for regression"),
            vec!["qai008"]);

        self.add_fact("qai011", FactCategory::QuantumAI,
            "Save and load model weights with model.save and model.load_weights",
            Some("model.save(\"/tmp/my_model.json\")\nmodel.load_weights(\"/tmp/my_model.json\")"),
            vec!["qai002"]);

        self.add_fact("qai012", FactCategory::QuantumAI,
            "LSTM for sequence data: LSTM::new(input_size, hidden_size, num_layers)",
            Some("let lstm = LSTM::new(4, 64, 2);\nlet (out, cell) = lstm.forward(&input)"),
            vec!["qai013"]);

        self.add_fact("qai013", FactCategory::QuantumAI,
            "GRU is a lighter alternative to LSTM for sequences",
            Some("let gru = GRU::new(4, 32);\nlet out = gru.forward(&input)"),
            vec!["qai012"]);

        self.add_fact("qai014", FactCategory::QuantumAI,
            "Anomaly detection with AnomalyDetector trains on normal data then scores new samples 0-100",
            Some("let mut det = AnomalyDetector::new(4);\ndet.train_on_normal(&normal_data, 100);\nprintln(det.describe(&new_sample))"),
            vec!["qai007"]);

        self.add_fact("qai015", FactCategory::QuantumAI,
            "Cross-validation with CrossValidation::run gives reliable accuracy estimate",
            Some("let cv = CrossValidation::run(&dataset, \"classifier\", 100, 5);\nprintln(cv.mean)"),
            vec!["qai004"]);
    }

    fn load_error_patterns(&mut self) {
        self.error_fixes.insert(
            "undefined variable".to_string(),
            "You used a variable that was not declared. Add 'let name = value' before using it.".to_string());
        self.error_fixes.insert(
            "type mismatch".to_string(),
            "You passed the wrong type to a function. Check what type is expected and convert if needed.".to_string());
        self.error_fixes.insert(
            "index out of bounds".to_string(),
            "You tried to access array index that doesn't exist. Check array length before indexing.".to_string());
        self.error_fixes.insert(
            "cannot borrow".to_string(),
            "Add 'mut' keyword: let mut variable = value. Or pass &mut reference.".to_string());
        self.error_fixes.insert(
            "expected rparen".to_string(),
            "Missing closing parenthesis ')'. Check your function calls and expressions.".to_string());
        self.error_fixes.insert(
            "expected rbrace".to_string(),
            "Missing closing brace '}'. Check your if/else/fn/loop blocks.".to_string());
        self.error_fixes.insert(
            "function not found".to_string(),
            "Function doesn't exist. Check spelling, or add 'import quantumai as qai' if it's a QuantumAI function.".to_string());
        self.error_fixes.insert(
            "model output collapsed".to_string(),
            "Model gives same output for all inputs. Fix: remove BatchNorm layers, use lr=0.01, train for more epochs.".to_string());
        self.error_fixes.insert(
            "low accuracy".to_string(),
            "Model accuracy is poor. Try: normalize data, increase epochs, use lr=0.01, add more neurons.".to_string());
        self.error_fixes.insert(
            "overfitting".to_string(),
            "Train accuracy >> test accuracy. Fix: add Dropout(0.3), collect more data, reduce model size.".to_string());
        self.error_fixes.insert(
            "accuracy is 0.33".to_string(),
            "Output collapse! Remove BatchNorm layers. Use lr=0.01. Train 150+ epochs. Normalize data.".to_string());
        self.error_fixes.insert(
            "accuracy 0.33".to_string(),
            "Output collapse! All samples predict same class. Fix: remove BatchNorm, use lr=0.01, train longer.".to_string());
        self.error_fixes.insert(
            "not improving".to_string(),
            "Model stuck. Try: lr=0.01, optimizer='adam', normalize data, remove BatchNorm layers.".to_string());
        self.error_fixes.insert(
            "same class".to_string(),
            "Output collapsed — model always predicts class 0. Remove BatchNorm. Use lr=0.01. Verify labels are one-hot.".to_string());
        self.error_fixes.insert(
            "loss not decreasing".to_string(),
            "Loss is stuck. Try: increase learning rate, check data normalization, use different optimizer.".to_string());
    }

    fn load_code_templates(&mut self) {
        self.code_templates.insert("hello_world".to_string(), r#"fn main() {
    println("Hello, Quantum World!")
}"#.to_string());

        self.code_templates.insert("classifier".to_string(), r#"use quantumai::*;

fn main() {
    // Load data
    let ds = DataPipeline::load("DATASET_NAME")
        .normalize()
        .dataset();
    let (train, test) = ds.train_test_split(0.2);

    // Create and train model
    let mut model = Model::create("classifier")
        .epochs(EPOCHS)
        .learning_rate(LR);
    model.train(&train.x, &train.y, EPOCHS, true);

    // Evaluate
    let report = BenchmarkReport::run(
        &mut model,
        &train.x, &train.y,
        &test.x, &test.y,
        N_CLASSES, &CLASS_NAMES,
    );
    report.print(&CLASS_NAMES);
    AIAssistant::analyze(&report);

    // Save
    model.save("model.json");
}"#.to_string());

        self.code_templates.insert("neural_network".to_string(), r#"use quantumai::*;

fn main() {
    let ds = DataPipeline::load("DATASET").normalize().dataset();
    let (train, test) = ds.train_test_split(0.2);

    let mut net = Sequential::new();
    net.dense(N_FEATURES, 64, "relu");
    net.dropout(0.2);
    net.dense(64, 32, "relu");
    net.dense(32, N_OUTPUTS, "OUTPUT_ACT");
    net.compile("adam", "LOSS_FN", 0.01);
    net.fit(&train.x, &train.y, 150, 32, true);

    println!("Test accuracy: {:.4}", net.compute_accuracy(&test.x, &test.y));
}"#.to_string());

        self.code_templates.insert("regression".to_string(), r#"use quantumai::*;
use quantumai::ml::*;

fn main() {
    let ds = DataPipeline::load("DATASET").normalize().dataset();
    let (train, test) = ds.train_test_split(0.2);

    let mut model = Model::create("regressor")
        .epochs(200)
        .learning_rate(0.01);
    model.train(&train.x, &train.y, 200, true);

    let ev = model.evaluate(&test.x, &test.y);
    println!("Test MSE: {:.4}", ev["loss"]);
}"#.to_string());

        self.code_templates.insert("anomaly_detection".to_string(), r#"use quantumai::*;

fn main() {
    // Load normal data to train on
    let normal = DataPipeline::load("NORMAL_DATA").normalize().dataset();

    // Train anomaly detector
    let mut detector = AnomalyDetector::new(N_FEATURES);
    detector.train_on_normal(&normal.x, 100);

    // Score new samples
    let new_sample = Tensor::new(vec![VAL1, VAL2, VAL3], vec![1, N_FEATURES]);
    println!("{}", detector.describe(&new_sample));
}"#.to_string());

        self.code_templates.insert("time_series".to_string(), r#"use quantumai::*;

fn main() {
    // Time series with LSTM
    let lstm = LSTM::new(INPUT_SIZE, 64, 2);
    // input shape: [batch, sequence_length, features]
    let input = Tensor::randn(vec![BATCH, SEQ_LEN, INPUT_SIZE]);
    let (output, cell_state) = lstm.forward(&input);
    println!("LSTM output shape: {:?}", output.shape);
}"#.to_string());

        self.code_templates.insert("automl".to_string(), r#"use quantumai::*;

fn main() {
    // One call does everything: load → clean → tune → train → report
    let result = AutoMLPipeline::run(
        "DATASET",
        "classifier",
        &["class1", "class2", "class3"],
        &["feature1", "feature2", "feature3", "feature4"],
    );
    println!("Best accuracy: {:.2}%", result.best_model.evaluate(
        &DataPipeline::load("DATASET").normalize().dataset().x,
        &DataPipeline::load("DATASET").normalize().dataset().y,
    )["accuracy"] * 100.0);
}"#.to_string());

        self.code_templates.insert("full_project".to_string(), r#"// FULL PROJECT: PROJECT_NAME
// Generated by QuantumMind
// Task: TASK_DESCRIPTION

use quantumai::*;

fn main() {
    println!("=== PROJECT_NAME ===");

    // Step 1: Load and analyze data
    let pipeline = DataPipeline::load("DATASET")
        .clean()
        .normalize()
        .shuffle();
    let ds = pipeline.dataset();
    let qr = DataQualityReport::analyze(&ds.x, &ds.y, &FEATURES, &CLASSES);
    qr.print(&CLASSES);

    // Step 2: Split data
    let (train, test) = ds.train_test_split(0.2);

    // Step 3: Hyperparameter search
    let best = tune(&train, 6);

    // Step 4: Train final model
    let mut model = Model::create("TASK_TYPE");
    model.lr = best.best_lr;
    model.opt_name = best.best_optimizer;
    model.train(&train.x, &train.y, 150, true);

    // Step 5: Full benchmark
    let report = BenchmarkReport::run(
        &mut model,
        &train.x, &train.y,
        &test.x, &test.y,
        N_CLASSES, &CLASSES,
    );
    report.print(&CLASSES);

    // Step 6: AI analysis
    AIAssistant::analyze(&report);

    // Step 7: Model card
    ModelCard::from_benchmark(&model, &report).print();

    // Step 8: Feature importance
    let imps = feature_importance(&model, &test.x, &test.y, &FEATURES);

    // Step 9: Save
    model.save("PROJECT_NAME_model.json");
    println!("Model saved. Run model.serve(8080) to deploy as API.");
}"#.to_string());

        self.code_templates.insert("gan".to_string(), r#"use quantumai::*;

fn main() {
    // Generative Adversarial Network
    let mut gan = GAN::new(LATENT_DIM, OUTPUT_DIM, HIDDEN_DIM);
    gan.summary();

    // Load real data
    let real_data = DataPipeline::load("DATASET").normalize().dataset();

    // Training loop
    for epoch in 0..EPOCHS {
        let batch_size = 32;
        let (d_loss, g_loss) = gan.train_step(&real_data.x, batch_size);
        if epoch % 10 == 0 {
            println!("Epoch {}: D_loss={:.4} G_loss={:.4}", epoch, d_loss, g_loss);
        }
    }

    // Generate new samples
    let fake_samples = gan.generate(10);
    println!("Generated {} new samples", fake_samples.shape[0]);
}"#.to_string());
    }

    fn load_concept_definitions(&mut self) {
        let defs = [
            ("machine learning", "Teaching computers to learn from data without being explicitly programmed. Instead of writing rules, you show examples and the computer finds the patterns."),
            ("neural network", "A system inspired by the human brain. It has layers of connected 'neurons' that transform input data into output predictions."),
            ("training", "The process of adjusting a model's internal values (weights) so it makes better predictions. Like practicing until you get better."),
            ("accuracy", "What percentage of predictions the model got correct. 100% = perfect, 50% = random guessing for 2 classes."),
            ("loss", "A number measuring how wrong the model is. 0 = perfect. We want this to go DOWN during training."),
            ("overfitting", "When a model memorizes training data but fails on new data. Like a student who memorizes answers but can't solve new problems."),
            ("epoch", "One complete pass through all training data. More epochs = more learning (but can lead to overfitting)."),
            ("learning rate", "How big the adjustment steps are during training. Too high = unstable, too low = very slow learning. Usually 0.001 or 0.01."),
            ("batch size", "How many examples to process at once during training. Usually 32 or 64."),
            ("optimizer", "The algorithm that adjusts model weights to reduce loss. Adam is the most popular and works well for most problems."),
            ("classification", "Predicting which category something belongs to. Example: is this email spam or not spam?"),
            ("regression", "Predicting a continuous number. Example: what will this house sell for?"),
            ("cross-validation", "A technique to reliably estimate model performance by testing on different parts of data multiple times."),
            ("confusion matrix", "A table showing what the model predicted vs what was actually true. Diagonal = correct predictions."),
            ("f1 score", "A metric that balances precision and recall. Good for imbalanced datasets where accuracy can be misleading."),
            ("gradient descent", "The mathematical process of finding the best model weights by repeatedly adjusting in the direction that reduces error."),
            ("backpropagation", "How neural networks learn — errors flow backwards through the network to adjust weights."),
            ("relu", "A common activation function: outputs 0 for negative inputs, same value for positive. Simple and effective."),
            ("softmax", "Converts raw scores into probabilities that sum to 1. Used in final layer for multi-class classification."),
            ("dropout", "A regularization technique: randomly disables neurons during training to prevent overfitting."),
            ("tensor", "A multi-dimensional array of numbers. A matrix is a 2D tensor. Neural networks process data as tensors."),
            ("hyperparameter", "Settings you choose before training, like learning rate and epochs. Different from model weights which are learned automatically."),
            ("transfer learning", "Taking a model trained on one task and adapting it for a different but related task. Faster than training from scratch."),
            ("anomaly detection", "Finding unusual or suspicious data points. Used for fraud detection, quality control, etc."),
            ("clustering", "Grouping similar data points together without labels. The algorithm finds structure in the data on its own."),
        ];
        for (concept, definition) in &defs {
            self.concept_definitions.insert(concept.to_string(), definition.to_string());
        }
    }

    fn load_project_patterns(&mut self) {
        self.add_fact("proj001", FactCategory::ProjectPattern,
            "Image classification project: load images, augment, use pretrained ResNet or EfficientNet, fine-tune",
            Some("Model::load(\"efficientnet\", true) then model.train(image_data, epochs: 20)"),
            vec!["qai001"]);

        self.add_fact("proj002", FactCategory::ProjectPattern,
            "Sentiment analysis project: load text CSV, tokenize, use BERT or create NLP classifier",
            Some("Model::create(\"nlp\") or Model::load(\"bert-tiny\", true)"),
            vec!["qai001"]);

        self.add_fact("proj003", FactCategory::ProjectPattern,
            "Fraud detection project: use AnomalyDetector trained on normal transactions",
            Some("AnomalyDetector::new(n_features).train_on_normal(normal_data, 200)"),
            vec!["qai014"]);

        self.add_fact("proj004", FactCategory::ProjectPattern,
            "Time series forecasting: use LSTM with windowed data, predict next N values",
            Some("LSTM::new(1, 64, 2) with sequence windows of past values"),
            vec!["qai012"]);

        self.add_fact("proj005", FactCategory::ProjectPattern,
            "Customer segmentation: use KMeans or DBSCAN clustering on customer behavior data",
            Some("KMeans::new(5).fit(&customer_features)"),
            vec!["qai007"]);

        self.add_fact("proj006", FactCategory::ProjectPattern,
            "Recommendation system: collaborative filtering using matrix factorization or neural network",
            Some("Build user-item matrix then train embedding model"),
            vec!["qai008"]);

        self.add_fact("proj007", FactCategory::ProjectPattern,
            "Object detection: use YOLOv8 pretrained model for detecting objects in images",
            Some("Model::load(\"yolov8\", true)"),
            vec!["qai001"]);

        self.add_fact("proj008", FactCategory::ProjectPattern,
            "Text generation: use GPT-2 or create language model with Transformer blocks",
            Some("Model::load(\"gpt2\", true) for text generation"),
            vec!["qai001"]);
    }

    /// Search knowledge base for relevant facts
    pub fn search(&self, query: &str) -> Vec<&Fact> {
        let words: Vec<String> = query.to_lowercase()
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| w.len() > 2)
            .collect();

        let mut scores: HashMap<usize, usize> = HashMap::new();
        for word in &words {
            if let Some(indices) = self.index.get(word) {
                for &idx in indices {
                    *scores.entry(idx).or_insert(0) += 1;
                }
            }
        }

        let mut sorted: Vec<(usize, usize)> = scores.into_iter().collect();
        sorted.sort_by(|a,b| b.1.cmp(&a.1));

        sorted.iter()
            .take(5)
            .filter_map(|(idx, _)| self.facts.get(*idx))
            .collect()
    }

    /// Look up an error and get the fix
    pub fn fix_error(&self, error: &str) -> Option<String> {
        let err_lower = error.to_lowercase();
        for (pattern, fix) in &self.error_fixes {
            if err_lower.contains(pattern.as_str()) {
                return Some(fix.clone());
            }
        }
        None
    }

    /// Get a code template for a task
    pub fn get_template(&self, task: &str) -> Option<String> {
        let task_lower = task.to_lowercase();
        let key = if task_lower.contains("classif") { "classifier" }
            else if task_lower.contains("regress") { "regression" }
            else if task_lower.contains("anomaly") || task_lower.contains("fraud") { "anomaly_detection" }
            else if task_lower.contains("time series") || task_lower.contains("forecast") { "time_series" }
            else if task_lower.contains("neural") || task_lower.contains("network") { "neural_network" }
            else if task_lower.contains("automl") || task_lower.contains("auto") { "automl" }
            else if task_lower.contains("gan") || task_lower.contains("generat") { "gan" }
            else if task_lower.contains("full") || task_lower.contains("project") { "full_project" }
            else if task_lower.contains("hello") { "hello_world" }
            else { return None };
        self.code_templates.get(key).cloned()
    }

    /// Define a concept in plain English
    pub fn define(&self, concept: &str) -> Option<String> {
        let c = concept.to_lowercase();
        for (key, def) in &self.concept_definitions {
            if c.contains(key.as_str()) || key.contains(c.as_str()) {
                return Some(def.clone());
            }
        }
        None
    }

    /// Learn a new fact from user interaction
    pub fn learn(&mut self, content: &str, example: Option<&str>) {
        let id = format!("learned_{}", self.facts.len());
        self.add_fact(&id, FactCategory::BestPractice, content, example, vec![]);
        println!("  🧠 QuantumMind learned: {}", &content[..content.len().min(60)]);
    }
}
