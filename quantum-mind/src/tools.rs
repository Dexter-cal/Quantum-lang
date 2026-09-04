//! Tool System — QuantumMind can create and use tools
//! Tools are Quantum functions that get created, tested, and added to the tool library.

use std::collections::HashMap;
use std::fs;

#[derive(Debug, Clone)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub code: String,
    pub input_types: Vec<String>,
    pub output_type: String,
    pub tested: bool,
    pub success_rate: f64,
    pub times_used: usize,
}

pub struct ToolLibrary {
    pub tools: HashMap<String, Tool>,
    pub tool_dir: String,
}

impl ToolLibrary {
    pub fn new() -> Self {
        let mut lib = Self {
            tools: HashMap::new(),
            tool_dir: "/tmp/quantum_tools".to_string(),
        };
        let _ = fs::create_dir_all(&lib.tool_dir);
        lib.load_builtin_tools();
        lib
    }

    fn load_builtin_tools(&mut self) {
        // Data tools
        self.register(Tool {
            name: "load_csv".to_string(),
            description: "Load a CSV file as a dataset".to_string(),
            code: r#"fn load_csv(path: string) -> Dataset {
    DataPipeline::load(path).normalize().dataset()
}"#.to_string(),
            input_types: vec!["string".to_string()],
            output_type: "Dataset".to_string(),
            tested: true, success_rate: 1.0, times_used: 0,
        });

        self.register(Tool {
            name: "quick_classify".to_string(),
            description: "Train a classifier in one line".to_string(),
            code: r#"fn quick_classify(dataset_name: string, epochs: int) -> Model {
    let ds = DataPipeline::load(dataset_name).normalize().dataset()
    let (train, test) = ds.train_test_split(0.2)
    let mut model = Model::create("classifier")
    model.lr = 0.01
    model.train(train.x, train.y, epochs, true)
    let ev = model.evaluate(test.x, test.y)
    println("Accuracy: " + ev["accuracy"])
    return model
}"#.to_string(),
            input_types: vec!["string".to_string(), "int".to_string()],
            output_type: "Model".to_string(),
            tested: true, success_rate: 0.95, times_used: 0,
        });

        self.register(Tool {
            name: "benchmark_all".to_string(),
            description: "Run full benchmark on a trained model".to_string(),
            code: r#"fn benchmark_all(model: Model, train: Dataset, test: Dataset, class_names: [string]) {
    let report = BenchmarkReport::run(
        model, train.x, train.y, test.x, test.y,
        class_names.len(), class_names
    )
    report.print(class_names)
    AIAssistant::analyze(report)
}"#.to_string(),
            input_types: vec!["Model".to_string(), "Dataset".to_string()],
            output_type: "void".to_string(),
            tested: true, success_rate: 1.0, times_used: 0,
        });

        self.register(Tool {
            name: "auto_tune".to_string(),
            description: "Automatically find best hyperparameters".to_string(),
            code: r#"fn auto_tune(dataset: Dataset, n_trials: int) -> Model {
    let best = tune(dataset, n_trials)
    let mut model = Model::create("classifier")
    model.lr = best.best_lr
    model.opt_name = best.best_optimizer
    return model
}"#.to_string(),
            input_types: vec!["Dataset".to_string(), "int".to_string()],
            output_type: "Model".to_string(),
            tested: true, success_rate: 0.9, times_used: 0,
        });

        self.register(Tool {
            name: "detect_anomalies".to_string(),
            description: "Detect anomalies in data using autoencoder".to_string(),
            code: r#"fn detect_anomalies(normal_data: Dataset, new_sample: Tensor) -> string {
    let mut detector = AnomalyDetector::new(normal_data.n_features)
    detector.train_on_normal(normal_data.x, 100)
    return detector.describe(new_sample)
}"#.to_string(),
            input_types: vec!["Dataset".to_string(), "Tensor".to_string()],
            output_type: "string".to_string(),
            tested: true, success_rate: 0.88, times_used: 0,
        });

        self.register(Tool {
            name: "cross_validate".to_string(),
            description: "Run k-fold cross-validation for reliable accuracy".to_string(),
            code: r#"fn cross_validate(dataset: Dataset, k: int) -> float {
    let cv = CrossValidation::run(dataset, "classifier", 100, k)
    println("CV mean accuracy: " + cv.mean)
    return cv.mean
}"#.to_string(),
            input_types: vec!["Dataset".to_string(), "int".to_string()],
            output_type: "float".to_string(),
            tested: true, success_rate: 1.0, times_used: 0,
        });

        self.register(Tool {
            name: "explain_prediction".to_string(),
            description: "Explain why model made a prediction".to_string(),
            code: r#"fn explain_prediction(model: Model, sample: Tensor, features: [string], classes: [string]) {
    PredictionExplainer::explain(model, sample, features, classes)
}"#.to_string(),
            input_types: vec!["Model".to_string(), "Tensor".to_string()],
            output_type: "void".to_string(),
            tested: true, success_rate: 1.0, times_used: 0,
        });
    }

    pub fn register(&mut self, tool: Tool) {
        let name = tool.name.clone();
        self.tools.insert(name.clone(), tool);
        // Save to disk
        let path = format!("{}/{}.qtm", self.tool_dir, name);
        let _ = fs::write(&path, &self.tools[&name].code);
    }

    /// Create a NEW tool from a description — QuantumMind invents tools it needs
    pub fn create_tool(&mut self, name: &str, description: &str, code: &str) -> &Tool {
        println!("  🔧 QuantumMind creating new tool: {}", name);
        let tool = Tool {
            name: name.to_string(),
            description: description.to_string(),
            code: code.to_string(),
            input_types: vec!["auto".to_string()],
            output_type: "auto".to_string(),
            tested: false,
            success_rate: 0.0,
            times_used: 0,
        };
        self.register(tool);
        println!("  ✅ Tool '{}' created and saved to library", name);
        self.tools.get(name).unwrap()
    }

    /// Find a tool by description/query
    pub fn find_tool(&self, query: &str) -> Option<&Tool> {
        let q = query.to_lowercase();
        self.tools.values().find(|t|
            t.description.to_lowercase().contains(&q) ||
            t.name.to_lowercase().contains(&q)
        )
    }

    /// Print all available tools as a table
    pub fn print_tools(&self) {
        println!("\n  🛠️  AVAILABLE TOOLS ({} total)", self.tools.len());
        println!("  ┌──────────────────────────┬──────────────────────────────────────────┐");
        println!("  │ Tool Name                │ Description                              │");
        println!("  ├──────────────────────────┼──────────────────────────────────────────┤");
        let mut tools: Vec<&Tool> = self.tools.values().collect();
        tools.sort_by(|a,b| a.name.cmp(&b.name));
        for t in &tools {
            println!("  │ {:<24} │ {:<40} │",
                &t.name[..t.name.len().min(24)],
                &t.description[..t.description.len().min(40)]);
        }
        println!("  └──────────────────────────┴──────────────────────────────────────────┘");
    }

    /// Mark a tool as tested and record success
    pub fn mark_tested(&mut self, name: &str, success: bool) {
        if let Some(t) = self.tools.get_mut(name) {
            t.tested = true;
            t.times_used += 1;
            let n = t.times_used as f64;
            t.success_rate = (t.success_rate * (n-1.0) + if success {1.0} else {0.0}) / n;
        }
    }
}
