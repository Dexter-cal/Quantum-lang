//! Training system — loads dataset from JSON, trains QuantumMind using QuantumAI

use std::collections::HashMap;
use std::fs;

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct TrainingExample {
    pub input: String,
    pub output: String,
    pub category: String,
}

pub struct TrainingCorpus {
    pub examples: Vec<TrainingExample>,
}

impl TrainingCorpus {
    /// Load from JSON dataset file if it exists, else use built-in examples
    pub fn new() -> Self {
        let candidates = [
            "quantum-mind/knowledge/training_data.json",
            "/home/claude/project/quantum-lang/quantum-mind/knowledge/training_data.json",
            "knowledge/training_data.json",
        ];
        for path in &candidates {
            if let Ok(json) = fs::read_to_string(path) {
                if let Ok(examples) = serde_json::from_str::<Vec<TrainingExample>>(&json) {
                    return Self { examples };
                }
            }
        }
        let mut corpus = Self { examples: Vec::new() };
        corpus.load_builtin();
        corpus
    }

    fn add(&mut self, input: &str, output: &str, cat: &str) {
        self.examples.push(TrainingExample {
            input: input.to_string(), output: output.to_string(), category: cat.to_string(),
        });
    }

    fn load_builtin(&mut self) {
        self.add("how do I train a classifier",
            "let mut model = Model::create(\"classifier\");\nmodel.train(&data.x, &data.y, 100, true);", "quantumai");
        self.add("what is machine learning",
            "Teaching computers to learn from data without explicit rules.", "concept");
        self.add("how do I fix overfitting",
            "Add Dropout(0.3) between layers, get more training data, reduce model size.", "debugging");
        self.add("what is sql injection",
            "Inserting malicious SQL into queries. Prevention: use parameterized queries.", "cybersecurity");
        self.add("hello", "Hello! I'm QuantumMind. How can I help?", "conversation");
    }

    pub fn stats(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for ex in &self.examples { *counts.entry(ex.category.clone()).or_insert(0) += 1; }
        counts
    }

    /// Smart search: find best matching example by word overlap
    pub fn find_best_match(&self, query: &str) -> Option<&TrainingExample> {
        let q = query.to_lowercase();
        let q_words: Vec<&str> = q.split_whitespace().filter(|w| w.len() > 2).collect();
        if q_words.is_empty() { return None; }

        let mut best_score = 0i32;
        let mut best_idx = None;

        for (idx, ex) in self.examples.iter().enumerate() {
            let inp = ex.input.to_lowercase();
            let inp_words: Vec<&str> = inp.split_whitespace().collect();

            let mut score = 0i32;
            for w in &q_words {
                if inp_words.iter().any(|iw| iw == w) { score += 2; }
                else if inp_words.iter().any(|iw| iw.contains(w) || w.contains(iw)) { score += 1; }
            }
            // Bonus for substring match
            let prefix_len = q.len().min(inp.len()).min(20);
            if prefix_len > 5 && (inp.contains(&q[..prefix_len.min(q.len())]) || q.contains(&inp[..prefix_len.min(inp.len())])) {
                score += 3;
            }

            if score > best_score {
                best_score = score;
                best_idx = Some(idx);
            }
        }

        if best_score >= 2 { best_idx.and_then(|i| self.examples.get(i)) } else { None }
    }
}

pub struct SelfImprover {
    pub improvement_log: Vec<String>,
    pub total_interactions: usize,
    pub successful_responses: usize,
}

impl SelfImprover {
    pub fn new() -> Self {
        Self { improvement_log: Vec::new(), total_interactions: 0, successful_responses: 0 }
    }
    pub fn record_success(&mut self, query: &str) {
        self.total_interactions += 1; self.successful_responses += 1;
        self.improvement_log.push(format!("✅ {}", &query[..query.len().min(50)]));
    }
    pub fn record_failure(&mut self, query: &str) {
        self.total_interactions += 1;
        self.improvement_log.push(format!("❌ {}", &query[..query.len().min(50)]));
    }
    pub fn success_rate(&self) -> f64 {
        if self.total_interactions == 0 { return 1.0; }
        self.successful_responses as f64 / self.total_interactions as f64
    }
    pub fn print_stats(&self) {
        println!("  Interactions: {}  |  Success: {:.1}%", self.total_interactions, self.success_rate()*100.0);
    }
}
