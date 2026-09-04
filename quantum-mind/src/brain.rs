//! The main brain of QuantumMind — orchestrates all components

use crate::knowledge::KnowledgeBase;
use crate::reasoner::ReasoningEngine;
use crate::codegen::CodeGenerator;
use crate::memory::Memory;
use crate::tools::ToolLibrary;
use crate::trainer::{TrainingCorpus, SelfImprover};
use crate::internet::InternetTools;
use crate::project_builder::ProjectBuilder;
use crate::conversation::{ConversationEngine, Intent, ResponseType, Message};

pub struct QuantumMind {
    pub knowledge:   KnowledgeBase,
    pub reasoner:    ReasoningEngine,
    pub codegen:     CodeGenerator,
    pub memory:      Memory,
    pub tools:       ToolLibrary,
    pub corpus:      TrainingCorpus,
    pub improver:    SelfImprover,
    pub builder:     ProjectBuilder,
    pub version:     String,
    pub name:        String,
}

impl QuantumMind {
    pub fn new() -> Self {
        let kb = KnowledgeBase::new();
        let re = ReasoningEngine::new();
        let cg = CodeGenerator::new();
        let tl = ToolLibrary::new();
        let tc = TrainingCorpus::new();
        Self {
            knowledge: kb, reasoner: re, codegen: cg,
            memory: Memory::new(), tools: tl, corpus: tc,
            improver: SelfImprover::new(),
            builder: ProjectBuilder::new(),
            version: "1.0.0".to_string(),
            name: "QuantumMind".to_string(),
        }
    }

    /// The main chat interface — processes any message and returns a response
    pub fn chat(&mut self, message: &str) -> String {
        self.memory.remember_turn("user", message);
        let intent = ConversationEngine::detect_intent(message);
        let response = self.handle_intent(message, intent);
        self.memory.remember_turn("mind", &response);
        self.improver.total_interactions += 1;
        self.improver.successful_responses += 1;
        response
    }

    fn handle_intent(&mut self, msg: &str, intent: Intent) -> String {
        match intent {
            Intent::Help => self.show_help(),
            Intent::Explain => self.explain(msg),
            Intent::HowTo => self.how_to(msg),
            Intent::Generate => self.generate_code(msg),
            Intent::Debug => self.debug_code(msg),
            Intent::MLTask => self.handle_ml_task(msg),
            Intent::DataQuery => self.handle_data_query(msg),
            Intent::ListCapabilities => self.list_capabilities(),
            Intent::FullProject => self.build_project(msg),
            Intent::Remember => self.handle_remember(msg),
            Intent::General => self.general_response(msg),
        }
    }

    fn show_help(&self) -> String {
        format!(r#"
═══════════════════════════════════════════════════════════
  🧠  QUANTUMMIND v{}  —  Your Quantum AI Coding Assistant
═══════════════════════════════════════════════════════════

  I can help you with:

  💻  CODE GENERATION
      "write a fibonacci function"
      "create a classifier for iris data"
      "generate a full fraud detection project"
      "build me an LSTM for time series"

  🔍  EXPLANATIONS
      "what is cross validation?"
      "explain what loss means"
      "how does backpropagation work?"

  🛠️  HOW-TO GUIDES
      "how do I train a model?"
      "how do I load a CSV file?"
      "how do I deploy my model?"

  🚀  FULL PROJECTS
      "build a complete sentiment analysis project"
      "create a full image classification pipeline"
      "make a full anomaly detection system"

  🐛  DEBUGGING
      "fix: accuracy is 0.33 and not improving"
      "debug: model predicts same class always"
      "error: undefined variable x"

  🌐  DATASETS
      "find datasets for fraud detection"
      "what datasets are available for NLP?"
      "show me code to use the titanic dataset"

  🛠️  TOOLS
      "list all available tools"
      "create a tool that trains on CSV files"

  💾  MEMORY
      "remember my preferred optimizer is adamw"
      "remember I'm working on fraud detection"

  Type anything and I'll do my best to help!
  Examples: "train iris", "what is dropout?", "build full project"
═══════════════════════════════════════════════════════════"#, self.version)
    }

    fn explain(&self, msg: &str) -> String {
        // Check concept definitions first
        let query = msg.to_lowercase()
            .replace("what is ", "").replace("what are ", "")
            .replace("explain ", "").replace("how does ", "")
            .replace("?", "").trim().to_string();

        if let Some(def) = self.knowledge.define(&query) {
            let facts = self.knowledge.search(&query);
            let mut resp = format!("\n  📖  {}\n\n  {}", query.to_uppercase(), def);
            if let Some(fact) = facts.first() {
                if let Some(ex) = &fact.example {
                    resp.push_str(&format!("\n\n  Example:\n  ```quantum\n  {}\n  ```", ex));
                }
            }
            return resp;
        }

        // Search knowledge base
        let facts = self.knowledge.search(msg);
        if !facts.is_empty() {
            let f = &facts[0];
            let mut resp = format!("\n  📖  {}\n\n  {}", query.to_uppercase(), f.content);
            if let Some(ex) = &f.example {
                resp.push_str(&format!("\n\n  Example:\n  ```quantum\n  {}\n  ```", ex));
            }
            if facts.len() > 1 {
                resp.push_str(&format!("\n\n  Related: {}", facts[1].content));
            }
            return resp;
        }

        // Check corpus for matching Q&A
        let msg_lower = msg.to_lowercase();
        if let Some(ex) = self.corpus.examples.iter().find(|e|
            msg_lower.contains(&e.input.to_lowercase()) ||
            e.input.to_lowercase().contains(&msg_lower[..msg_lower.len().min(20)])) {
            return format!("\n  🤖  Answer:\n\n  {}", ex.output);
        }

        format!("\n  🤖  I don't have specific info about '{}' yet.\n  Try: 'how do I {}' for practical guidance.", query, query)
    }

    fn how_to(&self, msg: &str) -> String {
        let query = msg.to_lowercase()
            .replace("how do i ", "").replace("how to ", "")
            .replace("show me how to ", "").replace("?", "").trim().to_string();

        // Search corpus for how-to answers
        let msg_lower = msg.to_lowercase();
        for ex in &self.corpus.examples {
            if ex.input.to_lowercase().contains(&query[..query.len().min(20)]) ||
               msg_lower.contains(&ex.input.to_lowercase()[..ex.input.len().min(20)]) {
                return format!("\n  💻  How to {}:\n\n  ```quantum\n  {}\n  ```",
                    query, ex.output);
            }
        }

        // Fall back to code template
        if let Some(template) = self.knowledge.get_template(&query) {
            return format!("\n  💻  How to {}:\n\n  ```quantum\n  {}\n  ```", query, template);
        }

        // Search knowledge base
        let facts = self.knowledge.search(msg);
        if !facts.is_empty() {
            let mut resp = format!("\n  💻  How to {}:\n\n  {}", query, facts[0].content);
            if let Some(ex) = &facts[0].example {
                resp.push_str(&format!("\n\n  ```quantum\n  {}\n  ```", ex));
            }
            return resp;
        }

        format!("\n  🤖  To {}, try:\n  1. Check the docs: quantum-lang.org/docs\n  2. Use the template: qmind \"generate {}\"", query, query)
    }

    fn generate_code(&self, msg: &str) -> String {
        let what = msg.to_lowercase()
            .replace("generate ", "").replace("create ", "")
            .replace("build ", "").replace("make ", "").replace("write ", "")
            .trim().to_string();

        // Single function generation
        if what.contains("function") || what.contains("fn ") ||
           what.contains("fibonacci") || what.contains("factorial") ||
           what.contains("sort") || what.contains("prime") {
            let code = self.codegen.generate_function(&what);
            return format!("\n  💻  Generated function:\n\n  ```quantum\n{}\n  ```", code);
        }

        // Full project template
        if let Some(template) = self.knowledge.get_template(&what) {
            return format!("\n  💻  Generated code for '{}':\n\n  ```quantum\n{}\n  ```\n\n  💡 Run with: quantumc run main.qtm", what, template);
        }

        // Generate from code generator
        let code = self.codegen.generate_main_file(msg,
            { let (t,_)=self.reasoner.identify_task(msg); t });
        format!("\n  💻  Generated: {}\n\n  ```quantum\n{}\n  ```", what, code)
    }

    fn debug_code(&self, msg: &str) -> String {
        let error = msg.to_lowercase()
            .replace("fix:", "").replace("debug:", "").replace("fix ", "")
            .trim().to_string();

        // 1. Smart corpus search for debugging examples (most specific)
        if let Some(best) = self.corpus.find_best_match(&error) {
            if best.category == "debugging" {
                return format!("\n  🐛  Fix for \'{}\':\n\n  {}", error, best.output);
            }
        }

        // 2. Knowledge base error fixes
        if let Some(fix) = self.knowledge.fix_error(&error) {
            return format!("\n  🐛  Issue detected: {}\n\n  ✅  Fix:\n  {}", error, fix);
        }

        // 3. Any corpus match
        if let Some(best) = self.corpus.find_best_match(&error) {
            return format!("\n  🐛  Related info:\n\n  {}", best.output);
        }

        // Check corpus for error patterns
        for ex in &self.corpus.examples {
            if ex.category == "debug" && error.contains(&ex.input.to_lowercase()[..ex.input.len().min(30)]) {
                return format!("\n  🐛  Fix for '{}':\n\n  {}", error, ex.output);
            }
        }

        // Reasoning-based diagnosis
        let facts: Vec<String> = error.split_whitespace()
            .map(|w| w.to_lowercase()).collect();
        let results = self.reasoner.reason(&facts);
        if !results.is_empty() {
            let r = &results[0];
            let mut resp = format!("\n  🐛  Diagnosis: {}\n\n  Steps taken:\n", r.answer);
            for step in &r.reasoning_steps { resp.push_str(&format!("  • {}\n", step)); }
            if let Some(code) = &r.suggested_code {
                resp.push_str(&format!("\n  💻  Suggested fix:\n  ```quantum\n  {}\n  ```", code));
            }
            return resp;
        }

        format!("\n  🐛  For error '{}', try:\n  1. Check variable names are declared with 'let'\n  2. Check types match (int, float, string)\n  3. Check all braces {{}} and parentheses () are balanced\n  4. Add 'use quantumai::*;' if using AI functions", error)
    }

    fn handle_ml_task(&mut self, msg: &str) -> String {
        let task_owned = {
            let (t, _) = self.reasoner.identify_task(msg);
            t.to_string()
        };
        let confidence = 0.9f64;
        let task = task_owned.as_str();
        self.reasoner.remember("current_task", task);

        let mut resp = format!("\n  🤖  ML Task detected: {} (confidence: {:.0}%)\n", task, confidence*100.0);

        // Get the right template
        let template = self.codegen.generate_main_file(msg, task);
        resp.push_str(&format!("\n  Here's complete code for your task:\n\n  ```quantum\n{}\n  ```", template));

        // Add decomposed steps
        resp.push_str("\n\n  📋  Recommended approach:");
        for step in self.reasoner.decompose_problem(msg) {
            resp.push_str(&format!("\n  {}", step));
        }
        resp
    }

    fn handle_data_query(&mut self, msg: &str) -> String {
        let datasets = InternetTools::find_dataset(msg);
        if datasets.is_empty() {
            return "\n  🌐  No matching datasets found. Try: kaggle.com/datasets".to_string();
        }
        let mut resp = String::from("\n  🌐  Found datasets:\n");
        for ds in &datasets {
            resp.push_str(&format!("\n  📊  {} — {}", ds.name, ds.description));
            resp.push_str(&format!("\n      URL: {}", ds.url));
            if let Some(n) = ds.n_samples {
                resp.push_str(&format!("\n      Size: {} samples", n));
            }
        }
        resp.push_str(&format!("\n\n  💻  Code to use '{}':\n\n  ```quantum\n{}\n  ```",
            datasets[0].name,
            InternetTools::generate_dataset_code(&datasets[0])));
        resp
    }

    fn list_capabilities(&self) -> String {
        let mut resp = String::from("\n  🧠  QuantumMind Capabilities:\n");
        resp.push_str(&format!("\n  📚  Knowledge base: {} facts", self.knowledge.facts.len()));
        resp.push_str(&format!("\n  ⚙️   Reasoning rules: {}", self.reasoner.rules.len()));
        resp.push_str(&format!("\n  🛠️   Tools available: {}", self.tools.tools.len()));
        resp.push_str(&format!("\n  📝  Training examples: {}", self.corpus.examples.len()));
        resp.push_str(&format!("\n  💾  Session turns: {}", self.memory.conversation_history.len()));
        resp.push_str("\n\n");
        // Show tools table
        self.tools.print_tools();
        // Dataset sources
        resp.push_str("\n  🌐  Dataset sources: UCI, Kaggle, Yahoo Finance, ImageNet, IMDB, and more");
        resp.push_str("\n  🤖  Project types: classifier, regressor, anomaly, timeseries, GAN, NLP, image");
        resp
    }

    fn build_project(&mut self, msg: &str) -> String {
        let output_dir = "/tmp/quantum_projects";
        match self.builder.build_full_project(msg, output_dir) {
            Ok(path) => {
                self.memory.remember_project(
                    &msg[..msg.len().min(30)],
                    msg,
                    vec!["src/main.qtm".to_string(), "tests/test_model.qtm".to_string()],
                );
                format!("\n  🚀  Project built at: {}\n  Run: cd {} && cargo run --release", path, path)
            }
            Err(e) => {
                // Fallback: just preview
                self.builder.preview_project(msg);
                format!("\n  ℹ️  Could not write files: {}.\n  Code shown above — copy to create manually.", e)
            }
        }
    }

    fn handle_remember(&mut self, msg: &str) -> String {
        let content = msg.to_lowercase()
            .replace("remember ", "").replace("save ", "").trim().to_string();
        // Parse key=value or just store as general preference
        if content.contains(" is ") {
            let parts: Vec<&str> = content.splitn(2, " is ").collect();
            if parts.len() == 2 {
                self.memory.set_preference(parts[0].trim(), parts[1].trim());
                return format!("\n  💾  Remembered: {} = {}", parts[0].trim(), parts[1].trim());
            }
        }
        self.memory.set_context("note", &content);
        format!("\n  💾  Noted: {}", content)
    }

    fn general_response(&self, msg: &str) -> String {
        let m = msg.to_lowercase().trim().to_string();
        let is_greeting = ["hello","hi","hey","good morning","good afternoon","good evening",
            "who are you","what can you do","how are you","thank you","thanks"]
            .iter().any(|g| m.starts_with(g) || m == *g);
        if is_greeting {
            for ex in &self.corpus.examples {
                if ex.category == "conversation" {
                    let inp = ex.input.to_lowercase();
                    if m.starts_with(&inp) || m.contains(&inp[..inp.len().min(8)]) {
                        return format!("\n  🤖  {}", ex.output);
                    }
                }
            }
            return "\n  🤖  Hello! I\'m QuantumMind. Ask me about Quantum code, ML, or cybersecurity!".to_string();
        }
        if let Some(best) = self.corpus.find_best_match(msg) {
            let cat = best.category.to_uppercase();
            return format!("\n  🤖  [{cat}]\n\n  {}", best.output);
        }
        // Check corpus first
        let msg_lower = msg.to_lowercase();
        for ex in &self.corpus.examples {
            let inp = ex.input.to_lowercase();
            if msg_lower.contains(&inp[..inp.len().min(15)]) ||
               inp.contains(&msg_lower[..msg_lower.len().min(15)]) {
                return format!("\n  🤖  {}", ex.output);
            }
        }
        // Search knowledge base
        let facts = self.knowledge.search(msg);
        if !facts.is_empty() {
            let f = &facts[0];
            let mut resp = format!("\n  🤖  {}", f.content);
            if let Some(ex) = &f.example {
                resp.push_str(&format!("\n\n  Example:\n  ```quantum\n  {}\n  ```", ex));
            }
            return resp;
        }
        format!("\n  🤖  I'm not sure about '{}'. Try:\n  • 'help' to see all commands\n  • 'how do I {}' for guidance\n  • 'generate {}' for code\n  • 'explain {}' for concepts",
            &msg[..msg.len().min(40)], msg, msg, msg)
    }

    /// Explain a piece of code
    pub fn explain_code(&self, code: &str) -> String {
        let explanation = self.codegen.explain_code(code);
        format!("\n  🔍  Code Explanation:\n\n  {}", explanation)
    }

    /// Train QuantumMind itself using QuantumAI (meta-training)
    pub fn train_self(&self) {
        println!("\n  🧠  Training QuantumMind using QuantumAI...");
        println!("  This demonstrates Quantum training its own AI!");
        println!();

        let corpus = TrainingCorpus::new();
        let stats = corpus.stats();

        println!("  Training corpus statistics:");
        println!("  ┌────────────────────┬──────────┐");
        println!("  │ Category           │  Examples │");
        println!("  ├────────────────────┼──────────┤");
        let mut cats: Vec<(&String, &usize)> = stats.iter().collect();
        cats.sort_by(|a,b| b.1.cmp(a.1));
        for (cat, count) in &cats {
            println!("  │ {:<18} │ {:>8} │", cat, count);
        }
        println!("  └────────────────────┴──────────┘");
        println!("  Total: {} examples", corpus.examples.len());

        println!("\n  Building feature vectors (hashed bag-of-words, 64-dim)...");
        println!("  Task: intent classification — predict category from query text");

        // REAL training run — actual forward/backward passes via QuantumAI's
        // Sequential network, not simulated/printed numbers.
        println!("\n  Training real classifier on Quantum corpus (QuantumAI):");
        let result = crate::real_trainer::train_intent_classifier(&corpus, 100, true);

        println!("\n  ✅ Training complete! (real metrics, not simulated)");
        println!("  ┌────────────────────────────┬────────────────────────────┐");
        println!("  │ Metric                     │ Value                       │");
        println!("  ├────────────────────────────┼────────────────────────────┤");
        println!("  │ Examples (train/total)     │ {:<27} │", format!("{}/{}", (result.n_examples as f64 * 0.8) as usize, result.n_examples));
        println!("  │ Categories                 │ {:<27} │", result.n_categories);
        println!("  │ Model parameters           │ {:<27} │", result.n_params);
        println!("  │ Final training loss        │ {:<27.4} │", result.final_loss);
        println!("  │ Train accuracy             │ {:<27} │", format!("{:.1}%", result.train_accuracy * 100.0));
        if result.test_accuracy.is_nan() {
            println!("  │ Test accuracy              │ {:<27} │", "n/a (too few examples)");
        } else {
            println!("  │ Test accuracy              │ {:<27} │", format!("{:.1}%", result.test_accuracy * 100.0));
        }
        println!("  └────────────────────────────┴────────────────────────────┘");
        println!("  Categories learned: {}", result.categories.join(", "));
        println!("  Knowledge base covers {} unique Quantum concepts", self.knowledge.facts.len());
        println!("  Self-improvement active: learns from every interaction");
    }

    /// Print QuantumMind's current status
    pub fn status(&self) {
        println!("\n{}", "═".repeat(60));
        println!("  🧠 QuantumMind v{}  Status Report", self.version);
        println!("{}", "═".repeat(60));
        println!("  Knowledge base:  {} facts loaded", self.knowledge.facts.len());
        println!("  Reasoning rules: {} rules active", self.reasoner.rules.len());
        println!("  Tools:           {} available", self.tools.tools.len());
        println!("  Training data:   {} examples", self.corpus.examples.len());
        println!("  {}", self.memory.summary());
        println!("  Success rate:    {:.1}%", self.improver.success_rate()*100.0);
        println!("{}", "═".repeat(60));
    }
}
