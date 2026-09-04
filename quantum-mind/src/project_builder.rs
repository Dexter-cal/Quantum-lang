//! Full project builder — generates complete multi-file projects from one prompt

use crate::codegen::{CodeGenerator, GeneratedProject};
use std::fs;
use std::path::Path;

pub struct ProjectBuilder {
    pub generator: CodeGenerator,
}

impl ProjectBuilder {
    pub fn new() -> Self {
        Self { generator: CodeGenerator::new() }
    }

    /// Build a full project on disk from a prompt
    pub fn build_full_project(&self, prompt: &str, output_dir: &str) -> Result<String, String> {
        println!("\n{}", "═".repeat(60));
        println!("  🚀 QuantumMind — Building Full Project");
        println!("  Prompt: {}", &prompt[..prompt.len().min(55)]);
        println!("{}", "═".repeat(60));

        // Generate the project
        let project = self.generator.generate_project(prompt);

        // Create directory structure
        let base = format!("{}/{}", output_dir, &project.name);
        for subdir in &["src", "tests", "data", "models"] {
            fs::create_dir_all(format!("{}/{}", base, subdir))
                .map_err(|e| format!("Could not create dir: {}", e))?;
        }

        // Write all files
        for file in &project.files {
            let path = format!("{}/{}", base, file.path);
            if let Some(parent) = Path::new(&path).parent() {
                let _ = fs::create_dir_all(parent);
            }
            fs::write(&path, &file.content)
                .map_err(|e| format!("Could not write {}: {}", file.path, e))?;
            println!("  ✅ Created: {}", file.path);
        }

        // Print summary
        println!("\n  📋 Project Summary:");
        println!("  ┌─────────────────────────────────────────────────────┐");
        println!("  │ Name:       {:<41} │", project.name);
        println!("  │ Files:      {:<41} │", project.files.len());
        println!("  │ Expected:   {:<41} │", &project.estimated_accuracy[..project.estimated_accuracy.len().min(41)]);
        println!("  └─────────────────────────────────────────────────────┘");

        println!("\n  📝 Build Steps:");
        for step in &project.instructions {
            println!("    {}", step);
        }

        println!("\n  🏃 To run your project:");
        println!("  cd {}", base);
        println!("  cargo run --release");

        Ok(base)
    }

    /// Generate and print a project without writing to disk
    pub fn preview_project(&self, prompt: &str) {
        let project = self.generator.generate_project(prompt);

        println!("\n{}", "═".repeat(62));
        println!("  🚀 PROJECT PREVIEW: {}", project.name.to_uppercase());
        println!("{}", "═".repeat(62));
        println!("  Description: {}", project.description);
        println!("  Expected:    {}", project.estimated_accuracy);

        println!("\n  FILES THAT WILL BE CREATED:");
        for f in &project.files {
            println!("  📄 {} — {}", f.path, f.description);
        }

        println!("\n  MAIN FILE PREVIEW (src/main.rs):");
        println!("  {}", "─".repeat(58));
        if let Some(main) = project.files.first() {
            for line in main.content.lines().take(30) {
                println!("  {}", line);
            }
            if main.content.lines().count() > 30 {
                println!("  ... ({} more lines)", main.content.lines().count() - 30);
            }
        }
        println!("  {}", "─".repeat(58));
    }
}
