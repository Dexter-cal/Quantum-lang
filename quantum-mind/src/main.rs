//! QuantumMind CLI — quantum mind "your question"
//! The AI coding assistant for the Quantum Programming Language

use quantum_mind::brain::QuantumMind;
use std::env;
use std::io::{self, Write, BufRead};

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse command
    let command = args.get(1).map(|s| s.as_str()).unwrap_or("chat");
    let rest: String = args[2..].join(" ");

    match command {
        "chat" | "ask" | "c" => {
            if rest.is_empty() {
                // Interactive REPL mode
                run_interactive();
            } else {
                let mut mind = QuantumMind::new();
                println!("{}", mind.chat(&rest));
            }
        }
        "generate" | "gen" | "g" => {
            let mut mind = QuantumMind::new();
            println!("{}", mind.chat(&format!("generate {}", rest)));
        }
        "build" | "project" | "b" => {
            let mut mind = QuantumMind::new();
            println!("{}", mind.chat(&format!("build full project {}", rest)));
        }
        "explain" | "e" => {
            let mind = QuantumMind::new();
            if rest.starts_with("code:") {
                println!("{}", mind.explain_code(&rest[5..]));
            } else {
                let mut m = mind;
                println!("{}", m.chat(&format!("explain {}", rest)));
            }
        }
        "debug" | "fix" | "d" => {
            let mut mind = QuantumMind::new();
            println!("{}", mind.chat(&format!("fix {}", rest)));
        }
        "dataset" | "data" => {
            let mut mind = QuantumMind::new();
            println!("{}", mind.chat(&format!("find dataset {}", rest)));
        }
        "tools" | "t" => {
            let mind = QuantumMind::new();
            mind.tools.print_tools();
        }
        "train" => {
            let mind = QuantumMind::new();
            mind.train_self();
        }
        "status" | "s" => {
            let mind = QuantumMind::new();
            mind.status();
        }
        "help" | "h" | "--help" => {
            print_help();
        }
        _ => {
            // Treat the whole thing as a question
            let full = args[1..].join(" ");
            let mut mind = QuantumMind::new();
            println!("{}", mind.chat(&full));
        }
    }
}

fn run_interactive() {
    let mut mind = QuantumMind::new();
    println!("\n  Interactive mode. Type 'exit' to quit, 'status' for stats.\n");
    let stdin = io::stdin();
    loop {
        print!("  You: ");
        io::stdout().flush().unwrap();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let input = line.trim().to_string();
        if input.is_empty() { continue; }
        if input == "exit" || input == "quit" || input == "q" { break; }
        if input == "status" { mind.status(); continue; }
        if input == "train" { mind.train_self(); continue; }
        if input == "tools" { mind.tools.print_tools(); continue; }
        println!("{}", mind.chat(&input));
        println!();
    }
    println!("\n  Goodbye! {}", mind.memory.summary());
}

fn print_help() {
    println!(r#"
  ╔══════════════════════════════════════════════════════════════╗
  ║  🧠  QuantumMind v1.0  —  Quantum Language AI Assistant     ║
  ╚══════════════════════════════════════════════════════════════╝

  USAGE:
    qmind <command> [args]

  COMMANDS:
    chat / ask        Interactive chat or one-shot question
    generate / gen    Generate Quantum code
    build / project   Build a complete project
    explain / e       Explain code or concepts
    debug / fix       Debug errors and fix code
    dataset / data    Find datasets for your task
    tools / t         List all available tools
    train             Train QuantumMind using QuantumAI
    status / s        Show QuantumMind status
    help / h          Show this help

  EXAMPLES:
    qmind chat
    qmind "how do I train a classifier?"
    qmind generate "fibonacci function"
    qmind build "fraud detection system with anomaly detection"
    qmind explain "what is overfitting?"
    qmind fix "accuracy is 0.33 and not improving"
    qmind dataset "credit card fraud"
    qmind train
    qmind status
"#);
}
