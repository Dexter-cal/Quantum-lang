// Quantum Compiler - Main Entry Point v2.0
// Fast, native compiler with ML-aware type system

use anyhow::{Context, Result};
use colored::*;
use std::path::PathBuf;
use std::time::Instant;

mod lexer;
mod parser;
mod ast;
mod typechecker;
mod mir;
mod codegen;
mod error;
mod utils;
mod dictionary;
mod preprocessor;
mod modules;
mod monomorphize;

use crate::lexer::Lexer;
use crate::parser::Parser as QParser;
use crate::typechecker::TypeChecker;
use crate::codegen::CodeGenerator;

/// Inline version of check_source for use within the compiler binary.
fn check_source_inline(source: &str) -> Vec<String> {
    let mut errors = Vec::new();
    let mut lexer = lexer::Lexer::new(source);
    let tokens = match lexer.tokenize() {
        Ok(t) => t,
        Err(e) => { errors.push(e); return errors; }
    };
    let mut parser = parser::Parser::new(tokens);
    let mut parse_errors = Vec::new();
    let ast = parser.parse_recovering(&mut parse_errors);
    errors.extend(parse_errors);
    let mut tc = typechecker::TypeChecker::new();
    errors.extend(tc.check_all(&ast));
    errors
}

fn format_repl_error(msg: &str, _input: &str) -> String {
    // Strip "Error: " prefix if present
    let msg = msg.trim_start_matches("Error: ");
    // Strip file path references like "/tmp/_repl_in.qtm:" that appear in
    // some compiler error messages.
    let msg = if let Some(idx) = msg.find("/tmp/_repl") {
        if let Some(colon) = msg[idx..].find(':') {
            msg[idx + colon + 1..].trim()
        } else {
            msg
        }
    } else {
        msg
    };
    // Clean up "at line N, column M" references that point into the synthetic
    // `fn main() { ... }` wrapper rather than the user's actual input — just
    // keep the core error description.
    msg.to_string()
}

fn print_usage() {
    println!("{}", "Quantum Programming Language v2.0".bright_cyan().bold());
    println!("Usage: quantumc <command> [options]");
    println!();
    println!("Commands:");
    println!("  compile <file>    Compile a .qtm file");
    println!("  run <file>        Compile and run a .qtm file");
    println!("  new <name>        Create a new project");
    println!("  build             Build current project");
    println!("  check <file>      Type-check without compiling");
    println!("  repl              Start interactive REPL");
    println!("  dict <sub>        Manage keyword dictionary (see: dict help)");
    println!("  help              Show this help");
    println!();
    println!("Options:");
    println!("  -o <file>         Output filename");
    println!("  -O<0-3>           Optimization level (default: 2)");
    println!("  --debug           Include debug info");
    println!("  --emit-c          Emit generated C code");
    println!("  --ai              AI project template (with new)");
    println!("  --link <lib>      Link an external C library (e.g. --link curl)");
    println!("  --lib             Library project (with new)");
}

/// Handle `quantumc dict <subcommand> [args]`
fn handle_dict_command(args: &[String]) {
    let sub = args.first().map(|s| s.as_str()).unwrap_or("");
    let mut rest = args.iter().skip(1).map(|s| s.as_str());
    match sub {
        // ── quantumc dict show ───────────────────────────────────────────
        "show" | "" => {
            let path = std::path::Path::new("quantum_lang.toml");
            if !path.exists() {
                println!("No quantum_lang.toml in current directory.");
                println!("Run `quantumc dict init` to create one, or `quantumc dict use <lang>` to apply a language pack.");
                return;
            }
            let dict = dictionary::Dictionary::load_from_dir(std::path::Path::new("."))
                .unwrap_or_default();
            println!("{}", "Active keyword aliases (from quantum_lang.toml):".bright_cyan());
            let entries = dict.entries();
            if entries.is_empty() {
                println!("  (no aliases defined)");
            }
            let mut grouped: std::collections::HashMap<&str, Vec<&str>> =
                std::collections::HashMap::new();
            let entries_ref: Vec<_> = entries.iter()
                .map(|(a, k)| (a.as_str(), k.as_str()))
                .collect();
            for (alias, kw) in &entries_ref {
                grouped.entry(kw).or_default().push(alias);
            }
            let mut kws: Vec<_> = grouped.keys().collect();
            kws.sort();
            for kw in kws {
                let aliases = grouped[kw].join(", ");
                println!("  {:12} → {}", kw, aliases);
            }
        }
        // ── quantumc dict init ───────────────────────────────────────────
        "init" => {
            let path = "quantum_lang.toml";
            if std::path::Path::new(path).exists() {
                println!("quantum_lang.toml already exists. Edit it directly or use `dict use <lang>` to append a language pack.");
                return;
            }
            let template = r#"# Quantum Language Dictionary
# Add aliases for any keyword below.
# Example: fn = ["def", "funcion", "🔧"]

[keywords]
# fn     = ["def"]
# let    = ["var"]
# if     = ["si", "wenn", "もし"]
# return = ["retornar", "retour"]
"#;
            std::fs::write(path, template).expect("failed to write quantum_lang.toml");
            println!("Created quantum_lang.toml — edit it to add your aliases.");
        }
        // ── quantumc dict use <lang> ─────────────────────────────────────
        "use" => {
            let lang = rest.next().unwrap_or("");
            match dictionary::builtin_pack(&lang) {
                None => {
                    println!("Unknown language pack: {:?}", lang);
                    println!("Available packs:");
                    for (name, desc) in dictionary::PACK_LIST {
                        println!("  {:12} — {}", name, desc);
                    }
                }
                Some(pack_toml) => {
                    // Append pack contents (or create new file) in current dir
                    let path = "quantum_lang.toml";
                    let existing = std::fs::read_to_string(path).unwrap_or_default();
                    if existing.contains("[keywords]") {
                        // Merge: just append the aliases as comments + overrides
                        let merged = format!("{}\n# ── {} pack ──\n{}", existing, lang, pack_toml);
                        std::fs::write(path, merged).expect("write failed");
                    } else {
                        std::fs::write(path, pack_toml).expect("write failed");
                    }
                    println!("Applied '{}' language pack to quantum_lang.toml.", lang);
                    println!("Run `quantumc dict show` to see all active aliases.");
                }
            }
        }
        // ── quantumc dict add <keyword> <alias> ─────────────────────────
        "add" => {
            let keyword = rest.next().unwrap_or("").to_string();
            let alias   = rest.next().unwrap_or("").to_string();
            if keyword.is_empty() || alias.is_empty() {
                println!("Usage: quantumc dict add <keyword> <alias>");
                println!("Example: quantumc dict add fn funcion");
                return;
            }
            let path = "quantum_lang.toml";
            let existing = std::fs::read_to_string(path).unwrap_or_default();
            let entry = format!("{} = [\"{}\"]", keyword, alias);
            if existing.is_empty() {
                std::fs::write(path, format!("[keywords]\n{}\n", entry)).ok();
            } else if existing.contains(&format!("{} =", keyword)) {
                println!("'{}' already has aliases. Edit quantum_lang.toml directly to add more.", keyword);
                return;
            } else {
                // Append to [keywords] section or end of file
                let updated = if existing.contains("[keywords]") {
                    format!("{}\n{}\n", existing.trim_end(), entry)
                } else {
                    format!("{}\n[keywords]\n{}\n", existing.trim_end(), entry)
                };
                std::fs::write(path, updated).ok();
            }
            println!("Added: {} → \"{}\"  (in quantum_lang.toml)", keyword, alias);
        }
        // ── quantumc dict packs ──────────────────────────────────────────
        "packs" | "list" => {
            println!("{}", "Built-in language packs:".bright_cyan());
            for (name, desc) in dictionary::PACK_LIST {
                println!("  {:12} — {}", name, desc);
            }
            println!("\nApply one: quantumc dict use <name>");
        }
        // ── quantumc dict reset ──────────────────────────────────────────
        "reset" => {
            if std::path::Path::new("quantum_lang.toml").exists() {
                std::fs::remove_file("quantum_lang.toml").ok();
                println!("Removed quantum_lang.toml — back to default English keywords.");
            } else {
                println!("No quantum_lang.toml to remove.");
            }
        }
        // ── quantumc dict help ───────────────────────────────────────────
        _ => {
            println!("{}", "quantumc dict — keyword dictionary management".bright_cyan());
            println!();
            println!("Subcommands:");
            println!("  show              Show active aliases from quantum_lang.toml");
            println!("  init              Create a starter quantum_lang.toml");
            println!("  use <lang>        Apply a built-in language pack");
            println!("  add <kw> <alias>  Add a single alias");
            println!("  packs             List available built-in language packs");
            println!("  reset             Remove quantum_lang.toml (restore defaults)");
            println!();
            println!("Example:");
            println!("  quantumc dict use spanish");
            println!("  quantumc dict add fn def");
            println!("  quantumc dict show");
        }
    }
}

struct Compiler {
    opt_level: u8,
    debug: bool,
    emit_c: bool,
    quiet: bool,  // suppress progress output (used by REPL)
    extra_link_libs: Vec<String>,
    extra_link_dirs: Vec<String>,
}

impl Compiler {
    fn new(opt_level: u8, debug: bool, emit_c: bool) -> Self {
        Self { opt_level, debug, emit_c, quiet: false, extra_link_libs: Vec::new(), extra_link_dirs: Vec::new() }
    }
    fn new_quiet(opt_level: u8) -> Self {
        Self { opt_level, debug: false, emit_c: false, quiet: true, extra_link_libs: Vec::new(), extra_link_dirs: Vec::new() }
    }

    fn compile_file(&self, input: &PathBuf, output: Option<PathBuf>) -> Result<()> {
        let start = Instant::now();

        if !self.quiet { println!("{}", "📝 Compiling...".bright_blue().bold()); }

        // Stage 1+2: Lexing + Parsing, with module resolution. Any `import`
        // statements in the entry file (or transitively in imported files)
        // are resolved relative to each file's own directory, parsed using
        // that file's own #syntax directive and quantum_lang.toml dictionary
        // (each imported file's syntax style is fully independent), and
        // merged into one combined Program before typechecking.
        if !self.quiet { print!("  {} Lexing & resolving imports...", "[1/6]".bright_black()); }
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let stage_start = Instant::now();

        let read_and_preprocess = |path: &std::path::Path| -> Result<String, String> {
            let raw = std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
            let normalized = if preprocessor::has_indentation_directive(&raw) {
                preprocessor::preprocess(&raw)
            } else {
                raw
            };
            Ok(normalized)
        };

        let ast = modules::resolve_modules(input, read_and_preprocess)
            .map_err(|e| anyhow::anyhow!(e))?;
        // Monomorphize generic functions before typechecking — every
        // generic call site is rewritten to call a concrete, specialized
        // function by this point, so the typechecker never has to know
        // generics existed at all.
        let ast = monomorphize::monomorphize(ast);
        if !self.quiet { println!(" {} ({:.2}ms)", "✓".green(), stage_start.elapsed().as_secs_f64() * 1000.0); }

        // Stage 3: Type Checking
        if !self.quiet { print!("  {} Type checking...", "[3/6]".bright_black()); }
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let stage_start = Instant::now();
        let mut typechecker = TypeChecker::new();
        let typed_ast = typechecker.check(ast).map_err(|e| anyhow::anyhow!(e))?;
        if !self.quiet { println!(" {} ({:.2}ms)", "✓".green(), stage_start.elapsed().as_secs_f64() * 1000.0); }

        // Stage 4: MIR
        if !self.quiet { print!("  {} Generating MIR...", "[4/6]".bright_black()); }
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let stage_start = Instant::now();
        let mir = mir::lower_to_mir(typed_ast).map_err(|e| anyhow::anyhow!(e))?;
        if !self.quiet { println!(" {} ({:.2}ms)", "✓".green(), stage_start.elapsed().as_secs_f64() * 1000.0); }

        // Stage 5: Optimization
        if !self.quiet { print!("  {} Optimizing...", "[5/6]".bright_black()); }
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let stage_start = Instant::now();
        let optimized_mir = mir::optimize(mir, self.opt_level);
        if !self.quiet { println!(" {} ({:.2}ms)", "✓".green(), stage_start.elapsed().as_secs_f64() * 1000.0); }

        // Stage 6: Code Generation
        if !self.quiet { print!("  {} Generating code...", "[6/6]".bright_black()); }
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let stage_start = Instant::now();
        let output_path = output.unwrap_or_else(|| input.with_extension(""));
        let mut codegen = CodeGenerator::new(self.opt_level, self.debug, self.emit_c);

        // Collect `#link "lib"` directives from the entry file and any
        // (transitively) imported files, so `extern "C"` declarations that
        // need an external library get it passed to the linker.
        codegen.link_libs = self.extra_link_libs.clone();
        codegen.link_dirs = self.extra_link_dirs.clone();
        if let Ok(entry_src) = std::fs::read_to_string(input) {
            codegen.link_libs.extend(modules::list_link_directives(&entry_src));
            codegen.link_dirs.extend(modules::list_lib_path_directives(&entry_src));
        }
        for imported in modules::list_imports(&std::fs::read_to_string(input).unwrap_or_default()) {
            let import_path = input.parent().unwrap_or_else(|| std::path::Path::new(".")).join(format!("{}.qtm", imported));
            if let Ok(src) = std::fs::read_to_string(&import_path) {
                codegen.link_libs.extend(modules::list_link_directives(&src));
                codegen.link_dirs.extend(modules::list_lib_path_directives(&src));
            }
        }

        codegen.generate(optimized_mir, &output_path).map_err(|e| anyhow::anyhow!(e))?;
        if !self.quiet { println!(" {} ({:.2}ms)", "✓".green(), stage_start.elapsed().as_secs_f64() * 1000.0); }

        if !self.quiet { println!(
            "\n{} Compiled {} → {} in {:.2}s",
            "✅".green(),
            input.display().to_string().bright_yellow(),
            output_path.display().to_string().bright_cyan(),
            start.elapsed().as_secs_f64()
        ); }

        Ok(())
    }

    fn check_file(&self, input: &PathBuf) -> Result<()> {
        let read_and_preprocess = |path: &std::path::Path| -> Result<String, String> {
            let raw = std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
            let normalized = if preprocessor::has_indentation_directive(&raw) {
                preprocessor::preprocess(&raw)
            } else {
                raw
            };
            Ok(normalized)
        };
        let ast = modules::resolve_modules(input, read_and_preprocess)
            .map_err(|e| anyhow::anyhow!(e))?;
        let ast = monomorphize::monomorphize(ast);
        let mut typechecker = TypeChecker::new();
        typechecker.check(ast).map_err(|e| anyhow::anyhow!(e))?;
        println!("{} Type checking passed for {}", "✅".green(), input.display());
        Ok(())
    }

    fn new_project(&self, name: &str, is_lib: bool, is_ai: bool) -> Result<()> {
        println!("{} Creating project: {}", "📦".bright_blue(), name.bright_cyan());
        let project_dir = PathBuf::from(name);
        std::fs::create_dir_all(&project_dir)?;
        std::fs::create_dir_all(project_dir.join("src"))?;
        std::fs::create_dir_all(project_dir.join("tests"))?;
        std::fs::create_dir_all(project_dir.join("data"))?;

        let config = format!(
            "[package]\nname = \"{}\"\nversion = \"0.1.0\"\nedition = \"2026\"\n\n[dependencies]\n{}\n",
            name,
            if is_ai { "quantumai = \"2.0\"\n# ML deps auto-included" } else { "" }
        );
        std::fs::write(project_dir.join("quantum.toml"), config)?;

        let main_content = if is_ai {
            "// AI project template\nimport quantumai as qai\n\nfn main() {\n    let model = qai.create(\"classifier\")\n    model.fit(\"data.csv\", 20)\n    print(\"AI model ready!\")\n}\n".to_string()
        } else {
            "fn main() {\n    print(\"Hello, Quantum!\")\n}\n".to_string()
        };
        std::fs::write(
            project_dir.join("src").join(if is_lib { "lib.qtm" } else { "main.qtm" }),
            main_content
        )?;

        println!("{} Created {}/ ({})", "✅".green(), name,
            if is_ai { "AI template" } else { "standard" });
        Ok(())
    }

    fn start_repl(&self) {
        use std::io::{self, BufRead, Write};

        println!("{}", "━".repeat(56).bright_cyan());
        println!("{}", "  ⚡ Quantum REPL v2.1".bright_cyan().bold());
        println!("{}", "  Expressions, let bindings, fn definitions.".bright_black());
        println!("{}", "  Commands: :help  :clear  :vars  :history  :exit".bright_black());
        println!("{}", "━".repeat(56).bright_cyan());
        println!();

        let stdin = io::stdin();
        let mut session_defs: Vec<String> = Vec::new();
        let mut session_lets: Vec<String> = Vec::new();
        let mut history: Vec<String> = Vec::new();
        let mut input_buffer = String::new();

        loop {
            let prompt = if input_buffer.is_empty() {
                format!("{} ", "qt›".bright_green().bold())
            } else {
                format!("{} ", "  …".bright_yellow())
            };
            print!("{}", prompt);
            let _ = io::stdout().flush();

            let mut line = String::new();
            match stdin.lock().read_line(&mut line) {
                Ok(0) | Err(_) => break,
                _ => {}
            }
            let trimmed = line.trim();

            match trimmed {
                ":exit" | ":quit" | "exit" | "quit" => {
                    println!("\n{}", "Goodbye! 👋".bright_cyan());
                    break;
                }
                ":clear" => {
                    session_defs.clear();
                    session_lets.clear();
                    println!("{}", "  Session cleared.".bright_black());
                    continue;
                }
                ":vars" => {
                    if session_lets.is_empty() {
                        println!("{}", "  No variables defined yet.".bright_black());
                    } else {
                        println!("{}", "  Session variables:".bright_cyan());
                        for l in &session_lets {
                            println!("    {}", l.trim().bright_white());
                        }
                    }
                    continue;
                }
                ":history" => {
                    for (i, h) in history.iter().enumerate() {
                        println!("  {} {}", format!("[{}]", i+1).bright_black(), h);
                    }
                    continue;
                }
                ":help" => {
                    println!("{}", "  Commands:".bright_cyan());
                    println!("    :vars     — show all defined variables");
                    println!("    :clear    — reset session (vars + fns)");
                    println!("    :history  — show input history");
                    println!("    :exit     — quit");
                    println!();
                    println!("  Tips:");
                    println!("    • let bindings persist across inputs");
                    println!("    • fn definitions persist across inputs");
                    println!("    • multi-line: keep typing until braces balance");
                    continue;
                }
                "" if input_buffer.is_empty() => continue,
                _ => {}
            }

            input_buffer.push_str(trimmed);
            input_buffer.push('\n');

            let opens  = input_buffer.chars().filter(|&c| c == '{').count();
            let closes = input_buffer.chars().filter(|&c| c == '}').count();
            if opens > closes { continue; }

            let input = input_buffer.trim().to_string();
            input_buffer.clear();
            if input.is_empty() { continue; }
            history.push(input.clone());

            let is_fn_def = input.starts_with("fn ")
                || input.starts_with("struct ")
                || input.starts_with("enum ")
                || input.starts_with("pub fn ")
                || input.starts_with("pub struct ");

            if is_fn_def {
                let check_src = format!("{}\n{}", session_defs.join("\n"), input);
                let diags = check_source_inline(&check_src);
                if diags.is_empty() {
                    session_defs.push(input.clone());
                    let name = input.split_whitespace().nth(1)
                        .unwrap_or("").split('(').next().unwrap_or("");
                    println!("{} Defined {}", "  ✓".bright_green(), name.bright_white().bold());
                } else {
                    for d in &diags {
                        println!("{} {}", "  ✗".bright_red(),
                            format_repl_error(d, &input).bright_red());
                    }
                }
                continue;
            }

            let is_let = input.starts_with("let ") || input.starts_with("let mut ");
            let prior = session_lets.join("\n    ");
            let prior_prefix = if prior.is_empty() {
                String::new()
            } else {
                format!("    {}\n", prior)
            };

            let body = if is_let {
                let varname: String = input
                    .trim_start_matches("let mut ")
                    .trim_start_matches("let ")
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                format!("{}    {}\n    println({})\n", prior_prefix, input, varname)
            } else {
                let is_print = input.starts_with("println(") || input.starts_with("print(");
                if is_print {
                    format!("{}    {}\n", prior_prefix, input)
                } else {
                    format!("{}    println({})\n", prior_prefix, input)
                }
            };

            let defs = session_defs.join("\n");
            let src = format!("{}\nfn main() {{\n{}}}\n", defs, body);
            let tmpfile = PathBuf::from("/tmp/_repl_in.qtm");
            let tmp_out = PathBuf::from("/tmp/_repl_out");
            let _ = std::fs::write(&tmpfile, &src);

            let quiet = Compiler::new_quiet(2);
            match quiet.compile_file(&tmpfile, Some(tmp_out.clone())) {
                Ok(()) => {
                    match std::process::Command::new(&tmp_out).output() {
                        Ok(out) => {
                            let stdout = String::from_utf8_lossy(&out.stdout);
                            let stderr = String::from_utf8_lossy(&out.stderr);
                            let result = stdout.trim();
                            if !result.is_empty() {
                                println!("{} {}", "  ⇒".bright_cyan(), result.bright_white().bold());
                            }
                            if !stderr.is_empty() {
                                println!("{} {}", "  ⚠".bright_yellow(), stderr.trim());
                            }
                            if is_let {
                                session_lets.push(format!("    {}", input));
                            }
                        }
                        Err(e) => println!("{} Runtime error: {}", "  ✗".bright_red(), e),
                    }
                }
                Err(e) => {
                    let msg = e.to_string();
                    println!("{} {}", "  ✗".bright_red(),
                        format_repl_error(&msg, &input).bright_red());
                }
            }
            let _ = std::fs::remove_file(&tmp_out);
        }
        println!();
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        return Ok(());
    }

    let mut opt_level: u8 = 2;
    let mut debug = false;
    let mut emit_c = false;
    let mut output: Option<PathBuf> = None;
    let mut is_ai = false;
    let mut is_lib = false;

    // Parse flags
    let positional: Vec<String> = args[1..].iter().filter(|a| {
        if a.starts_with("-O") {
            opt_level = a[2..].parse().unwrap_or(2);
            false
        } else if *a == "--debug" { debug = true; false }
        else if *a == "--emit-c" { emit_c = true; false }
        else if *a == "--ai" { is_ai = true; false }
        else if *a == "--lib" { is_lib = true; false }
        else { true }
    }).cloned().collect();

    // Handle -o and --link flags (both take a following value)
    let mut filtered = Vec::new();
    let mut skip_next = false;
    let mut link_libs: Vec<String> = Vec::new();
    let mut link_dirs: Vec<String> = Vec::new();
    for (i, arg) in positional.iter().enumerate() {
        if skip_next { skip_next = false; continue; }
        if arg == "-o" {
            if let Some(next) = positional.get(i + 1) {
                output = Some(PathBuf::from(next));
                skip_next = true;
            }
        } else if arg == "--link" {
            if let Some(next) = positional.get(i + 1) {
                link_libs.push(next.clone());
                skip_next = true;
            }
        } else if arg == "--lib-path" {
            if let Some(next) = positional.get(i + 1) {
                link_dirs.push(next.clone());
                skip_next = true;
            }
        } else {
            filtered.push(arg.clone());
        }
    }

    let mut compiler = Compiler::new(opt_level, debug, emit_c);
    compiler.extra_link_libs = link_libs;
    compiler.extra_link_dirs = link_dirs;

    match filtered.first().map(|s| s.as_str()) {
        Some("compile") => {
            let file = filtered.get(1).map(PathBuf::from)
                .ok_or_else(|| anyhow::anyhow!("Expected file argument"))?;
            compiler.compile_file(&file, output)?;
        }
        Some("run") => {
            let file = filtered.get(1).map(PathBuf::from)
                .ok_or_else(|| anyhow::anyhow!("Expected file argument"))?;
            let temp = PathBuf::from("/tmp/quantum_run_tmp");
            compiler.compile_file(&file, Some(temp.clone()))?;
            println!("\n{} Running...\n", "🚀".bright_blue());
            let run_args: Vec<String> = filtered[2..].to_vec();
            let status = std::process::Command::new(&temp).args(&run_args).status()?;
            let _ = std::fs::remove_file(&temp);
            if !status.success() { anyhow::bail!("Program exited with error"); }
        }
        Some("new") => {
            let name = filtered.get(1).ok_or_else(|| anyhow::anyhow!("Expected project name"))?;
            compiler.new_project(name, is_lib, is_ai)?;
        }
        Some("build") => {
            let release = filtered.contains(&"--release".to_string());
            let c = Compiler::new(if release { 3 } else { 0 }, !release, emit_c);
            c.compile_file(&PathBuf::from("src/main.qtm"), Some(PathBuf::from("target/app")))?;
        }
        Some("check") => {
            let file = filtered.get(1).map(PathBuf::from)
                .ok_or_else(|| anyhow::anyhow!("Expected file argument"))?;
            compiler.check_file(&file)?;
        }
        Some("repl") => { compiler.start_repl(); }
        Some("dict") => {
            handle_dict_command(&filtered[1..]);
        }
        Some("help") | Some("--help") | Some("-h") => { print_usage(); }
        Some(cmd) => {
            // Try treating as a file to run directly
            let file = PathBuf::from(cmd);
            if file.exists() {
                let temp = PathBuf::from("/tmp/quantum_run_tmp");
                compiler.compile_file(&file, Some(temp.clone()))?;
                println!("\n{} Running...\n", "🚀".bright_blue());
                let _ = std::process::Command::new(&temp).status()?;
                let _ = std::fs::remove_file(&temp);
            } else {
                eprintln!("{} Unknown command: {}", "error:".bright_red(), cmd);
                print_usage();
            }
        }
        None => { print_usage(); }
    }

    Ok(())
}
