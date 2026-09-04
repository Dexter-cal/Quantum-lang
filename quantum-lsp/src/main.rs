//! quantum-lsp — a minimal Language Server Protocol implementation for
//! Quantum (.qtm) files.
//!
//! Implements just enough LSP over stdio to be useful in editors:
//!   - initialize / initialized / shutdown / exit
//!   - textDocument/didOpen, didChange, didSave -> publishDiagnostics
//!     (runs lex -> parse -> typecheck via quantum_compiler::check_source
//!      and reports the first error found, with line/column when available)
//!   - textDocument/hover -> short docs for Quantum/QuantumAI keywords,
//!     looked up from a small built-in table (kept self-contained rather
//!     than depending on quantum-mind to avoid a heavy dependency here).
//!
//! No external LSP crate is used (tower-lsp's dependency tree requires a
//! newer Rust edition than is available in this environment), so this
//! implements the Content-Length-framed JSON-RPC transport directly.

use quantum_compiler::{check_source, Severity};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{self, Write};

mod hover_docs;
mod transport;
mod completions;
mod definitions;

use transport::{read_message, write_message};

fn main() {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let stdout = io::stdout();
    let mut writer = stdout.lock();

    // Open document texts, keyed by file URI.
    let mut documents: HashMap<String, String> = HashMap::new();

    loop {
        let msg = match read_message(&mut reader) {
            Ok(Some(m)) => m,
            Ok(None) => break, // EOF
            Err(e) => {
                eprintln!("quantum-lsp: transport error: {}", e);
                break;
            }
        };

        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let id = msg.get("id").cloned();

        match method {
            "initialize" => {
                let result = json!({
                    "capabilities": {
                        "textDocumentSync": {
                            "openClose": true,
                            "change": 1
                        },
                        "hoverProvider": true,
                        "definitionProvider": true,
                        "completionProvider": {
                            "triggerCharacters": [".", ":", " "],
                            "resolveProvider": false
                        }
                    },
                    "serverInfo": {
                        "name": "quantum-lsp",
                        "version": "0.1.0"
                    }
                });
                respond(&mut writer, id, result);
            }
            "initialized" => {}
            "shutdown" => {
                respond(&mut writer, id, Value::Null);
            }
            "exit" => break,
            "textDocument/didOpen" => {
                if let Some(params) = msg.get("params") {
                    let doc = &params["textDocument"];
                    let uri = doc["uri"].as_str().unwrap_or("").to_string();
                    let text = doc["text"].as_str().unwrap_or("").to_string();
                    documents.insert(uri.clone(), text.clone());
                    publish_diagnostics(&mut writer, &uri, &text);
                }
            }
            "textDocument/didChange" => {
                if let Some(params) = msg.get("params") {
                    let uri = params["textDocument"]["uri"].as_str().unwrap_or("").to_string();
                    if let Some(changes) = params["contentChanges"].as_array() {
                        if let Some(last) = changes.last() {
                            if let Some(text) = last["text"].as_str() {
                                documents.insert(uri.clone(), text.to_string());
                                publish_diagnostics(&mut writer, &uri, text);
                            }
                        }
                    }
                }
            }
            "textDocument/didSave" => {
                if let Some(params) = msg.get("params") {
                    let uri = params["textDocument"]["uri"].as_str().unwrap_or("").to_string();
                    if let Some(text) = documents.get(&uri).cloned() {
                        publish_diagnostics(&mut writer, &uri, &text);
                    }
                }
            }
            "textDocument/didClose" => {
                if let Some(params) = msg.get("params") {
                    let uri = params["textDocument"]["uri"].as_str().unwrap_or("").to_string();
                    documents.remove(&uri);
                    publish_diagnostics_raw(&mut writer, &uri, Vec::new());
                }
            }
            "textDocument/hover" => {
                let result = handle_hover(&msg, &documents);
                respond(&mut writer, id, result);
            }
            "textDocument/definition" => {
                let result = handle_definition(&msg, &documents);
                respond(&mut writer, id, result);
            }
            "textDocument/completion" => {
                let result = handle_completion(&msg, &documents);
                respond(&mut writer, id, result);
            }
            "" => {}
            _ => {
                if id.is_some() {
                    respond(&mut writer, id, Value::Null);
                }
            }
        }
    }
}

fn publish_diagnostics<W: Write>(writer: &mut W, uri: &str, text: &str) {
    let diags = check_source(text);
    let lsp_diags: Vec<Value> = diags.iter().map(|d| {
        let line = d.line.map(|l| l.saturating_sub(1)).unwrap_or(0) as u64;
        let col = d.column.map(|c| c.saturating_sub(1)).unwrap_or(0) as u64;
        let severity = match d.severity {
            Severity::Error => 1,
            Severity::Warning => 2,
        };
        json!({
            "range": {
                "start": { "line": line, "character": col },
                "end": { "line": line, "character": col + 1 }
            },
            "severity": severity,
            "source": "quantumc",
            "message": d.message,
        })
    }).collect();

    publish_diagnostics_raw(writer, uri, lsp_diags);
}

fn publish_diagnostics_raw<W: Write>(writer: &mut W, uri: &str, diagnostics: Vec<Value>) {
    let notification = json!({
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": {
            "uri": uri,
            "diagnostics": diagnostics
        }
    });
    let _ = write_message(writer, &notification);
}

fn respond<W: Write>(writer: &mut W, id: Option<Value>, result: Value) {
    let response = json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    });
    let _ = write_message(writer, &response);
}

fn handle_definition(msg: &Value, documents: &HashMap<String, String>) -> Value {
    let params = match msg.get("params") {
        Some(p) => p,
        None => return Value::Null,
    };
    let uri = params["textDocument"]["uri"].as_str().unwrap_or("");
    let line = params["position"]["line"].as_u64().unwrap_or(0) as usize;
    let character = params["position"]["character"].as_u64().unwrap_or(0) as usize;

    let text = match documents.get(uri) {
        Some(t) => t,
        None => return Value::Null,
    };

    // Find the word under the cursor
    let word = match word_at_position(text, line, character) {
        Some(w) if !w.is_empty() => w,
        _ => return Value::Null,
    };

    // Build the definition index and look up the word
    let index = definitions::build_index(text);
    let def = match definitions::find_definition(&index, &word) {
        Some(d) => d,
        None => return Value::Null,
    };

    // Return an LSP Location: same file, at the definition site.
    json!({
        "uri": uri,
        "range": {
            "start": { "line": def.line, "character": def.column },
            "end":   { "line": def.line, "character": def.column + word.len() }
        }
    })
}

fn handle_completion(msg: &Value, documents: &HashMap<String, String>) -> Value {
    let params = match msg.get("params") {
        Some(p) => p,
        None => return json!({ "isIncomplete": false, "items": [] }),
    };
    let uri = params["textDocument"]["uri"].as_str().unwrap_or("");
    let line = params["position"]["line"].as_u64().unwrap_or(0) as usize;
    let character = params["position"]["character"].as_u64().unwrap_or(0) as usize;

    let text = match documents.get(uri) {
        Some(t) => t.as_str(),
        None => return json!({ "isIncomplete": false, "items": [] }),
    };

    completions::complete(text, line, character)
}

fn handle_hover(msg: &Value, documents: &HashMap<String, String>) -> Value {
    let params = match msg.get("params") {
        Some(p) => p,
        None => return Value::Null,
    };
    let uri = params["textDocument"]["uri"].as_str().unwrap_or("");
    let line = params["position"]["line"].as_u64().unwrap_or(0) as usize;
    let character = params["position"]["character"].as_u64().unwrap_or(0) as usize;

    let text = match documents.get(uri) {
        Some(t) => t,
        None => return Value::Null,
    };

    let word = word_at_position(text, line, character);
    let word = match word {
        Some(w) if !w.is_empty() => w,
        _ => return Value::Null,
    };

    match hover_docs::lookup(&word) {
        Some(doc) => json!({
            "contents": {
                "kind": "markdown",
                "value": doc
            }
        }),
        None => Value::Null,
    }
}

fn word_at_position(text: &str, line: usize, character: usize) -> Option<String> {
    let line_text = text.lines().nth(line)?;
    let chars: Vec<char> = line_text.chars().collect();
    if chars.is_empty() { return None; }
    let pos = character.min(chars.len().saturating_sub(1));

    let is_word_char = |c: char| c.is_alphanumeric() || c == '_';

    if !is_word_char(chars.get(pos).copied().unwrap_or(' ')) {
        if pos == 0 || !is_word_char(chars[pos - 1]) {
            return None;
        }
    }

    let mut start = pos.min(chars.len().saturating_sub(1));
    while start > 0 && is_word_char(chars[start - 1]) { start -= 1; }
    let mut end = pos;
    while end < chars.len() && is_word_char(chars[end]) { end += 1; }
    if start >= end { return None; }

    Some(chars[start..end].iter().collect())
}
