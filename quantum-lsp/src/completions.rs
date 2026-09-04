//! Completion engine for `textDocument/completion`.
//!
//! Three layers of suggestions:
//!
//! 1. **Keywords** — `fn`, `let`, `for`, `if`, `return`, etc.
//! 2. **Top-level names** — QuantumAI types, stdlib functions, and names
//!    already defined in the open document (extracted by a lightweight
//!    scan of the source text, no full parse required).
//! 3. **Method completions** — triggered when the character before the
//!    cursor is `.`; suggestions depend on the receiver expression.
//!    Currently keyed on known type names in the text before the dot
//!    (e.g. `Sequential` → `.dense`, `.dropout`, `.compile`, `.fit`, …).

use serde_json::{json, Value};

/// LSP CompletionItemKind constants
const KIND_KEYWORD: u32 = 14;
const KIND_CLASS: u32 = 7;
const KIND_METHOD: u32 = 2;
const KIND_FUNCTION: u32 = 3;
const KIND_VARIABLE: u32 = 6;
const KIND_SNIPPET: u32 = 15;

/// Quantum language keywords
static KEYWORDS: &[(&str, &str)] = &[
    ("fn",       "fn ${1:name}(${2:params}) {\n\t$0\n}"),
    ("let",      "let ${1:name} = $0"),
    ("let mut",  "let mut ${1:name} = $0"),
    ("if",       "if ${1:condition} {\n\t$0\n}"),
    ("if else",  "if ${1:condition} {\n\t$2\n} else {\n\t$0\n}"),
    ("for",      "for ${1:i} in ${2:0..10} {\n\t$0\n}"),
    ("while",    "while ${1:condition} {\n\t$0\n}"),
    ("return",   "return $0"),
    ("match",    "match ${1:value} {\n\t${2:_} => $0\n}"),
    ("struct",   "struct ${1:Name} {\n\t${2:field}: ${3:int},\n}"),
    ("enum",     "enum ${1:Name} {\n\t$0\n}"),
    ("impl",     "impl ${1:Type} {\n\t$0\n}"),
    ("import",   "import $0"),
    ("println",  "println($0)"),
    ("print",    "print($0)"),
];

/// Top-level QuantumAI types and functions
static TOP_LEVEL: &[(&str, u32, &str)] = &[
    // DataPipeline
    ("DataPipeline",    KIND_CLASS,    "Data loading and preprocessing pipeline"),
    // Model
    ("Model",           KIND_CLASS,    "High-level ML model wrapper"),
    // Sequential
    ("Sequential",      KIND_CLASS,    "Layer-by-layer neural network builder"),
    // Tensor
    ("Tensor",          KIND_CLASS,    "Multi-dimensional array — core ML data type"),
    // Training utilities
    ("BenchmarkReport", KIND_CLASS,    "Runs accuracy, speed, and confusion matrix benchmarks"),
    ("AIAssistant",     KIND_CLASS,    "Plain-English model advice from a BenchmarkReport"),
    ("ModelCard",       KIND_CLASS,    "One-page model summary (strengths, weaknesses, hardware)"),
    ("CrossValidation", KIND_CLASS,    "k-fold cross-validation for reliable accuracy estimates"),
    ("AnomalyDetector", KIND_CLASS,    "Autoencoder-based anomaly / outlier detector"),
    ("AutoMLPipeline",  KIND_CLASS,    "End-to-end AutoML: loads, tunes, trains, benchmarks"),
    ("PredictionExplainer", KIND_CLASS, "Explains individual predictions"),
    // Standalone functions
    ("feature_importance", KIND_FUNCTION, "Ranks features by permutation importance"),
    ("train_test_split",   KIND_FUNCTION, "Splits a dataset into train and test sets"),
];

/// Method completions keyed by receiver type hint.
/// The receiver hint is the nearest capitalized identifier on the left of `.`.
static METHODS: &[(&str, &[(&str, &str, &str)])] = &[
    ("DataPipeline", &[
        ("load",      "load(\"${1:path}\")",           "Load a dataset from a file or built-in name"),
        ("normalize", "normalize()",                   "Standardise features to mean=0, std=1"),
        ("shuffle",   "shuffle()",                     "Randomly shuffle the dataset"),
        ("batch",     "batch(${1:32})",                "Create mini-batches of given size"),
        ("split",     "split(${1:0.8})",               "Split into train/test fractions"),
        ("dataset",   "dataset()",                     "Materialise the pipeline into a Dataset struct"),
        ("filter",    "filter(|x| $0)",                "Filter samples by a predicate"),
        ("augment",   "augment()",                     "Apply data augmentation"),
    ]),
    ("Sequential", &[
        ("dense",    "dense(${1:in_features}, ${2:out_features}, \"${3:relu}\")",  "Add a fully-connected layer"),
        ("dropout",  "dropout(${1:0.1})",              "Add a dropout layer (rate 0.0–1.0)"),
        ("compile",  "compile(\"${1:adam}\", \"${2:cross_entropy}\", ${3:0.01})", "Set optimizer, loss, and learning rate"),
        ("fit",      "fit(&${1:x_train}, &${2:y_train}, ${3:epochs}, ${4:32}, ${5:true})", "Train the network"),
        ("predict",  "predict(&${1:x})",               "Run forward pass, returns output Tensor"),
        ("predict_class", "predict_class(&${1:x})",    "Returns predicted class index (argmax)"),
        ("compute_accuracy", "compute_accuracy(&${1:x}, &${2:y})", "Accuracy on a labelled dataset"),
        ("save",     "save(\"${1:model.json}\")",      "Save weights to JSON"),
    ]),
    ("Model", &[
        ("train",    "train(&${1:x}, &${2:y}, ${3:100}, ${4:true})",  "Train the model"),
        ("predict",  "predict(&${1:x})",               "Run inference"),
        ("evaluate", "evaluate(&${1:x}, &${2:y})",     "Returns {accuracy, loss} HashMap"),
        ("save",     "save(\"${1:model.json}\")",      "Save weights"),
    ]),
    ("AnomalyDetector", &[
        ("train_on_normal", "train_on_normal(&${1:x}, ${2:100})", "Train on normal (non-anomalous) data"),
        ("score",    "score(&${1:sample})",            "Returns anomaly score 0–100"),
        ("describe", "describe(&${1:sample})",         "Human-readable anomaly description"),
    ]),
    ("Tensor", &[
        ("new",      "new(vec![$1], vec![$2])",        "Tensor::new(data, shape)"),
        ("zeros",    "zeros(vec![$1])",                "Tensor filled with zeros"),
        ("ones",     "ones(vec![$1])",                 "Tensor filled with ones"),
        ("shape",    "shape",                          "Shape (Vec<usize>)"),
        ("data",     "data",                           "Raw data (Vec<f64>)"),
        ("argmax",   "argmax()",                       "Index of maximum element"),
        ("mean",     "mean()",                         "Element-wise mean"),
        ("std",      "std()",                          "Standard deviation"),
    ]),
    // Generic fallback — shown for any `.` trigger we can't identify
    ("_any_", &[
        ("len",        "len()",         "Number of elements"),
        ("to_string",  "to_string()",   "Convert to String"),
        ("clone",      "clone()",       "Clone the value"),
    ]),
];

/// Given the source text and cursor position, produce LSP completion items.
pub fn complete(source: &str, line: usize, character: usize) -> Value {
    let lines: Vec<&str> = source.lines().collect();
    let current_line = lines.get(line).copied().unwrap_or("");

    // Is this a method completion triggered by `.`?
    let char_before = if character > 0 {
        current_line.chars().nth(character - 1)
    } else {
        None
    };

    if char_before == Some('.') {
        let before_dot = &current_line[..character - 1];
        let receiver = last_identifier(before_dot);
        return method_completions(receiver, source);
    }

    // Otherwise: keyword + top-level completions, filtered by the prefix
    // the user has already typed.
    let prefix = current_word_prefix(current_line, character);
    let doc_names = extract_document_names(source);

    let mut items: Vec<Value> = Vec::new();

    // Keywords (as snippets)
    for (kw, snippet) in KEYWORDS {
        if kw.starts_with(&prefix) || prefix.is_empty() {
            items.push(json!({
                "label": kw,
                "kind": KIND_KEYWORD,
                "insertText": snippet,
                "insertTextFormat": 2,  // Snippet
                "detail": "keyword",
            }));
        }
    }

    // Top-level QuantumAI names
    for (name, kind, detail) in TOP_LEVEL {
        if name.to_lowercase().starts_with(&prefix.to_lowercase()) || prefix.is_empty() {
            items.push(json!({
                "label": name,
                "kind": kind,
                "detail": detail,
                "insertText": name,
            }));
        }
    }

    // Names extracted from the current document (user-defined fns/vars)
    for name in &doc_names {
        if name.to_lowercase().starts_with(&prefix.to_lowercase()) || prefix.is_empty() {
            // Don't duplicate names already in the static lists
            let already_listed = TOP_LEVEL.iter().any(|(n, _, _)| n == name)
                || KEYWORDS.iter().any(|(k, _)| k == name);
            if !already_listed {
                items.push(json!({
                    "label": name,
                    "kind": KIND_VARIABLE,
                    "detail": "defined in document",
                    "insertText": name,
                }));
            }
        }
    }

    // Deduplicate labels (stable: first occurrence wins)
    let mut seen = std::collections::HashSet::new();
    items.retain(|item| {
        let label = item["label"].as_str().unwrap_or("").to_string();
        seen.insert(label)
    });

    json!({ "isIncomplete": false, "items": items })
}

/// Produce method completions for a given receiver variable name,
/// looking up what type it was assigned to in the source.
fn method_completions(receiver: &str, source: &str) -> Value {
    // First try: receiver is itself a known type name (e.g. `Sequential.`)
    let direct = METHODS.iter()
        .find(|(ty, _)| ty.to_lowercase() == receiver.to_lowercase())
        .map(|(_, ms)| *ms);

    let methods = if let Some(ms) = direct {
        ms
    } else {
        // Second try: look for `let <receiver> = <TypeName>` in the source.
        // Use prefix matching so partial names while typing (e.g. "Seq" for
        // "Sequential") still resolve correctly.
        let assigned_type = infer_variable_type(receiver, source);
        METHODS.iter()
            .find(|(ty, _)| {
                if *ty == "_any_" { return false; }
                let ty_lc = ty.to_lowercase();
                let at_lc = assigned_type.to_lowercase();
                // Full match or either side is a prefix of the other
                ty_lc == at_lc
                    || ty_lc.starts_with(&at_lc)
                    || at_lc.starts_with(&ty_lc)
            })
            .or_else(|| METHODS.iter().find(|(ty, _)| *ty == "_any_"))
            .map(|(_, ms)| *ms)
            .unwrap_or(&[])
    };

    let items: Vec<Value> = methods.iter().map(|(label, snippet, detail)| {
        json!({
            "label": label,
            "kind": KIND_METHOD,
            "detail": detail,
            "insertText": snippet,
            "insertTextFormat": 2,
        })
    }).collect();

    json!({ "isIncomplete": false, "items": items })
}

/// Scan `source` for the pattern `let <varname> = <Expr>` or
/// `let mut <varname> = <Expr>` and return the first token of the RHS
/// (likely the type name, e.g. "Sequential" from `let net = Sequential::new()`).
fn infer_variable_type(varname: &str, source: &str) -> String {
    for line in source.lines() {
        let t = line.trim();
        // Match `let [mut] <varname> = <rhs>`
        for prefix in ["let mut ", "let "] {
            if let Some(rest) = t.strip_prefix(prefix) {
                // rest = "<varname> = <rhs>" or "<varname>: <type> = <rhs>"
                let name_end = rest.find(|c: char| !c.is_alphanumeric() && c != '_')
                    .unwrap_or(rest.len());
                let name = &rest[..name_end];
                if name == varname {
                    // Find the RHS — after the last `=`
                    if let Some(eq_pos) = rest.rfind('=') {
                        let rhs = rest[eq_pos + 1..].trim();
                        // Return the leading identifier from the RHS
                        let type_name: String = rhs.chars()
                            .take_while(|c| c.is_alphanumeric() || *c == '_')
                            .collect();
                        if !type_name.is_empty() {
                            return type_name;
                        }
                    }
                }
            }
        }
    }
    String::new()
}

/// Extract the partial word the user is currently typing before `character`.
fn current_word_prefix(line: &str, character: usize) -> String {
    let up_to = &line[..character.min(line.len())];
    let start = up_to.rfind(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|i| i + 1)
        .unwrap_or(0);
    up_to[start..].to_string()
}

/// Find the last identifier (word) at the end of `text`.
fn last_identifier(text: &str) -> &str {
    let trimmed = text.trim_end();
    let start = trimmed.rfind(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|i| i + 1)
        .unwrap_or(0);
    &trimmed[start..]
}

/// Very lightweight scan of the source to find `fn`, `let`, and `struct`
/// names defined by the user — so local names show up in completion too.
fn extract_document_names(source: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in source.lines() {
        let t = line.trim();
        for prefix in ["fn ", "let ", "let mut ", "struct ", "enum "] {
            if let Some(rest) = t.strip_prefix(prefix) {
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() {
                    names.push(name);
                }
            }
        }
    }
    names
}
