//! Go-to-definition support for the Quantum LSP.
//!
//! We build a lightweight symbol index by scanning the source text for
//! definition sites — `fn`, `let [mut]`, `struct`, `enum`, `trait`, `impl`,
//! `const` — without doing a full parse (so this works even when the file
//! has parse errors). Each definition records the name and the 0-indexed
//! line/column of the name token itself (not the keyword), which is what the
//! LSP uses for the target range.
//!
//! Lookup is O(n) in the number of definitions (files are small enough
//! that this is fine without a HashMap).

/// A single definition site found in the source.
#[derive(Debug, Clone)]
pub struct Definition {
    pub name: String,
    /// 0-indexed line (as required by LSP).
    pub line: usize,
    /// 0-indexed column of the first character of the name.
    pub column: usize,
}

/// Build a symbol index from `source` by scanning for definition keywords.
pub fn build_index(source: &str) -> Vec<Definition> {
    let mut defs = Vec::new();

    for (line_idx, line_text) in source.lines().enumerate() {
        let trimmed = line_text.trim_start();
        let indent = line_text.len() - trimmed.len();

        // Patterns that introduce a named definition.
        // We strip one or more prefix words and extract the next identifier.
        let candidates: &[&[&str]] = &[
            // function definition keywords
            &["fn", "def", "function", "func", "funcion", "función",
              "fonction", "funktion", "关数"],
            // variable / let
            &["let mut", "let mutable", "let", "var", "variable"],
            // constant
            &["const", "final", "readonly"],
            // struct / class
            &["pub struct", "struct", "pub class", "class", "pub record", "record"],
            // enum
            &["pub enum", "enum"],
            // trait / interface
            &["pub trait", "trait", "pub interface", "interface"],
            // impl
            &["impl"],
        ];

        'outer: for group in candidates {
            for &kw in *group {
                // Try both "keyword name" (space-separated) and
                // "keyword(name" (for future syntax) forms.
                if let Some(rest) = strip_keyword(trimmed, kw) {
                    let rest = rest.trim_start();
                    // Skip generic params or angle brackets before the name
                    let rest = rest.trim_start_matches('<');
                    // Extract the identifier
                    let name: String = rest
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    if name.is_empty() { continue; }

                    // Column = indent + keyword_len + spaces before name
                    let kw_end = indent + kw.len();
                    let after_kw = &line_text[kw_end..];
                    let spaces = after_kw.len() - after_kw.trim_start().len();
                    let col = kw_end + spaces;

                    defs.push(Definition {
                        name,
                        line: line_idx,
                        column: col,
                    });
                    break 'outer;
                }
            }
        }
    }

    defs
}

/// Strip a keyword prefix (case-insensitive, with required trailing
/// space/paren/brace/colon) from `text`. Returns the remainder if matched.
fn strip_keyword<'a>(text: &'a str, kw: &str) -> Option<&'a str> {
    // Check length to avoid out-of-bounds
    if text.len() < kw.len() { return None; }
    if !text[..kw.len()].eq_ignore_ascii_case(kw) { return None; }
    let rest = &text[kw.len()..];
    // Must be followed by whitespace, `(`, `{`, or end-of-line
    match rest.chars().next() {
        None | Some(' ') | Some('\t') | Some('(') | Some('{') | Some('<') => Some(rest),
        _ => None,
    }
}

/// Find the definition of `name` in the index built from `source`.
/// Returns the first match (usually the only one, since Quantum doesn't
/// support overloading yet).
pub fn find_definition<'a>(
    index: &'a [Definition],
    name: &str,
) -> Option<&'a Definition> {
    index.iter().find(|d| d.name == name)
}
