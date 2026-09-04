//! Indentation preprocessor — converts Python-style indented blocks into
//! brace-delimited blocks that the Quantum parser understands.
//!
//! This runs only when the source file uses the `#syntax indentation`
//! directive OR when a `quantum_lang.toml` with `[syntax] indentation = true`
//! is present. Without the directive, the source is passed through unchanged,
//! so existing brace-style code is unaffected.
//!
//! ## How it works
//!
//! The preprocessor scans lines and maintains an indentation stack. When the
//! indentation increases after a line that ends with `:` (Python style), it
//! inserts a `{`. When the indentation decreases back, it inserts the matching
//! `}`. Lines ending with `:` have the `:` stripped.
//!
//! ## Example
//!
//! Input:
//! ```text
//! fn greet(name: string):
//!     let msg = "Hello " + name
//!     if msg.len() > 5:
//!         println(msg)
//! ```
//!
//! Output:
//! ```text
//! fn greet(name: string) {
//!     let msg = "Hello " + name
//!     if msg.len() > 5 {
//!         println(msg)
//!     }
//! }
//! ```

/// Returns true if `source` declares indentation mode via the first-line
/// directive `#syntax indentation` or `#mode python`.
pub fn has_indentation_directive(source: &str) -> bool {
    for line in source.lines().take(5) {
        let t = line.trim();
        if t == "#syntax indentation"
            || t == "#mode python"
            || t == "#syntax python"
            || t == "# syntax: indentation"
        {
            return true;
        }
    }
    false
}

/// Convert indented block syntax to brace syntax.
/// Lines ending in `:` (that aren't dict/type-annotation colons) introduce
/// a new indented block; decreasing indentation closes blocks with `}`.
pub fn preprocess(source: &str) -> String {
    let mut output = String::with_capacity(source.len() + 256);
    let mut indent_stack: Vec<usize> = vec![0]; // current indent levels
    let mut pending_open = false; // true after a `:` line, waiting to insert `{`
    let mut last_indent = 0usize;

    for raw_line in source.lines() {
        // Skip the directive line itself
        let t = raw_line.trim();
        if t == "#syntax indentation"
            || t == "#mode python"
            || t == "#syntax python"
            || t == "# syntax: indentation"
        {
            continue;
        }

        // Skip blank lines and pure-comment lines
        if t.is_empty() || t.starts_with("//") || t.starts_with("--") {
            output.push_str(raw_line);
            output.push('\n');
            continue;
        }

        // Measure indentation (spaces; treat 1 tab = 4 spaces)
        let indent = raw_line
            .chars()
            .take_while(|c| *c == ' ' || *c == '\t')
            .map(|c| if c == '\t' { 4 } else { 1 })
            .sum::<usize>();

        // If we were waiting to open a block and indentation increased → insert {
        if pending_open && indent > last_indent {
            indent_stack.push(indent);
            // Replace the trailing `:` on the previous line with ` {`
            // (already emitted — patch the output buffer)
            if output.ends_with(":\n") {
                let len = output.len();
                output.truncate(len - 2);
                output.push_str(" {\n");
            }
            pending_open = false;
        } else if pending_open {
            // No indent increase — just strip the `:` (it was a type annotation)
            if output.ends_with(":\n") {
                let len = output.len();
                output.truncate(len - 2);
                output.push('\n');
            }
            pending_open = false;
        }

        // Dedent — close blocks
        // Special case: `else` and `elif` at a lower indent level must
        // merge with the preceding `}` so the parser sees `} else {` on
        // one line rather than `}\nelse {` (which the Quantum parser does
        // not accept — it would see a standalone `else` after a closed block).
        let is_else_or_elif = matches!(t.split_whitespace().next(), Some("else") | Some("elif")
            | Some("sino") | Some("sinon") | Some("sonst") | Some("sonstwenn"));

        while indent < *indent_stack.last().unwrap_or(&0) {
            indent_stack.pop();
            let close_indent = indent_stack.last().copied().unwrap_or(0);
            if is_else_or_elif && indent == *indent_stack.last().unwrap_or(&0) {
                // This is the last dedent before an else/elif — merge
                // the `}` onto the same line as the upcoming `else`/`elif`
                // by trimming the trailing `\n` and not adding a new one.
                // We'll prepend `} ` to the current line when we emit it.
                output.push_str(&" ".repeat(close_indent));
                output.push_str("} ");
            } else {
                output.push_str(&" ".repeat(close_indent));
                output.push_str("}\n");
            }
        }

        last_indent = indent;

        // Check if this line introduces a new block (ends with `:` after
        // stripping comments, but not a dict literal or type annotation).
        // Heuristic: the `:` must be the very last non-whitespace character,
        // and the line must start with a block keyword or look like a
        // function/struct definition.
        let stripped = strip_trailing_comment(t);
        let introduces_block = stripped.ends_with(':')
            && is_block_intro(stripped);

        if introduces_block {
            // Emit the line with the `:` — we'll replace it with `{` when
            // we see the next line's indentation level.
            if is_else_or_elif {
                // The `} ` prefix was already emitted by the dedent loop;
                // emit just the trimmed keyword (no leading spaces).
                output.push_str(t);
                output.push('\n');
            } else {
                output.push_str(raw_line);
                output.push('\n');
            }
            pending_open = true;
        } else {
            output.push_str(raw_line);
            output.push('\n');
        }
    }

    // Close any remaining open blocks at EOF
    while indent_stack.len() > 1 {
        indent_stack.pop();
        let close_indent = indent_stack.last().copied().unwrap_or(0);
        output.push_str(&" ".repeat(close_indent));
        output.push_str("}\n");
    }
    // Also handle pending_open at EOF (empty block)
    if pending_open && output.ends_with(":\n") {
        let len = output.len();
        output.truncate(len - 2);
        output.push_str(" {}\n");
    }

    output
}

/// Strip a trailing `//` or `#` comment from a line.
fn strip_trailing_comment(line: &str) -> &str {
    // Find the first `//` or ` #` that's not inside a string
    let mut in_str = false;
    let mut quote_char = '"';
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '"' | '\'' if !in_str => { in_str = true; quote_char = chars[i]; }
            c if in_str && c == quote_char => { in_str = false; }
            '/' if !in_str && i + 1 < chars.len() && chars[i + 1] == '/' => {
                return line[..i].trim_end();
            }
            '#' if !in_str => {
                return line[..i].trim_end();
            }
            _ => {}
        }
        i += 1;
    }
    line.trim_end()
}

/// Returns true if the line (without trailing `:`) looks like it introduces
/// a new indented block in Python/indentation syntax. Avoids treating dict
/// literals and type annotations as block introductions.
fn is_block_intro(line: &str) -> bool {
    let t = line.trim_end_matches(':').trim();
    // Block keywords
    let starts_with_kw = |kw: &str| {
        t.starts_with(kw) && t[kw.len()..].starts_with(|c: char| c.is_whitespace() || c == '(')
    };
    starts_with_kw("fn")
        || starts_with_kw("def")
        || starts_with_kw("function")
        || starts_with_kw("funcion")
        || starts_with_kw("fonction")
        || starts_with_kw("funktion")
        || starts_with_kw("if")
        || starts_with_kw("si")
        || starts_with_kw("wenn")
        || starts_with_kw("else")
        || starts_with_kw("elif")
        || starts_with_kw("while")
        || starts_with_kw("for")
        || starts_with_kw("para")
        || starts_with_kw("loop")
        || starts_with_kw("match")
        || starts_with_kw("struct")
        || starts_with_kw("class")
        || starts_with_kw("enum")
        || starts_with_kw("trait")
        || starts_with_kw("impl")
        // Bare `else:` and `elif ...:` lines
        || t == "else"
}
