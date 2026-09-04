//! Library interface to the Quantum compiler's front-end stages
//! (lexer, parser, typechecker). Exposed so external tools — notably
//! `quantum-lsp` — can run diagnostics without shelling out to the
//! `quantumc` binary.

pub mod lexer;
pub mod parser;
pub mod ast;
pub mod typechecker;
pub mod mir;
pub mod codegen;
pub mod error;
pub mod utils;
pub mod dictionary;
pub mod preprocessor;
pub mod monomorphize;

pub use lexer::{Lexer, Span, Token, TokenKind};
pub use parser::Parser as QParser;
pub use typechecker::TypeChecker;
pub use dictionary::Dictionary;
pub use preprocessor::{has_indentation_directive, preprocess};

/// A diagnostic produced while checking a source file, with best-effort
/// position information. `line`/`column` are 1-indexed when known (matching
/// the lexer's `Span`); `None` means the underlying error didn't carry
/// position info (this is currently true for most typechecker errors).
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub message: String,
    pub line: Option<usize>,
    pub column: Option<usize>,
    pub severity: Severity,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Severity {
    Error,
    Warning,
}

/// Attempt to extract "line N, column M" or "at line N, column M" from an
/// error message produced by the parser (see parser.rs's `expect`/
/// `parse_primary` error sites, which embed this format).
fn extract_position(msg: &str) -> (Option<usize>, Option<usize>) {
    // Look for "line <N>, column <M>"
    if let Some(idx) = msg.find("line ") {
        let rest = &msg[idx + 5..];
        let line_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(line) = line_str.parse::<usize>() {
            if let Some(cidx) = rest.find("column ") {
                let crest = &rest[cidx + 7..];
                let col_str: String = crest.chars().take_while(|c| c.is_ascii_digit()).collect();
                if let Ok(col) = col_str.parse::<usize>() {
                    return (Some(line), Some(col));
                }
            }
            return (Some(line), None);
        }
    }
    (None, None)
}

/// Strip a trailing " at line N, column M" (or " at line N") suffix from an
/// error message, since that information is conveyed structurally via
/// `Diagnostic.line`/`column` (and the LSP `range`) — repeating it in the
/// message text is redundant for editor tooling, though the CLI (`quantumc
/// check`) still benefits from the inline text.
fn strip_position_suffix(msg: &str) -> String {
    if let Some(idx) = msg.find(" at line ") {
        msg[..idx].to_string()
    } else {
        msg.to_string()
    }
}

/// Run lex -> parse -> typecheck on the given source text and collect all
/// diagnostics found. Stops at the first error in each stage (the current
/// compiler front-end is not error-recovering), but always runs as many
/// stages as possible: a lex error prevents parsing, a parse error prevents
/// typechecking, etc.
pub fn check_source(source: &str) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();

    // Preprocessing: handle indentation-based syntax before lexing
    let preprocessed;
    let source = if preprocessor::has_indentation_directive(source) {
        preprocessed = preprocessor::preprocess(source);
        preprocessed.as_str()
    } else {
        source
    };

    let mut lexer = Lexer::new(source);
    let tokens = match lexer.tokenize() {
        Ok(t) => t,
        Err(e) => {
            let (line, column) = extract_position(&e);
            diagnostics.push(Diagnostic { message: strip_position_suffix(&e), line, column, severity: Severity::Error });
            return diagnostics;
        }
    };

    // Parse stage — recovering: collects multiple parse errors
    let mut parse_errors: Vec<String> = Vec::new();
    let mut parser = QParser::new(tokens);
    let ast = parser.parse_recovering(&mut parse_errors);
    for e in parse_errors {
        let (line, column) = extract_position(&e);
        diagnostics.push(Diagnostic { message: strip_position_suffix(&e), line, column, severity: Severity::Error });
    }
    // Monomorphize generics before typechecking, same as the main
    // compile pipeline, so generic-function diagnostics behave
    // consistently with what the actual compiler will accept.
    let ast = monomorphize::monomorphize(ast);

    // Typecheck stage — recovering: collects multiple type errors
    let mut typechecker = TypeChecker::new();
    let type_errors = typechecker.check_all(&ast);
    for e in type_errors {
        let (line, column) = extract_position(&e);
        diagnostics.push(Diagnostic {
            message: strip_position_suffix(&e),
            line,
            column,
            severity: Severity::Error,
        });
    }

    diagnostics
}
