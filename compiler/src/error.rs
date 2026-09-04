// Quantum Error Handling
use crate::lexer::Span;
use thiserror::Error;
use colored::*;

#[derive(Error, Debug)]
pub enum QuantumError {
    #[error("Lexical error: {0}")]
    LexError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Type error: {0}")]
    TypeError(String),

    #[error("Codegen error: {0}")]
    CodegenError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub struct ErrorReporter {
    source: String,
    filename: String,
}

impl ErrorReporter {
    pub fn new(source: String, filename: String) -> Self {
        Self { source, filename }
    }

    pub fn report(&self, error: &str, span: Span) {
        eprintln!("\n{} {}", "error:".bright_red().bold(), error);

        let line = self.get_line(span.line);

        eprintln!(
            "  {} {}:{}:{}",
            "-->".bright_blue().bold(),
            self.filename,
            span.line,
            span.column
        );

        eprintln!("   {}", "|".bright_blue().bold());
        eprintln!(
            " {} {} {}",
            format!("{}", span.line).bright_blue().bold(),
            "|".bright_blue().bold(),
            line
        );

        let padding = " ".repeat(span.column - 1);
        let underline = "^".repeat(span.end - span.start);
        eprintln!(
            "   {} {}{}",
            "|".bright_blue().bold(),
            padding,
            underline.bright_red().bold()
        );
        eprintln!();
    }

    fn get_line(&self, line_num: usize) -> String {
        self.source
            .lines()
            .nth(line_num - 1)
            .unwrap_or("")
            .to_string()
    }
}
