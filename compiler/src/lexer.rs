// Quantum Lexer - High-performance tokenizer using logos
use logos::Logos;
use std::fmt;

#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r]+")] // whitespace
#[logos(skip r"//[^\n]*")]  // C++/Quantum line comments
#[logos(skip r"--[^\n]*")]  // Lua / Haskell line comments
#[logos(skip r"#[^\n]*")]   // Python / Ruby / Shell line comments
#[logos(skip r"/\*([^*]|\*[^/])*\*/")]  // block comments
pub enum TokenKind {
    // Literals
    #[regex(r"-?[0-9][0-9_]*", |lex| lex.slice().replace('_', "").parse::<i64>().ok())]
    Int(i64),

    #[regex(r"-?[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9]+)?", |lex| lex.slice().replace('_', "").parse::<f64>().ok())]
    Float(f64),


    // F-string interpolation: f"hello {name}, age {age}"
    // Matches the whole f"..." literal; parser handles the {expr} splitting.
    #[regex(r#"f"([^"\\]|\\.)*""#, |lex| {
        let s = lex.slice();
        Some(s[2..s.len()-1].to_string())
    })]
    FString(String),

    // Double-quoted strings  "hello world"
    #[regex(r#""([^"\\]|\\.)*""#, |lex| {
        let s = lex.slice();
        Some(s[1..s.len()-1].to_string())
    })]
    // Single-quoted strings  'hello world'  (Python/JS/Ruby style)
    // Note: single chars like 'a' are handled by the Char rule below, but
    // multi-char single-quoted strings like 'hello' match this rule because
    // the Char rule requires exactly one character (or one escape sequence).
    #[regex(r#"'([^'\\]|\\.){2,}'"#, |lex| {
        let s = lex.slice();
        Some(s[1..s.len()-1].to_string())
    })]
    // Backtick strings  `hello world`  (JavaScript/Go raw-string style)
    #[regex(r#"`[^`]*`"#, |lex| {
        let s = lex.slice();
        Some(s[1..s.len()-1].to_string())
    })]
    String(String),

    #[regex(r"'[^'\\]'|'\\[ntr\\']'", |lex| lex.slice().chars().nth(1))]
    Char(char),

    #[token("true")]
    #[token("yes")]
    True,

    #[token("false")]
    #[token("no")]
    False,

    #[token("null")]
    #[token("nil")]
    #[token("none")]
    Null,

    // Keywords
    #[token("fn")]
    #[token("function")]
    #[token("def")]
    Fn,

    #[token("let")]
    #[token("var")]
    Let,

    #[token("mut")]
    #[token("mutable")]
    Mut,

    #[token("const")]
    Const,

    #[token("if")]
    If,

    #[token("else")]
    Else,

    #[token("elif")]
    #[token("elseif")]
    Elif,

    #[token("while")]
    While,

    #[token("for")]
    For,

    #[token("in")]
    In,

    #[token("loop")]
    Loop,

    #[token("do")]
    Do,

    #[token("break")]
    Break,

    #[token("continue")]
    Continue,

    #[token("return")]
    Return,

    #[token("Some")]
    SomeKw,

    #[token("None")]
    NoneKw,

    #[token("Ok")]
    OkKw,

    #[token("Err")]
    ErrKw,

    #[token("match")]
    #[token("switch")]
    Match,

    #[token("struct")]
    #[token("class")]
    Struct,

    #[token("enum")]
    Enum,

    #[token("trait")]
    #[token("interface")]
    Trait,

    #[token("extern")]
    Extern,

    #[token("impl")]
    Impl,

    #[token("import")]
    #[token("use")]
    #[token("require")]
    Import,

    #[token("from")]
    From,

    #[token("as")]
    As,

    #[token("pub")]
    #[token("public")]
    Pub,

    #[token("private")]
    Private,

    #[token("async")]
    Async,

    #[token("await")]
    Await,


    #[token("trainer")]
    Trainer,

    #[token("self")]
    SelfKw,

    #[token("super")]
    Super,

    #[token("print")]
    Print,
    #[token("println")]
    Println,

    // Identifiers (ASCII) + Unicode / emoji identifiers.
    // The second regex matches sequences of Unicode chars that are NOT
    // ASCII printable or whitespace — this covers emoji (e.g. 🔧),
    // CJK characters (e.g. 関数), and other non-ASCII scripts, allowing
    // the keyword dictionary to map them to canonical tokens.
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    #[regex(r"[\x{0080}-\x{10FFFF}][\x{0080}-\x{10FFFF}]*", |lex| lex.slice().to_string())]
    Ident(String),

    // Operators
    #[token("+")]
    Plus,

    #[token("-")]
    Minus,

    #[token("*")]
    Star,

    #[token("/")]
    Slash,

    #[token("%")]
    Percent,

    #[token("**")]
    Power,

    #[token("=")]
    Eq,

    #[token("==")]
    EqEq,

    #[token("!=")]
    Ne,

    #[token("<")]
    Lt,

    #[token("<=")]
    Le,

    #[token(">")]
    Gt,

    #[token(">=")]
    Ge,

    #[token("&&")]
    And,

    #[token("||")]
    Or,

    #[token("!")]
    Not,

    #[token("&")]
    Ampersand,

    #[token("|")]
    Pipe,

    #[token("^")]
    Caret,

    #[token("~")]
    Tilde,

    #[token("<<")]
    Lshift,

    #[token(">>")]
    Rshift,

    #[token("+=")]
    PlusEq,

    #[token("-=")]
    MinusEq,

    #[token("*=")]
    StarEq,

    #[token("/=")]
    SlashEq,

    // Delimiters
    #[token("(")]
    LParen,

    #[token(")")]
    RParen,

    #[token("{")]
    LBrace,

    #[token("}")]
    RBrace,

    #[token("[")]
    LBracket,

    #[token("]")]
    RBracket,

    #[token(",")]
    Comma,

    #[token(".")]
    Dot,

    #[token("..")]
    DotDot,

    #[token("..=")]
    DotDotEq,

    #[token(":")]
    Colon,

    #[token("::")]
    ColonColon,

    #[token(";")]
    Semicolon,

    #[token("->")]
    Arrow,

    #[token("=>")]
    FatArrow,

    #[token("?")]
    Question,

    #[token("@")]
    At,

    #[token("|>")]
    PipeOp,

    // Special
    #[token("\n")]
    Newline,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TokenKind::Int(n) => write!(f, "{}", n),
            TokenKind::Float(n) => write!(f, "{}", n),
            TokenKind::String(s) => write!(f, "\"{}\"", s),
            TokenKind::FString(s) => write!(f, "f\"{}\"", s),
            TokenKind::Char(c) => write!(f, "'{}'", c),
            TokenKind::Ident(s) => write!(f, "{}", s),
            _ => write!(f, "{:?}", self),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub column: usize,
}

pub struct Lexer<'source> {
    lexer: logos::Lexer<'source, TokenKind>,
    source: &'source str,
    line: usize,
    line_start: usize,
}

impl<'source> Lexer<'source> {
    pub fn new(source: &'source str) -> Self {
        Self {
            lexer: TokenKind::lexer(source),
            source,
            line: 1,
            line_start: 0,
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::new();

        while let Some(result) = self.lexer.next() {
            let kind = result.map_err(|_| {
                format!(
                    "Invalid token at line {}, column {}",
                    self.line,
                    self.lexer.span().start - self.line_start + 1
                )
            })?;

            // Track newlines for line numbers
            if matches!(kind, TokenKind::Newline) {
                self.line += 1;
                self.line_start = self.lexer.span().end;
                continue; // Skip newlines in token stream for now
            }

            let span = Span {
                start: self.lexer.span().start,
                end: self.lexer.span().end,
                line: self.line,
                column: self.lexer.span().start - self.line_start + 1,
            };

            tokens.push(Token { kind, span });
        }

        // Dictionary normalization pass: if a quantum_lang.toml is present
        // in the current working directory, replace any Ident tokens that
        // match a user-defined alias with the corresponding canonical
        // TokenKind. This is what makes Quantum multilingual.
        if let Some(dict) = crate::dictionary::Dictionary::load_from_dir(
            std::path::Path::new(".")
        ) {
            if !dict.is_empty() {
                for token in tokens.iter_mut() {
                    if let TokenKind::Ident(ref s) = token.kind.clone() {
                        if let Some(canonical) = dict.normalize(s) {
                            token.kind = canonical;
                        }
                    }
                }
            }
        }

        Ok(tokens)
    }

    pub fn source(&self) -> &'source str {
        self.source
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer_basic() {
        let source = "let x = 42";
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens.len(), 4);
        assert!(matches!(tokens[0].kind, TokenKind::Let));
        assert!(matches!(tokens[1].kind, TokenKind::Ident(_)));
        assert!(matches!(tokens[2].kind, TokenKind::Eq));
        assert!(matches!(tokens[3].kind, TokenKind::Int(42)));
    }

    #[test]
    fn test_lexer_function() {
        let source = r#"fn add(a: int, b: int) -> int {
    return a + b
}"#;
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();

        assert!(tokens.len() > 0);
        assert!(matches!(tokens[0].kind, TokenKind::Fn));
    }

    #[test]
    fn test_lexer_strings() {
        let source = r#"let s = "hello world""#;
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();

        if let TokenKind::String(s) = &tokens[3].kind {
            assert_eq!(s, "hello world");
        } else {
            panic!("Expected string token");
        }
    }
}

// ─── Additional tokens from language spec ───────────────────────────────────
// These were in the spec docs but missing from our lexer
