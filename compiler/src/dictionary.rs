//! Keyword dictionary — allows any word (or emoji) to be used as an alias
//! for a Quantum keyword, loaded from `quantum_lang.toml` at compile time.
//!
//! This is what makes Quantum multilingual: a Spanish developer can write
//! `funcion`, `si`, `sino`, `retornar` and the compiler accepts them
//! identically to `fn`, `if`, `else`, `return`. A Python developer can
//! write `def`, `True`, `False`, `None`. An emoji enthusiast can write `🔧`.
//!
//! # Config file format (`quantum_lang.toml`)
//! ```toml
//! [keywords]
//! fn     = ["funcion", "función", "def", "🔧"]
//! let    = ["variable", "var"]
//! if     = ["si", "wenn", "もし"]
//! return = ["retornar", "retour", "zurückgeben"]
//! # ... etc.
//! ```
//!
//! The file is searched for in the current directory, then the directory of
//! the source file being compiled.

use crate::lexer::TokenKind;
use std::collections::HashMap;
use std::path::Path;

/// Maps alias strings to their canonical `TokenKind`.
/// Built once per compilation and passed into the lexer's normalization pass.
#[derive(Default, Clone)]
pub struct Dictionary {
    /// alias (lowercase) → canonical TokenKind
    map: HashMap<String, TokenKind>,
}

impl Dictionary {
    /// Load a dictionary from the given TOML source text.
    /// Unknown keyword names are silently skipped (forward-compat).
    pub fn from_toml(toml_src: &str) -> Self {
        let mut dict = Dictionary::default();
        // Minimal TOML parser — we only need [keywords] section with
        // `key = [...]` arrays. Avoids adding a toml crate dependency.
        let mut in_keywords = false;
        for raw_line in toml_src.lines() {
            let line = raw_line.trim();
            if line.starts_with('[') {
                in_keywords = line == "[keywords]" || line.starts_with("[keywords]");
                continue;
            }
            if !in_keywords || line.starts_with('#') || line.is_empty() {
                continue;
            }
            // Parse: `key = ["alias1", "alias2", ...]`
            if let Some((key, rest)) = line.split_once('=') {
                let key = key.trim();
                let aliases = parse_string_array(rest.trim());
                let kind = match canonical_kind(key) {
                    Some(k) => k,
                    None => continue,
                };
                for alias in aliases {
                    dict.map.insert(alias.to_lowercase(), kind.clone());
                    // Also store the exact-case version so emoji / CJK chars
                    // (which don't have a "lowercase") are matched verbatim.
                    if alias.to_lowercase() != alias {
                        dict.map.insert(alias.clone(), kind.clone());
                    }
                }
            }
        }
        dict
    }

    /// Try to load a dictionary from `quantum_lang.toml` in `search_dir`.
    /// Returns `None` if the file doesn't exist or can't be parsed.
    pub fn load_from_dir(search_dir: &Path) -> Option<Self> {
        let path = search_dir.join("quantum_lang.toml");
        let src = std::fs::read_to_string(&path).ok()?;
        Some(Self::from_toml(&src))
    }

    /// Try to normalise `ident` to its canonical `TokenKind`.
    /// Returns `None` if this identifier is not a known alias.
    pub fn normalize(&self, ident: &str) -> Option<TokenKind> {
        // Try exact match first (preserves emoji / CJK)
        if let Some(k) = self.map.get(ident) {
            return Some(k.clone());
        }
        // Then lowercase
        self.map.get(&ident.to_lowercase()).cloned()
    }

    /// Is this dictionary empty (no aliases configured)?
    pub fn is_empty(&self) -> bool { self.map.is_empty() }

    /// Return all entries as (alias, canonical_name) pairs, sorted.
    pub fn entries(&self) -> Vec<(String, String)> {
        let mut v: Vec<_> = self.map.iter()
            .map(|(alias, kind)| (alias.clone(), kind_name(kind).to_string()))
            .collect();
        v.sort();
        v
    }
}

/// Map a canonical keyword name (from the TOML key) to its `TokenKind`.
fn canonical_kind(key: &str) -> Option<TokenKind> {
    Some(match key.trim() {
        "fn" | "function"    => TokenKind::Fn,
        "let" | "variable"   => TokenKind::Let,
        "mut" | "mutable"    => TokenKind::Mut,
        "const"              => TokenKind::Const,
        "if"                 => TokenKind::If,
        "else"               => TokenKind::Else,
        "elif"               => TokenKind::Elif,
        "while"              => TokenKind::While,
        "for"                => TokenKind::For,
        "in"                 => TokenKind::In,
        "loop"               => TokenKind::Loop,
        "break"              => TokenKind::Break,
        "continue"           => TokenKind::Continue,
        "return"             => TokenKind::Return,
        "match" | "switch"   => TokenKind::Match,
        "struct" | "class"   => TokenKind::Struct,
        "enum"               => TokenKind::Enum,
        "trait" | "interface" => TokenKind::Trait,
        "impl"               => TokenKind::Impl,
        "true"               => TokenKind::True,
        "false"              => TokenKind::False,
        "null" | "nil" | "none" => TokenKind::Null,
        "print"              => TokenKind::Print,
        "println"            => TokenKind::Println,
        "and"                => TokenKind::And,
        "or"                 => TokenKind::Or,
        "not"                => TokenKind::Not,
        "pub" | "public"     => TokenKind::Pub,
        "import" | "use"     => TokenKind::Import,
        "async"              => TokenKind::Async,
        "await"              => TokenKind::Await,
        "return_type"        => TokenKind::Arrow,
        _                    => return None,
    })
}

/// Human-readable name for a TokenKind (for `quantum dict show` output).
pub fn kind_name(kind: &TokenKind) -> &'static str {
    match kind {
        TokenKind::Fn       => "fn",
        TokenKind::Let      => "let",
        TokenKind::Mut      => "mut",
        TokenKind::Const    => "const",
        TokenKind::If       => "if",
        TokenKind::Else     => "else",
        TokenKind::Elif     => "elif",
        TokenKind::While    => "while",
        TokenKind::For      => "for",
        TokenKind::In       => "in",
        TokenKind::Loop     => "loop",
        TokenKind::Break    => "break",
        TokenKind::Continue => "continue",
        TokenKind::Return   => "return",
        TokenKind::Match    => "match",
        TokenKind::Struct   => "struct",
        TokenKind::Enum     => "enum",
        TokenKind::Trait    => "trait",
        TokenKind::Impl     => "impl",
        TokenKind::True     => "true",
        TokenKind::False    => "false",
        TokenKind::Null     => "null",
        TokenKind::Print    => "print",
        TokenKind::Println  => "println",
        TokenKind::And      => "and",
        TokenKind::Or       => "or",
        TokenKind::Not     => "not",
        TokenKind::Pub      => "pub",
        TokenKind::Import   => "import",
        TokenKind::Async    => "async",
        TokenKind::Await    => "await",
        _                   => "?",
    }
}

/// Parse a TOML string array `["a", "b", "c"]` into a Vec<String>.
/// Handles both `"double"` and `'single'` quoted strings and emoji.
fn parse_string_array(s: &str) -> Vec<String> {
    let mut result = Vec::new();
    let inner = s.trim().trim_start_matches('[').trim_end_matches(']');
    let mut current = String::new();
    let mut in_str = false;
    let mut quote_char = '"';

    for ch in inner.chars() {
        match ch {
            '"' | '\'' if !in_str => {
                in_str = true;
                quote_char = ch;
            }
            c if in_str && c == quote_char => {
                in_str = false;
                if !current.is_empty() {
                    result.push(current.clone());
                    current.clear();
                }
            }
            ',' if !in_str => {}
            c if in_str => current.push(c),
            _ => {}
        }
    }
    result
}

// ── Built-in language packs ───────────────────────────────────────────────

/// Return the TOML source for a named built-in language pack.
pub fn builtin_pack(name: &str) -> Option<&'static str> {
    match name {
        "python" => Some(PACK_PYTHON),
        "spanish" | "es" => Some(PACK_SPANISH),
        "french"  | "fr" => Some(PACK_FRENCH),
        "german"  | "de" => Some(PACK_GERMAN),
        "japanese"| "ja" => Some(PACK_JAPANESE),
        "emoji"          => Some(PACK_EMOJI),
        _ => None,
    }
}

pub const PACK_LIST: &[(&str, &str)] = &[
    ("python",   "Python-style syntax (def, True, False, None, elif)"),
    ("spanish",  "Spanish keywords (funcion, si, sino, para, mientras, retornar)"),
    ("french",   "French keywords (fonction, si, sinon, pour, tantque, retourner)"),
    ("german",   "German keywords (funktion, wenn, sonst, für, während, zurückgeben)"),
    ("japanese", "Japanese keywords (関数, もし, そうでなければ, 返す)"),
    ("emoji",    "Emoji keywords (🔧 fn, 📦 let, ❓ if, 🔁 for, ↩️ return, 💬 println)"),
];

static PACK_PYTHON: &str = r#"
[keywords]
fn       = ["def"]
if       = ["if"]
else     = ["else"]
elif     = ["elif"]
while    = ["while"]
for      = ["for"]
return   = ["return"]
true     = ["True"]
false    = ["False"]
null     = ["None"]
and      = ["and"]
or       = ["or"]
not      = ["not"]
struct   = ["class"]
import   = ["import"]
print    = ["print"]
println  = ["print"]
"#;

static PACK_SPANISH: &str = r#"
[keywords]
fn       = ["funcion", "función"]
let      = ["variable", "var"]
mut      = ["mutable"]
if       = ["si"]
else     = ["sino"]
elif     = ["sinosi"]
while    = ["mientras"]
for      = ["para"]
in       = ["en"]
return   = ["retornar", "devolver"]
true     = ["verdadero"]
false    = ["falso"]
null     = ["nulo"]
and      = ["y"]
or       = ["o"]
not      = ["no"]
struct   = ["estructura"]
print    = ["imprimir"]
println  = ["mostrar"]
import   = ["importar"]
break    = ["romper"]
continue = ["continuar"]
"#;

static PACK_FRENCH: &str = r#"
[keywords]
fn       = ["fonction"]
let      = ["variable", "var"]
mut      = ["mutable"]
if       = ["si"]
else     = ["sinon"]
elif     = ["sinonsi"]
while    = ["tantque"]
for      = ["pour"]
in       = ["dans"]
return   = ["retourner", "renvoyer"]
true     = ["vrai"]
false    = ["faux"]
null     = ["nul", "rien"]
and      = ["et"]
or       = ["ou"]
not      = ["non"]
struct   = ["structure"]
print    = ["afficher"]
println  = ["afficherln"]
import   = ["importer"]
break    = ["arreter"]
continue = ["continuer"]
"#;

static PACK_GERMAN: &str = r#"
[keywords]
fn       = ["funktion"]
let      = ["variable", "var"]
mut      = ["veränderlich"]
if       = ["wenn"]
else     = ["sonst"]
elif     = ["sonstwenn"]
while    = ["während", "waehrend"]
for      = ["für", "fuer"]
in       = ["in"]
return   = ["zurückgeben", "zurueckgeben"]
true     = ["wahr"]
false    = ["falsch"]
null     = ["nichts"]
and      = ["und"]
or       = ["oder"]
not      = ["nicht"]
struct   = ["struktur"]
print    = ["drucken"]
println  = ["ausgeben"]
import   = ["importieren"]
break    = ["abbrechen"]
continue = ["fortfahren"]
"#;

static PACK_JAPANESE: &str = r#"
[keywords]
fn       = ["関数", "かんすう"]
let      = ["変数", "へんすう"]
mut      = ["可変", "かへん"]
if       = ["もし", "場合", "ばあい"]
else     = ["そうでなければ", "でなければ"]
while    = ["間", "あいだ"]
for      = ["繰り返し", "くりかえし"]
return   = ["返す", "かえす"]
true     = ["真", "しん"]
false    = ["偽", "ぎ"]
null     = ["無", "む"]
print    = ["表示", "ひょうじ"]
println  = ["印刷", "いんさつ"]
break    = ["停止", "ていし"]
continue = ["続ける", "つづける"]
"#;

static PACK_EMOJI: &str = r#"
[keywords]
fn       = ["🔧", "⚙️"]
let      = ["📦", "📌"]
mut      = ["✏️"]
if       = ["❓", "🤔"]
else     = ["❗", "💡"]
while    = ["⏰"]
for      = ["🔁", "🔃"]
in       = ["→", "=>"]
return   = ["↩️", "⬅️"]
true     = ["✅", "👍"]
false    = ["❌", "👎"]
null     = ["⭕", "🚫"]
print    = ["🗣️"]
println  = ["💬", "📢"]
break    = ["🛑"]
continue = ["⏭️"]
import   = ["📥"]
"#;
