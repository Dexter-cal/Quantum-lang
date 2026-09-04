//! Module resolution — lets a Quantum project span multiple `.qtm` files.
//!
//! `import math_utils` in `main.qtm` looks for `math_utils.qtm` in the same
//! directory, parses it (respecting *that file's own* `#syntax` directive
//! and `quantum_lang.toml` dictionary — each file's syntax style is
//! independent), and merges its top-level functions/structs/enums into the
//! combined program before typechecking.
//!
//! This is what makes "different syntax per file in the same project"
//! actually meaningful: `main.qtm` can be brace-style while
//! `math_utils.qtm` is Python-style, and `import math_utils` stitches them
//! together transparently.
//!
//! Resolution is a simple one-pass BFS over the import graph with cycle
//! detection (a file that (transitively) imports itself is an error, not
//! an infinite loop).

use crate::ast::{Item, Program};
use crate::lexer::Lexer;
use crate::parser::Parser as QParser;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Resolve and merge all imports starting from `entry_path`, returning a
/// single combined `Program` containing every function/struct/enum from
/// the entry file and all (transitively) imported files.
///
/// `read_and_preprocess` is a callback so the caller controls how a file's
/// raw text becomes lexer-ready source (i.e. applies the indentation
/// preprocessor and dictionary normalization exactly as the single-file
/// compile path does) — this avoids duplicating that logic here.
pub fn resolve_modules(
    entry_path: &Path,
    read_and_preprocess: impl Fn(&Path) -> Result<String, String> + Copy,
) -> Result<Program, String> {
    let mut visited: HashSet<PathBuf> = HashSet::new();
    let mut all_items: Vec<Item> = Vec::new();
    let mut stack: Vec<PathBuf> = Vec::new();
    let mut safe_mode = false;
    let mut namespace_aliases: NamespaceMap = std::collections::HashMap::new();

    resolve_recursive(entry_path, &read_and_preprocess, &mut visited, &mut stack, &mut all_items, &mut safe_mode, &mut namespace_aliases)?;

    // Rewrite namespaced calls: `qai.create(x)` → `ai_create(x)`
    // based on `import ai as qai` declarations collected during resolution.
    if !namespace_aliases.is_empty() {
        for item in all_items.iter_mut() {
            rewrite_item_namespaces(item, &namespace_aliases);
        }
    }

    Ok(Program { items: all_items, safe_mode })
}

/// Maps alias -> module_name (e.g. "qai" -> "ai", "data" -> "quantumai_data")
type NamespaceMap = std::collections::HashMap<String, String>;

fn rewrite_item_namespaces(item: &mut Item, ns: &NamespaceMap) {
    use crate::ast::*;
    match item {
        Item::Function(f) => rewrite_block_ns(&mut f.body, ns),
        Item::Impl(impl_block) => {
            for m in impl_block.methods.iter_mut() {
                rewrite_block_ns(&mut m.body, ns);
            }
        }
        _ => {}
    }
}

fn rewrite_block_ns(block: &mut crate::ast::Block, ns: &NamespaceMap) {
    for stmt in block.stmts.iter_mut() { rewrite_stmt_ns(stmt, ns); }
    if let Some(e) = block.expr.as_mut() { rewrite_expr_ns(e, ns); }
}

fn rewrite_stmt_ns(stmt: &mut crate::ast::Stmt, ns: &NamespaceMap) {
    use crate::ast::Stmt;
    match stmt {
        Stmt::Let { init: Some(e), .. } => rewrite_expr_ns(e, ns),
        Stmt::Expr { expr, .. } => rewrite_expr_ns(expr, ns),
        Stmt::Assign { value, .. } => rewrite_expr_ns(value, ns),
        Stmt::Return { value: Some(e), .. } => rewrite_expr_ns(e, ns),
        Stmt::While { condition, body, .. } => { rewrite_expr_ns(condition, ns); rewrite_block_ns(body, ns); }
        Stmt::For { iter, body, .. } => { rewrite_expr_ns(iter, ns); rewrite_block_ns(body, ns); }
        Stmt::Loop { body, .. } => rewrite_block_ns(body, ns),
        Stmt::DoWhile { body, condition, .. } => { rewrite_block_ns(body, ns); rewrite_expr_ns(condition, ns); }

        _ => {}
    }
}

fn rewrite_expr_ns(expr: &mut crate::ast::Expr, ns: &NamespaceMap) {
    use crate::ast::Expr;
    match expr {
        // `alias.func(args)` → `func(args)`
        // The alias is just the namespace qualifier — the function lives
        // in the merged program under its own name (not prefixed). This
        // matches Python's `import ai as qai; qai.banner()` semantics.
        Expr::MethodCall { receiver, method, args, span } => {
            if let Expr::Ident { name: recv_name, .. } = receiver.as_ref() {
                if ns.contains_key(recv_name.as_str()) {
                    // Rewrite: qai.banner() → banner()
                    let new_func = Box::new(Expr::Ident { name: method.clone(), span: *span });
                    let mut new_args = args.clone();
                    for a in new_args.iter_mut() { rewrite_expr_ns(a, ns); }
                    *expr = Expr::Call { func: new_func, args: new_args, span: *span };
                    return;
                }
            }
            // Not a namespace call — recurse normally
            rewrite_expr_ns(receiver, ns);
            for a in args.iter_mut() { rewrite_expr_ns(a, ns); }
        }
        Expr::Call { func, args, .. } => {
            rewrite_expr_ns(func, ns);
            for a in args.iter_mut() { rewrite_expr_ns(a, ns); }
        }
        Expr::Binary { left, right, .. } => { rewrite_expr_ns(left, ns); rewrite_expr_ns(right, ns); }
        Expr::Unary { expr, .. } => rewrite_expr_ns(expr, ns),
        Expr::If { condition, then_block, else_block, .. } => {
            rewrite_expr_ns(condition, ns);
            rewrite_block_ns(then_block, ns);
            if let Some(b) = else_block { rewrite_block_ns(b, ns); }
        }
        Expr::Ternary { condition, then_expr, else_expr, .. } => {
            rewrite_expr_ns(condition, ns);
            rewrite_expr_ns(then_expr, ns);
            rewrite_expr_ns(else_expr, ns);
        }
        Expr::FieldAccess { expr, .. } => rewrite_expr_ns(expr, ns),
        Expr::Index { expr, index, .. } => { rewrite_expr_ns(expr, ns); rewrite_expr_ns(index, ns); }
        Expr::Array { elements, .. } => { for e in elements.iter_mut() { rewrite_expr_ns(e, ns); } }
        Expr::StructInit { fields, .. } => { for (_, e) in fields.iter_mut() { rewrite_expr_ns(e, ns); } }
        Expr::Some { value, .. } | Expr::Ok { value, .. } | Expr::Err { value, .. } => rewrite_expr_ns(value, ns),
        Expr::Try { expr, .. } => rewrite_expr_ns(expr, ns),
        Expr::NamedArg { value, .. } => rewrite_expr_ns(value, ns),
        _ => {}
    }
}

fn resolve_recursive(
    path: &Path,
    read_and_preprocess: &impl Fn(&Path) -> Result<String, String>,
    visited: &mut HashSet<PathBuf>,
    stack: &mut Vec<PathBuf>,
    all_items: &mut Vec<Item>,
    safe_mode: &mut bool,
    namespace_aliases: &mut NamespaceMap,
) -> Result<(), String> {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

    if stack.contains(&canonical) {
        return Err(format!(
            "Circular import detected: {} imports itself (directly or transitively)",
            path.display()
        ));
    }
    if visited.contains(&canonical) {
        return Ok(()); // already merged — diamond imports are fine
    }
    visited.insert(canonical.clone());
    stack.push(canonical.clone());

    let source = read_and_preprocess(path)?;
    if source.contains("#[safe_mode]") { *safe_mode = true; }

    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize().map_err(|e| format!("{} — in {}", e, path.display()))?;
    let mut parser = QParser::new(tokens);
    let program = parser.parse().map_err(|e| format!("{} — in {}", e, path.display()))?;

    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));

    for item in program.items {
        match &item {
            Item::Import(import) => {
                let module_name = import.path.first()
                    .ok_or_else(|| "Empty import path".to_string())?;

                let import_file = find_module_file(base_dir, &import.path)
                    .ok_or_else(|| format!(
                        "Cannot find module '{}' (searched local directory '{}' and standard library locations)",
                        module_name, base_dir.display()
                    ))?;

                // Collect alias mapping: `import foo as bar` → bar→foo
                if let Some(alias) = &import.alias {
                    namespace_aliases.insert(alias.clone(), module_name.clone());
                }
                resolve_recursive(&import_file, read_and_preprocess, visited, stack, all_items, safe_mode, namespace_aliases)?;
            }
            _ => {
                all_items.push(item);
            }
        }
    }

    stack.pop();
    Ok(())
}

/// Helper function to locate module files locally or in standard library search paths (`qtm-std`).
fn find_module_file(base_dir: &Path, path_segments: &[String]) -> Option<PathBuf> {
    if path_segments.is_empty() { return None; }
    let module_name = &path_segments[0];

    // 1. Direct match in base_dir: base_dir/module_name.qtm
    let candidate = base_dir.join(format!("{}.qtm", module_name));
    if candidate.exists() { return Some(candidate); }

    // 2. Multi-segment match in base_dir: base_dir/segment1/segment2.qtm
    if path_segments.len() > 1 {
        let rel_path = format!("{}.qtm", path_segments.join("/"));
        let candidate = base_dir.join(rel_path);
        if candidate.exists() { return Some(candidate); }
    }

    // 3. Search standard library directories (qtm-std)
    let mut std_dirs = vec![
        PathBuf::from("qtm-std"),
        PathBuf::from("../qtm-std"),
        base_dir.join("qtm-std"),
        base_dir.join("../qtm-std"),
        base_dir.join("../../qtm-std"),
    ];

    if let Ok(env_path) = std::env::var("QUANTUM_STD_PATH") {
        std_dirs.push(PathBuf::from(env_path));
    }
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(parent) = exe_path.parent() {
            std_dirs.push(parent.join("qtm-std"));
            std_dirs.push(parent.join("../qtm-std"));
            std_dirs.push(parent.join("../share/quantum/qtm-std"));
        }
    }

    // System standard locations
    std_dirs.push(PathBuf::from("/usr/local/lib/quantum/qtm-std"));
    std_dirs.push(PathBuf::from("/usr/share/quantum/qtm-std"));

    // Name mapping for standard/builtin framework imports
    let mut mapped_names = Vec::new();
    match module_name.as_str() {
        "quantumai" | "ai" => {
            mapped_names.push("ai.qtm".to_string());
            mapped_names.push("quantumai.qtm".to_string());
        }
        "std" if path_segments.len() > 1 => {
            let sub = &path_segments[1];
            mapped_names.push(format!("{}.qtm", sub));
        }
        _ => {
            mapped_names.push(format!("{}.qtm", module_name));
        }
    }

    for dir in &std_dirs {
        for file_name in &mapped_names {
            let candidate = dir.join(file_name);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    None
}

/// Scan source text for `#link "libname"` directives, used to request
/// additional libraries be passed to the linker (e.g. `#link "curl"` for
/// `extern "C" { fn curl_easy_init() -> int }`). Returns the library names
/// in the order they appear, without the `-l` prefix.
pub fn list_link_directives(source: &str) -> Vec<String> {
    let mut libs = Vec::new();
    for line in source.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("#link ") {
            let rest = rest.trim();
            let name = rest.trim_matches('"').trim_matches('\'');
            if !name.is_empty() {
                libs.push(name.to_string());
            }
        }
    }
    libs
}

/// Scan source text for `#lib-path "/path/to/lib"` directives.
/// These add -L and -rpath entries to the linker invocation.
pub fn list_lib_path_directives(source: &str) -> Vec<String> {
    let mut paths = Vec::new();
    for line in source.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("#lib-path ") {
            let rest = rest.trim();
            let path = rest.trim_matches('"').trim_matches('\'');
            if !path.is_empty() {
                paths.push(path.to_string());
            }
        }
    }
    paths
}

/// Scan source file's text for `import` statements without fully parsing
/// it — used for quick dependency listing (e.g. `quantumc deps <file>`).
pub fn list_imports(source: &str) -> Vec<String> {
    let mut imports = Vec::new();
    for line in source.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("import ") {
            let name: String = rest.chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                imports.push(name);
            }
        }
    }
    imports
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_module_file_stdlib() {
        let base_dir = Path::new("examples");
        let math_mod = find_module_file(base_dir, &["math".to_string()]);
        assert!(math_mod.is_some(), "Expected to find math in qtm-std");
        assert!(math_mod.unwrap().ends_with("math.qtm"));

        let std_math = find_module_file(base_dir, &["std".to_string(), "math".to_string()]);
        assert!(std_math.is_some(), "Expected to find std.math in qtm-std");

        let ai_mod = find_module_file(base_dir, &["quantumai".to_string()]);
        assert!(ai_mod.is_some(), "Expected to find quantumai as ai.qtm in qtm-std");
    }
}
