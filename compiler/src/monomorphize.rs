//! Monomorphization — Quantum's strategy for compiling generic functions.
//!
//! Quantum compiles to plain C, which has no generics. Rather than
//! building any runtime generic mechanism (vtables-of-everything, boxed
//! values, etc.), Quantum follows the same approach as C++ templates and
//! Rust generics: for every concrete type a generic function is actually
//! called with, generate a separate, fully concrete specialized copy of
//! that function, and rewrite the call site to use it. A generic function
//! that's never called produces no code at all; a generic function called
//! with three different types produces three separate functions.
//!
//! This runs as a pre-pass on the whole program, after module resolution
//! but before typechecking — by the time typechecking runs, every
//! generic function call has already been rewritten into a call to a
//! concrete, monomorphized function, and the typechecker never needs to
//! know generics existed at all.

use crate::ast::*;
use std::collections::HashMap;

pub fn monomorphize(mut program: Program) -> Program {
    // ── Generic struct monomorphization ──────────────────────────────
    // Collect generic struct definitions, scan usages, emit concrete versions,
    // and rename struct init expressions to use mangled names.
    {
        let generic_structs: HashMap<String, crate::ast::Struct> = program.items.iter()
            .filter_map(|item| {
                if let crate::ast::Item::Struct(s) = item {
                    if !s.type_params.is_empty() { Some((s.name.clone(), s.clone())) }
                    else { None }
                } else { None }
            })
            .collect();

        if !generic_structs.is_empty() {
            let mut new_structs: Vec<crate::ast::Item> = Vec::new();
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

            // Walk all items to find StructInit uses, collect concrete instantiations
            let snapshot = program.items.clone();
            for item in &snapshot {
                collect_struct_inits(item, &generic_structs, &mut seen, &mut new_structs);
            }

            // Insert concrete struct definitions right after generic definitions
            let insert_pos = program.items.iter().position(|it| {
                if let crate::ast::Item::Struct(s) = it {
                    generic_structs.contains_key(&s.name)
                } else { false }
            }).map(|p| p + 1).unwrap_or(0);

            for (i, new_item) in new_structs.into_iter().enumerate() {
                program.items.insert(insert_pos + i, new_item);
            }

            // Rewrite StructInit expressions to use mangled names
            for item in program.items.iter_mut() {
                rewrite_struct_inits(item, &generic_structs);
            }

            // Remove original generic struct definitions — only concrete
            // versions (e.g. Pair_int_float) should reach codegen.
            program.items.retain(|item| {
                if let crate::ast::Item::Struct(s) = item {
                    !generic_structs.contains_key(&s.name)
                } else {
                    true
                }
            });
        }
    }

    // Separate generic function definitions from everything else.
    let mut generic_fns: HashMap<String, Function> = HashMap::new();
    let mut other_items: Vec<Item> = Vec::new();

    for item in program.items {
        match item {
            Item::Function(f) if !f.type_params.is_empty() => {
                generic_fns.insert(f.name.clone(), f);
            }
            other => other_items.push(other),
        }
    }

    if generic_fns.is_empty() {
        // No generics in this program — nothing to do, restore items as-is.
        program.items = other_items;
        return program;
    }

    // Collect every (generic_fn_name, concrete_type_args) combination
    // actually used anywhere in the program, by scanning every call
    // expression in every function/method body for calls to a generic
    // function name, and inferring the concrete type of each type
    // parameter from the literal/known-type shape of the arguments.
    //
    // This is necessarily a heuristic (Quantum has no full type-inference
    // pass independent of monomorphization), so it only handles the
    // common, unambiguous case: a generic function called with
    // straightforward literal or already-typed-variable arguments. It's
    // the same class of heuristic already used elsewhere in Quantum's
    // pipeline (e.g. the Python-style untyped-parameter inference).
    let mut needed: Vec<(String, Vec<Type>)> = Vec::new();
    for item in &other_items {
        match item {
            Item::Function(f) => scan_block_for_generic_calls(&f.body, &generic_fns, &mut needed),
            Item::Impl(impl_block) => {
                for m in &impl_block.methods {
                    scan_block_for_generic_calls(&m.body, &generic_fns, &mut needed);
                }
            }
            _ => {}
        }
    }
    // Dedupe while preserving first-seen order (stable monomorphized
    // function ordering makes generated C easier to read/debug).
    let mut seen = std::collections::HashSet::new();
    needed.retain(|(name, args)| {
        let key = format!("{}::{}", name, args.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
        seen.insert(key)
    });

    // For each needed (generic_fn, concrete_args) pair, generate a
    // monomorphized concrete function and remember its mangled name so
    // call-site rewriting can redirect calls to it.
    let mut mangled_name_for: HashMap<(String, Vec<Type>), String> = HashMap::new();
    let mut monomorphized_fns: Vec<Item> = Vec::new();

    for (fn_name, type_args) in &needed {
        let generic_fn = match generic_fns.get(fn_name) {
            Some(f) => f,
            None => continue, // shouldn't happen, but stay defensive
        };
        if generic_fn.type_params.len() != type_args.len() {
            continue; // arity mismatch — leave for the typechecker to report as an error elsewhere
        }
        let substitution: HashMap<String, Type> = generic_fn.type_params.iter()
            .cloned()
            .zip(type_args.iter().cloned())
            .collect();

        let mangled = mangle_generic_name(fn_name, type_args);
        let mut specialized = generic_fn.clone();
        specialized.name = mangled.clone();
        specialized.type_params = Vec::new(); // fully concrete now
        for p in specialized.params.iter_mut() {
            p.ty = substitute_type(&p.ty, &substitution);
        }
        specialized.return_type = specialized.return_type.map(|t| substitute_type(&t, &substitution));
        substitute_block(&mut specialized.body, &substitution);

        mangled_name_for.insert((fn_name.clone(), type_args.clone()), mangled);
        monomorphized_fns.push(Item::Function(specialized));
    }

    // Rewrite every call site in the program to use the mangled,
    // monomorphized function name instead of the generic one.
    for item in other_items.iter_mut() {
        match item {
            Item::Function(f) => rewrite_block_calls(&mut f.body, &generic_fns, &needed, &mangled_name_for),
            Item::Impl(impl_block) => {
                for m in impl_block.methods.iter_mut() {
                    rewrite_block_calls(&mut m.body, &generic_fns, &needed, &mangled_name_for);
                }
            }
            _ => {}
        }
    }

    other_items.extend(monomorphized_fns);
    program.items = other_items;
    program
}

/// Produces a unique, readable C-safe function name for a monomorphized
/// instantiation, e.g. `identity` + `[int]` -> `identity__int`.
fn mangle_generic_name(base: &str, type_args: &[Type]) -> String {
    let suffix: String = type_args.iter()
        .map(|t| sanitize_type_name(t))
        .collect::<Vec<_>>()
        .join("_");
    format!("{}__{}", base, suffix)
}

fn sanitize_type_name(ty: &Type) -> String {
    match ty {
        Type::Int => "int".to_string(),
        Type::Float => "float".to_string(),
        Type::Bool => "bool".to_string(),
        Type::String => "string".to_string(),
        Type::Named(n) => n.clone(),
        other => other.to_string().replace(['<', '>', ',', ' '], "_"),
    }
}

/// Replaces every occurrence of a generic type parameter (Type::TypeParam)
/// in `ty` with its concrete substitution, recursively through compound
/// types (Option<T>, Result<T,E>, arrays, etc).
fn substitute_type(ty: &Type, subst: &HashMap<String, Type>) -> Type {
    match ty {
        Type::TypeParam(name) => subst.get(name).cloned().unwrap_or_else(|| ty.clone()),
        Type::Option(inner) => Type::Option(Box::new(substitute_type(inner, subst))),
        Type::Result(ok, err) => Type::Result(
            Box::new(substitute_type(ok, subst)),
            Box::new(substitute_type(err, subst)),
        ),
        Type::Array(inner, n) => Type::Array(Box::new(substitute_type(inner, subst)), *n),
        Type::Slice(inner) => Type::Slice(Box::new(substitute_type(inner, subst))),
        other => other.clone(),
    }
}

fn substitute_block(block: &mut Block, subst: &HashMap<String, Type>) {
    for stmt in block.stmts.iter_mut() {
        substitute_stmt(stmt, subst);
    }
    if let Some(expr) = block.expr.as_mut() {
        substitute_expr(expr, subst);
    }
}

fn substitute_stmt(stmt: &mut Stmt, subst: &HashMap<String, Type>) {
    match stmt {
        Stmt::Let { ty, init, .. } => {
            if let Some(t) = ty { *t = substitute_type(t, subst); }
            if let Some(e) = init { substitute_expr(e, subst); }
        }
        Stmt::Expr { expr, .. } => substitute_expr(expr, subst),
        Stmt::Assign { target, value, .. } => {
            substitute_expr(target, subst);
            substitute_expr(value, subst);
        }
        Stmt::Return { value, .. } => {
            if let Some(e) = value { substitute_expr(e, subst); }
        }
        Stmt::While { condition, body, .. } => {
            substitute_expr(condition, subst);
            substitute_block(body, subst);
        }
        Stmt::For { iter, body, .. } => {
            substitute_expr(iter, subst);
            substitute_block(body, subst);
        }
        Stmt::Loop { body, .. } => substitute_block(body, subst),
        Stmt::DoWhile { body, condition, .. } => {
            substitute_block(body, subst);
            substitute_expr(condition, subst);
        }
        _ => {}
    }
}

fn substitute_expr(expr: &mut Expr, subst: &HashMap<String, Type>) {
    match expr {
        Expr::Cast { expr, target_type, .. } => {
            substitute_expr(expr, subst);
            *target_type = substitute_type(target_type, subst);
        }
        Expr::Binary { left, right, .. } => {
            substitute_expr(left, subst);
            substitute_expr(right, subst);
        }
        Expr::Unary { expr, .. } => substitute_expr(expr, subst),
        Expr::Call { func, args, .. } => {
            substitute_expr(func, subst);
            for a in args.iter_mut() { substitute_expr(a, subst); }
        }
        Expr::MethodCall { receiver, args, .. } => {
            substitute_expr(receiver, subst);
            for a in args.iter_mut() { substitute_expr(a, subst); }
        }
        Expr::If { condition, then_block, else_block, .. } => {
            substitute_expr(condition, subst);
            substitute_block(then_block, subst);
            if let Some(b) = else_block { substitute_block(b, subst); }
        }
        Expr::Ternary { condition, then_expr, else_expr, .. } => {
            substitute_expr(condition, subst);
            substitute_expr(then_expr, subst);
            substitute_expr(else_expr, subst);
        }
        Expr::FieldAccess { expr, .. } => substitute_expr(expr, subst),
        Expr::Index { expr, index, .. } => {
            substitute_expr(expr, subst);
            substitute_expr(index, subst);
        }
        Expr::Array { elements, .. } => {
            for e in elements.iter_mut() { substitute_expr(e, subst); }
        }
        Expr::StructInit { fields, .. } => {
            for (_, e) in fields.iter_mut() { substitute_expr(e, subst); }
        }
        Expr::Some { value, .. } | Expr::Ok { value, .. } | Expr::Err { value, .. } => {
            substitute_expr(value, subst);
        }
        Expr::Try { expr, .. } => substitute_expr(expr, subst),
        _ => {}
    }
}

/// Scans a block (recursively) for calls to known generic functions,
/// inferring the concrete type argument(s) from the call's actual
/// argument expressions, and records (fn_name, concrete_type_args).
fn scan_block_for_generic_calls(block: &Block, generics: &HashMap<String, Function>, out: &mut Vec<(String, Vec<Type>)>) {
    let mut var_types: HashMap<String, Type> = HashMap::new();
    for stmt in &block.stmts {
        scan_stmt_for_generic_calls(stmt, generics, out, &mut var_types);
    }
    if let Some(expr) = &block.expr {
        scan_expr_for_generic_calls(expr, generics, out, &var_types);
    }
}

fn scan_stmt_for_generic_calls(stmt: &Stmt, generics: &HashMap<String, Function>, out: &mut Vec<(String, Vec<Type>)>, var_types: &mut HashMap<String, Type>) {
    match stmt {
        Stmt::Let { name, init: Some(e), .. } => {
            scan_expr_for_generic_calls(e, generics, out, var_types);
            if let Some(t) = literal_type_of(e, var_types) {
                var_types.insert(name.clone(), t);
            }
        }
        Stmt::Expr { expr, .. } => scan_expr_for_generic_calls(expr, generics, out, var_types),
        Stmt::Assign { target, value, .. } => {
            scan_expr_for_generic_calls(value, generics, out, var_types);
            if let Expr::Ident { name, .. } = target {
                if let Some(t) = literal_type_of(value, var_types) {
                    var_types.insert(name.clone(), t);
                }
            }
        }
        Stmt::Return { value: Some(e), .. } => scan_expr_for_generic_calls(e, generics, out, var_types),
        Stmt::While { condition, body, .. } => {
            scan_expr_for_generic_calls(condition, generics, out, var_types);
            scan_block_with_outer_vars(body, generics, out, var_types);
        }
        Stmt::For { iter, body, .. } => {
            scan_expr_for_generic_calls(iter, generics, out, var_types);
            scan_block_with_outer_vars(body, generics, out, var_types);
        }
        Stmt::Loop { body, .. } => scan_block_with_outer_vars(body, generics, out, var_types),
        Stmt::DoWhile { body, condition, .. } => {
            scan_block_with_outer_vars(body, generics, out, var_types);
            scan_expr_for_generic_calls(condition, generics, out, var_types);
        }
        _ => {}
    }
}

/// Scans a nested block (loop body etc), seeded with the enclosing
/// function's known variable types so far — loop bodies can reference
/// variables declared before the loop, and any new `let`s inside the
/// loop body are scoped to a local copy (not leaked back to the caller,
/// matching normal block-scoping semantics).
fn scan_block_with_outer_vars(block: &Block, generics: &HashMap<String, Function>, out: &mut Vec<(String, Vec<Type>)>, outer_vars: &HashMap<String, Type>) {
    let mut inner_vars = outer_vars.clone();
    for stmt in &block.stmts {
        scan_stmt_for_generic_calls(stmt, generics, out, &mut inner_vars);
    }
    if let Some(expr) = &block.expr {
        scan_expr_for_generic_calls(expr, generics, out, &inner_vars);
    }
}

fn scan_expr_for_generic_calls(expr: &Expr, generics: &HashMap<String, Function>, out: &mut Vec<(String, Vec<Type>)>, var_types: &HashMap<String, Type>) {
    match expr {
        Expr::Call { func, args, .. } => {
            if let Expr::Ident { name, .. } = func.as_ref() {
                if let Some(generic_fn) = generics.get(name) {
                    if let Some(type_args) = infer_type_args(generic_fn, args, var_types) {
                        out.push((name.clone(), type_args));
                    }
                }
            }
            for a in args { scan_expr_for_generic_calls(a, generics, out, var_types); }
        }
        Expr::Binary { left, right, .. } => {
            scan_expr_for_generic_calls(left, generics, out, var_types);
            scan_expr_for_generic_calls(right, generics, out, var_types);
        }
        Expr::Unary { expr, .. } => scan_expr_for_generic_calls(expr, generics, out, var_types),
        Expr::Cast { expr, .. } => scan_expr_for_generic_calls(expr, generics, out, var_types),
        Expr::MethodCall { receiver, args, .. } => {
            scan_expr_for_generic_calls(receiver, generics, out, var_types);
            for a in args { scan_expr_for_generic_calls(a, generics, out, var_types); }
        }
        Expr::If { condition, then_block, else_block, .. } => {
            scan_expr_for_generic_calls(condition, generics, out, var_types);
            scan_block_with_outer_vars(then_block, generics, out, var_types);
            if let Some(b) = else_block { scan_block_with_outer_vars(b, generics, out, var_types); }
        }
        Expr::Ternary { condition, then_expr, else_expr, .. } => {
            scan_expr_for_generic_calls(condition, generics, out, var_types);
            scan_expr_for_generic_calls(then_expr, generics, out, var_types);
            scan_expr_for_generic_calls(else_expr, generics, out, var_types);
        }
        Expr::FieldAccess { expr, .. } => scan_expr_for_generic_calls(expr, generics, out, var_types),
        Expr::Index { expr, index, .. } => {
            scan_expr_for_generic_calls(expr, generics, out, var_types);
            scan_expr_for_generic_calls(index, generics, out, var_types);
        }
        Expr::Array { elements, .. } => {
            for e in elements { scan_expr_for_generic_calls(e, generics, out, var_types); }
        }
        Expr::Some { value, .. } | Expr::Ok { value, .. } | Expr::Err { value, .. } => {
            scan_expr_for_generic_calls(value, generics, out, var_types);
        }
        Expr::Try { expr, .. } => scan_expr_for_generic_calls(expr, generics, out, var_types),
        Expr::NamedArg { value, .. } => scan_expr_for_generic_calls(value, generics, out, var_types),
        _ => {}
    }
}

/// Best-effort literal/shape-based type of an expression, for inferring
/// what concrete type a generic call's argument has. Deliberately
/// conservative — only literals are recognized; anything else (a plain
/// variable reference, a complex expression) returns None, meaning that
/// particular call site can't be monomorphized from this heuristic.
/// Best-effort type of an expression for inferring generic call-site type
/// arguments: a literal's obvious type, or — now — a variable reference
/// resolved against `var_types` (tracks `let`/assign bindings seen so far,
/// in source order, within the current function only). This mirrors the
/// same single-function-deep heuristic MIR's Python-style parameter
/// inference already uses; it's not a real cross-function type-inference
/// pass, but it covers the common case of `let n = 5; identity(n)`.
fn literal_type_of(expr: &Expr, var_types: &HashMap<String, Type>) -> Option<Type> {
    match expr {
        Expr::Literal { value: Literal::Int(_), .. } => Some(Type::Int),
        Expr::Literal { value: Literal::Float(_), .. } => Some(Type::Float),
        Expr::Literal { value: Literal::Bool(_), .. } => Some(Type::Bool),
        Expr::Literal { value: Literal::String(_), .. } => Some(Type::String),
        Expr::Ident { name, .. } => var_types.get(name).cloned(),
        _ => None,
    }
}

/// Infers concrete type arguments for a generic function's type
/// parameters, by matching each declared parameter's type against the
/// actual argument expression at the call site. Returns None if any type
/// parameter can't be inferred this way (the call site is then left
/// un-monomorphized and will surface as a normal "unknown function"
/// error from the typechecker, rather than silently doing nothing).
fn infer_type_args(generic_fn: &Function, call_args: &[Expr], var_types: &HashMap<String, Type>) -> Option<Vec<Type>> {
    let mut inferred: HashMap<String, Type> = HashMap::new();
    for (param, arg_expr) in generic_fn.params.iter().zip(call_args.iter()) {
        if let Type::TypeParam(name) = &param.ty {
            if !inferred.contains_key(name) {
                let arg_ty = literal_type_of(arg_expr, var_types)?;
                inferred.insert(name.clone(), arg_ty);
            }
        }
    }
    let mut result = Vec::with_capacity(generic_fn.type_params.len());
    for tp in &generic_fn.type_params {
        result.push(inferred.get(tp).cloned()?);
    }
    Some(result)
}

/// Rewrites every call to a generic function into a call to its
/// monomorphized concrete version, using the same type-inference logic
/// as the scan pass (so the rewrite always agrees with what was
/// generated).
fn rewrite_block_calls(
    block: &mut Block,
    generics: &HashMap<String, Function>,
    _needed: &[(String, Vec<Type>)],
    mangled_name_for: &HashMap<(String, Vec<Type>), String>,
) {
    let mut var_types: HashMap<String, Type> = HashMap::new();
    for stmt in block.stmts.iter_mut() {
        rewrite_stmt_calls(stmt, generics, mangled_name_for, &mut var_types);
    }
    if let Some(expr) = block.expr.as_mut() {
        rewrite_expr_calls(expr, generics, mangled_name_for, &var_types);
    }
}

fn rewrite_block_with_outer_vars(
    block: &mut Block,
    generics: &HashMap<String, Function>,
    mangled: &HashMap<(String, Vec<Type>), String>,
    outer_vars: &HashMap<String, Type>,
) {
    let mut inner_vars = outer_vars.clone();
    for stmt in block.stmts.iter_mut() {
        rewrite_stmt_calls(stmt, generics, mangled, &mut inner_vars);
    }
    if let Some(expr) = block.expr.as_mut() {
        rewrite_expr_calls(expr, generics, mangled, &inner_vars);
    }
}

fn rewrite_stmt_calls(stmt: &mut Stmt, generics: &HashMap<String, Function>, mangled: &HashMap<(String, Vec<Type>), String>, var_types: &mut HashMap<String, Type>) {
    match stmt {
        Stmt::Let { name, init: Some(e), .. } => {
            rewrite_expr_calls(e, generics, mangled, var_types);
            if let Some(t) = literal_type_of(e, var_types) {
                var_types.insert(name.clone(), t);
            }
        }
        Stmt::Expr { expr, .. } => rewrite_expr_calls(expr, generics, mangled, var_types),
        Stmt::Assign { target, value, .. } => {
            rewrite_expr_calls(value, generics, mangled, var_types);
            if let Expr::Ident { name, .. } = target {
                if let Some(t) = literal_type_of(value, var_types) {
                    var_types.insert(name.clone(), t);
                }
            }
        }
        Stmt::Return { value: Some(e), .. } => rewrite_expr_calls(e, generics, mangled, var_types),
        Stmt::While { condition, body, .. } => {
            rewrite_expr_calls(condition, generics, mangled, var_types);
            rewrite_block_with_outer_vars(body, generics, mangled, var_types);
        }
        Stmt::For { iter, body, .. } => {
            rewrite_expr_calls(iter, generics, mangled, var_types);
            rewrite_block_with_outer_vars(body, generics, mangled, var_types);
        }
        Stmt::Loop { body, .. } => rewrite_block_with_outer_vars(body, generics, mangled, var_types),
        Stmt::DoWhile { body, condition, .. } => {
            rewrite_block_with_outer_vars(body, generics, mangled, var_types);
            rewrite_expr_calls(condition, generics, mangled, var_types);
        }
        _ => {}
    }
}

fn rewrite_expr_calls(expr: &mut Expr, generics: &HashMap<String, Function>, mangled: &HashMap<(String, Vec<Type>), String>, var_types: &HashMap<String, Type>) {
    match expr {
        Expr::Call { func, args, .. } => {
            for a in args.iter_mut() { rewrite_expr_calls(a, generics, mangled, var_types); }
            if let Expr::Ident { name, .. } = func.as_mut() {
                if let Some(generic_fn) = generics.get(name.as_str()) {
                    if let Some(type_args) = infer_type_args(generic_fn, args, var_types) {
                        if let Some(new_name) = mangled.get(&(name.clone(), type_args)) {
                            *name = new_name.clone();
                        }
                    }
                }
            }
        }
        Expr::Binary { left, right, .. } => {
            rewrite_expr_calls(left, generics, mangled, var_types);
            rewrite_expr_calls(right, generics, mangled, var_types);
        }
        Expr::Unary { expr, .. } => rewrite_expr_calls(expr, generics, mangled, var_types),
        Expr::Cast { expr, .. } => rewrite_expr_calls(expr, generics, mangled, var_types),
        Expr::MethodCall { receiver, args, .. } => {
            rewrite_expr_calls(receiver, generics, mangled, var_types);
            for a in args.iter_mut() { rewrite_expr_calls(a, generics, mangled, var_types); }
        }
        Expr::If { condition, then_block, else_block, .. } => {
            rewrite_expr_calls(condition, generics, mangled, var_types);
            rewrite_block_with_outer_vars(then_block, generics, mangled, var_types);
            if let Some(b) = else_block { rewrite_block_with_outer_vars(b, generics, mangled, var_types); }
        }
        Expr::Ternary { condition, then_expr, else_expr, .. } => {
            rewrite_expr_calls(condition, generics, mangled, var_types);
            rewrite_expr_calls(then_expr, generics, mangled, var_types);
            rewrite_expr_calls(else_expr, generics, mangled, var_types);
        }
        Expr::FieldAccess { expr, .. } => rewrite_expr_calls(expr, generics, mangled, var_types),
        Expr::Index { expr, index, .. } => {
            rewrite_expr_calls(expr, generics, mangled, var_types);
            rewrite_expr_calls(index, generics, mangled, var_types);
        }
        Expr::Array { elements, .. } => {
            for e in elements.iter_mut() { rewrite_expr_calls(e, generics, mangled, var_types); }
        }
        Expr::Some { value, .. } | Expr::Ok { value, .. } | Expr::Err { value, .. } => {
            rewrite_expr_calls(value, generics, mangled, var_types);
        }
        Expr::Try { expr, .. } => rewrite_expr_calls(expr, generics, mangled, var_types),
        Expr::NamedArg { value, .. } => rewrite_expr_calls(value, generics, mangled, var_types),
        _ => {}
    }
}


// ── Generic struct monomorphization helpers ───────────────────────────

fn collect_struct_inits(
    item: &crate::ast::Item,
    gs: &HashMap<String, crate::ast::Struct>,
    seen: &mut std::collections::HashSet<String>,
    out: &mut Vec<crate::ast::Item>,
) {
    match item {
        crate::ast::Item::Function(f) => walk_block(&f.body, gs, seen, out),
        crate::ast::Item::Impl(imp) => {
            for m in &imp.methods { walk_block(&m.body, gs, seen, out); }
        }
        _ => {}
    }
}

fn walk_block(
    block: &crate::ast::Block,
    gs: &HashMap<String, crate::ast::Struct>,
    seen: &mut std::collections::HashSet<String>,
    out: &mut Vec<crate::ast::Item>,
) {
    for stmt in &block.stmts { walk_stmt(stmt, gs, seen, out); }
    if let Some(ref e) = block.expr { walk_expr(e, gs, seen, out); }
}

fn walk_stmt(
    stmt: &crate::ast::Stmt,
    gs: &HashMap<String, crate::ast::Struct>,
    seen: &mut std::collections::HashSet<String>,
    out: &mut Vec<crate::ast::Item>,
) {
    use crate::ast::Stmt;
    match stmt {
        Stmt::Let { init: Some(e), .. } | Stmt::Expr { expr: e, .. } |
        Stmt::Return { value: Some(e), .. } | Stmt::Assign { value: e, .. } =>
            walk_expr(e, gs, seen, out),
        Stmt::While { condition, body, .. } | Stmt::DoWhile { condition, body, .. } => {
            walk_expr(condition, gs, seen, out);
            walk_block(body, gs, seen, out);
        }
        Stmt::For { iter, body, .. } => {
            walk_expr(iter, gs, seen, out);
            walk_block(body, gs, seen, out);
        }
        Stmt::Loop { body, .. } => walk_block(body, gs, seen, out),
        _ => {}
    }
}

fn walk_expr(
    expr: &crate::ast::Expr,
    gs: &HashMap<String, crate::ast::Struct>,
    seen: &mut std::collections::HashSet<String>,
    out: &mut Vec<crate::ast::Item>,
) {
    use crate::ast::{Expr, Literal, Type};
    match expr {
        Expr::StructInit { name, fields, .. } => {
            if let Some(def) = gs.get(name) {
                // Infer concrete types from field expressions
                let types: Vec<Type> = fields.iter().map(|(_, e)| infer_type(e)).collect();
                if types.len() >= def.type_params.len() {
                    let suffix: String = types.iter().take(def.type_params.len())
                        .map(|t| type_suffix(t)).collect::<Vec<_>>().join("_");
                    let mangled = format!("{}_{}", name, suffix);
                    if seen.insert(mangled.clone()) {
                        let subst: HashMap<String, Type> = def.type_params.iter()
                            .cloned().zip(types.iter().cloned()).collect();
                        let concrete_fields: Vec<crate::ast::StructField> =
                            def.fields.iter().map(|f| crate::ast::StructField {
                                name: f.name.clone(),
                                ty: sub_type(&f.ty, &subst),
                                is_pub: f.is_pub,
                            }).collect();
                        out.push(crate::ast::Item::Struct(crate::ast::Struct {
                            id: 0, name: mangled,
                            type_params: vec![],
                            fields: concrete_fields,
                            is_pub: def.is_pub, span: def.span,
                        }));
                    }
                }
            }
            for (_, e) in fields { walk_expr(e, gs, seen, out); }
        }
        Expr::Call { func, args, .. } => {
            walk_expr(func, gs, seen, out);
            for a in args { walk_expr(a, gs, seen, out); }
        }
        Expr::Binary { left, right, .. } => {
            walk_expr(left, gs, seen, out); walk_expr(right, gs, seen, out);
        }
        Expr::FieldAccess { expr, .. } | Expr::Unary { expr, .. } |
        Expr::Try { expr, .. } | Expr::NamedArg { value: expr, .. } =>
            walk_expr(expr, gs, seen, out),
        Expr::If { condition, then_block, else_block, .. } => {
            walk_expr(condition, gs, seen, out);
            walk_block(then_block, gs, seen, out);
            if let Some(b) = else_block { walk_block(b, gs, seen, out); }
        }
        Expr::Array { elements, .. } => { for e in elements { walk_expr(e, gs, seen, out); } }
        _ => {}
    }
}

fn rewrite_struct_inits(
    item: &mut crate::ast::Item,
    gs: &HashMap<String, crate::ast::Struct>,
) {
    match item {
        crate::ast::Item::Function(f) => rw_block(&mut f.body, gs),
        crate::ast::Item::Impl(imp) => {
            for m in imp.methods.iter_mut() { rw_block(&mut m.body, gs); }
        }
        _ => {}
    }
}

fn rw_block(block: &mut crate::ast::Block, gs: &HashMap<String, crate::ast::Struct>) {
    for stmt in block.stmts.iter_mut() { rw_stmt(stmt, gs); }
    if let Some(ref mut e) = block.expr { rw_expr(e, gs); }
}

fn rw_stmt(stmt: &mut crate::ast::Stmt, gs: &HashMap<String, crate::ast::Struct>) {
    use crate::ast::Stmt;
    match stmt {
        Stmt::Let { init: Some(e), .. } | Stmt::Expr { expr: e, .. } |
        Stmt::Return { value: Some(e), .. } | Stmt::Assign { value: e, .. } => rw_expr(e, gs),
        Stmt::While { condition, body, .. } | Stmt::DoWhile { condition, body, .. } => {
            rw_expr(condition, gs); rw_block(body, gs);
        }
        Stmt::For { iter, body, .. } => { rw_expr(iter, gs); rw_block(body, gs); }
        Stmt::Loop { body, .. } => rw_block(body, gs),
        _ => {}
    }
}

fn rw_expr(expr: &mut crate::ast::Expr, gs: &HashMap<String, crate::ast::Struct>) {
    use crate::ast::{Expr, Type};
    match expr {
        Expr::StructInit { name, fields, .. } => {
            if let Some(def) = gs.get(name.as_str()) {
                let types: Vec<Type> = fields.iter().map(|(_, e)| infer_type(e)).collect();
                if types.len() >= def.type_params.len() {
                    let suffix: String = types.iter().take(def.type_params.len())
                        .map(|t| type_suffix(t)).collect::<Vec<_>>().join("_");
                    *name = format!("{}_{}", name, suffix);
                }
            }
            for (_, e) in fields.iter_mut() { rw_expr(e, gs); }
        }
        Expr::Call { func, args, .. } => {
            rw_expr(func, gs);
            for a in args.iter_mut() { rw_expr(a, gs); }
        }
        Expr::Binary { left, right, .. } => { rw_expr(left, gs); rw_expr(right, gs); }
        Expr::FieldAccess { expr, .. } | Expr::Unary { expr, .. } |
        Expr::Try { expr, .. } | Expr::NamedArg { value: expr, .. } => rw_expr(expr, gs),
        Expr::If { condition, then_block, else_block, .. } => {
            rw_expr(condition, gs);
            rw_block(then_block, gs);
            if let Some(b) = else_block { rw_block(b, gs); }
        }
        Expr::Array { elements, .. } => { for e in elements.iter_mut() { rw_expr(e, gs); } }
        _ => {}
    }
}

fn infer_type(expr: &crate::ast::Expr) -> crate::ast::Type {
    use crate::ast::{Expr, Literal, Type};
    match expr {
        Expr::Literal { value, .. } => match value {
            Literal::Int(_) => Type::Int,
            Literal::Float(_) => Type::Float,
            Literal::Bool(_) => Type::Bool,
            Literal::String(_) => Type::String,
            _ => Type::Int,
        },
        Expr::Cast { target_type, .. } => target_type.clone(),
        _ => Type::Int,
    }
}

fn type_suffix(t: &crate::ast::Type) -> String {
    use crate::ast::Type;
    match t {
        Type::Int => "int".to_string(),
        Type::Float => "float".to_string(),
        Type::Bool => "bool".to_string(),
        Type::String => "str".to_string(),
        Type::I64 => "i64".to_string(),
        Type::Named(n) => n.clone(),
        _ => "any".to_string(),
    }
}

fn sub_type(ty: &crate::ast::Type, subst: &HashMap<String, crate::ast::Type>) -> crate::ast::Type {
    match ty {
        crate::ast::Type::TypeParam(n) => subst.get(n).cloned().unwrap_or(ty.clone()),
        crate::ast::Type::Named(n) => subst.get(n).cloned().unwrap_or(ty.clone()),
        _ => ty.clone(),
    }
}
