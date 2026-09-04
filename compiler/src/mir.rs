// Quantum MIR - Mid-level Intermediate Representation
use crate::ast::*;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct MirProgram {
    pub functions: Vec<MirFunction>,
    pub globals: Vec<MirGlobal>,
    pub externs: Vec<crate::ast::ExternFunction>,
    pub structs: Vec<MirStructDef>,
    pub traits: Vec<MirTraitDef>,
    pub trait_impls: Vec<(String, String)>,
    /// If true, codegen wraps main() in a crash handler that catches
    /// panics and shows a beginner-friendly error message. Activated by
    /// placing `#[safe_mode]` at the top of any source file.
    pub safe_mode: bool,
}

#[derive(Debug, Clone)]
pub struct MirTraitDef {
    pub name: String,
    pub methods: Vec<MirTraitMethodSig>,
}

#[derive(Debug, Clone)]
pub struct MirTraitMethodSig {
    pub name: String,
    pub param_types: Vec<crate::ast::Type>,
    pub return_type: crate::ast::Type,
}

#[derive(Debug, Clone)]
pub struct MirStructDef {
    pub name: String,
    pub fields: Vec<(String, crate::ast::Type)>,
}

#[derive(Debug, Clone)]
pub struct MirFunction {
    pub name: String,
    pub params: Vec<MirLocal>,
    pub locals: Vec<MirLocal>,
    pub basic_blocks: Vec<MirBasicBlock>,
    pub return_type: Type,
}

#[derive(Debug, Clone)]
pub struct MirLocal {
    pub id: usize,
    pub name: String,
    pub ty: Type,
    pub mutable: bool,
}

#[derive(Debug, Clone)]
pub struct MirBasicBlock {
    pub id: usize,
    pub statements: Vec<MirStatement>,
    pub terminator: MirTerminator,
}

#[derive(Debug, Clone)]
pub enum MirStatement {
    Assign {
        place: MirPlace,
        rvalue: MirRvalue,
    },
    StorageLive(usize),
    StorageDead(usize),
}

#[derive(Debug, Clone)]
pub enum MirPlace {
    Local(usize),
    Deref(Box<MirPlace>),
    Field(Box<MirPlace>, String),
    Index(Box<MirPlace>, Box<MirOperand>),
}

#[derive(Debug, Clone)]
pub enum MirRvalue {
    Use(MirOperand),
    BinaryOp(BinOp, MirOperand, MirOperand),
    UnaryOp(UnOp, MirOperand),
    Ref(bool, MirPlace), // mutable, place
    Len(MirPlace),
    Cast(MirOperand, Type),
    Aggregate(Vec<MirOperand>),
    /// Struct construction — `Point { x: 10, y: 20 }`. Kept separate from
    /// Aggregate (used for array literals) since struct fields need their
    /// names for correct C designated-initializer syntax
    /// (`(Point){.x=10, .y=20}`), not just positional values.
    StructInit(String, Vec<(String, MirOperand)>),
    /// Wraps a concrete struct value in a vtable struct for dynamic
    /// dispatch — emitted when a struct is passed where a trait-typed
    /// parameter is expected (e.g. `Circle` passed as `Shape`).
    VTableConstruct {
        trait_name: String,
        struct_name: String,
        data_operand: MirOperand,
        method_names: Vec<String>,
    },
    MakeSome(MirOperand),
    MakeNone,
    MakeOk(MirOperand),
    MakeErr(MirOperand),
}

#[derive(Debug, Clone)]
pub enum MirOperand {
    Copy(MirPlace),
    Move(MirPlace),
    Constant(MirConstant),
}

#[derive(Debug, Clone)]
pub enum MirConstant {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Null,
}

#[derive(Debug, Clone)]
pub enum MirTerminator {
    Return(Option<MirOperand>),
    Goto(usize),
    If {
        condition: MirOperand,
        then_block: usize,
        else_block: usize,
    },
    Call {
        func: String,
        args: Vec<MirOperand>,
        destination: Option<MirPlace>,
        target: usize,
    },
    Unreachable,
}

#[derive(Debug, Clone)]
pub struct MirGlobal {
    pub name: String,
    pub ty: Type,
    pub value: MirConstant,
}

pub struct MirBuilder {
    current_function: Option<MirFunction>,
    local_counter: usize,
    block_counter: usize,
    locals_map: HashMap<String, usize>,
    /// Pre-computed return type for every function/extern in the program
    /// (including this one), built once before any function bodies are
    /// lowered. Lets `Expr::Call` lowering give the destination temp the
    /// correct C type instead of always defaulting to Inferred/int64_t —
    /// without this, a `float`-returning function's result would get
    /// silently truncated when stored in an int64_t local.
    function_return_types: HashMap<String, Type>,
    function_param_types: HashMap<String, Vec<Type>>,
    trait_method_names: HashMap<String, Vec<String>>,
    trait_names_set: std::collections::HashSet<String>,
    /// Maps struct name -> (field name -> field type), built once before
    /// any function bodies are lowered. Lets FieldAccess reads (`p.x`)
    /// give the destination temp the correct type instead of defaulting
    /// to Inferred/int64_t.
    struct_field_types: HashMap<String, HashMap<String, Type>>,
    /// Stack of (continue_target, break_target) block indices for
    /// currently-enclosing loops, innermost last. `break`/`continue`
    /// jump to the top entry; nested loops push/pop as they're entered/
    /// exited. Empty outside any loop (a `break`/`continue` there is a
    /// typechecker-level error, not handled here).
    loop_stack: Vec<(usize, usize)>,
}

impl MirBuilder {
    pub fn new() -> Self {
        Self {
            current_function: None,
            local_counter: 0,
            block_counter: 0,
            locals_map: HashMap::new(),
            function_return_types: HashMap::new(),
            function_param_types: HashMap::new(),
            trait_method_names: HashMap::new(),
            trait_names_set: std::collections::HashSet::new(),
            struct_field_types: HashMap::new(),
            loop_stack: Vec::new(),
        }
    }

    fn new_local(&mut self, name: String, ty: Type, mutable: bool) -> usize {
        let id = self.local_counter;
        self.local_counter += 1;
        self.locals_map.insert(name.clone(), id);

        if let Some(func) = &mut self.current_function {
            func.locals.push(MirLocal { id, name, ty, mutable });
        }

        id
    }

    fn new_block(&mut self) -> usize {
        let id = self.block_counter;
        self.block_counter += 1;

        if let Some(func) = &mut self.current_function {
            func.basic_blocks.push(MirBasicBlock {
                id,
                statements: Vec::new(),
                terminator: MirTerminator::Unreachable,
            });
        }

        id
    }

    fn lookup_local(&self, name: &str) -> Option<usize> {
        self.locals_map.get(name).copied()
    }
    fn lookup_local_type(&self, name: &str) -> Option<Type> {
        let id = self.locals_map.get(name).copied()?;
        self.current_function.as_ref()?.locals.iter()
            .find(|l| l.id == id)
            .map(|l| l.ty.clone())
    }
}

pub fn lower_to_mir(program: Program) -> Result<MirProgram, String> {
    let mut functions = Vec::new();
    let mut globals = Vec::new();
    let mut externs = Vec::new();
    let mut structs: Vec<MirStructDef> = Vec::new();

    // Pre-pass: collect every function/extern's return type up front so
    // Expr::Call lowering (which happens per-function, before all
    // functions are necessarily lowered) can look up the correct type for
    // the call's destination temp, instead of defaulting to Inferred.
    let mut return_types: HashMap<String, Type> = HashMap::new();
    // Memory wrappers return I64 (64-bit pointer values)

    // Int methods
    return_types.insert("q_int_abs".to_string(),       Type::I64);
    return_types.insert("q_int_to_float".to_string(),  Type::Float);
    return_types.insert("q_int_to_str".to_string(),    Type::String);
    return_types.insert("q_int_is_even".to_string(),   Type::Int);
    return_types.insert("q_int_is_odd".to_string(),    Type::Int);
    return_types.insert("q_int_pow".to_string(),        Type::I64);
    return_types.insert("q_int_min".to_string(),        Type::I64);
    return_types.insert("q_int_max".to_string(),        Type::I64);
    return_types.insert("q_int_clamp".to_string(),      Type::I64);
    return_types.insert("q_int_bit_count".to_string(), Type::Int);
    // Float methods
    return_types.insert("q_float_to_int".to_string(),      Type::I64);
    return_types.insert("q_float_to_str".to_string(),      Type::String);
    return_types.insert("q_float_is_nan".to_string(),      Type::Int);
    return_types.insert("q_float_is_infinite".to_string(), Type::Int);
    return_types.insert("q_float_is_finite".to_string(),   Type::Int);
    return_types.insert("q_float_is_positive".to_string(), Type::Int);
    return_types.insert("q_float_is_negative".to_string(), Type::Int);
    // Bool methods
    return_types.insert("q_bool_to_str".to_string(), Type::String);
    return_types.insert("q_bool_to_int".to_string(), Type::I64);
    return_types.insert("q_bool_and".to_string(),    Type::Int);
    return_types.insert("q_bool_or".to_string(),     Type::Int);
    return_types.insert("q_bool_not".to_string(),    Type::Int);
    // Char methods
    return_types.insert("q_char_to_int".to_string(),   Type::Int);
    return_types.insert("q_char_to_str".to_string(),   Type::String);
    return_types.insert("q_char_is_alpha".to_string(), Type::Int);
    return_types.insert("q_char_is_digit".to_string(), Type::Int);
    return_types.insert("q_char_is_upper".to_string(), Type::Int);
    return_types.insert("q_char_is_lower".to_string(), Type::Int);
    return_types.insert("q_char_is_space".to_string(), Type::Int);
    return_types.insert("q_char_upper".to_string(),    Type::Char);
    return_types.insert("q_char_lower".to_string(),    Type::Char);
    // Extra string methods
    return_types.insert("q_str_capitalize".to_string(), Type::String);
    return_types.insert("q_str_title".to_string(),      Type::String);
    return_types.insert("q_str_lstrip".to_string(),     Type::String);
    return_types.insert("q_str_rstrip".to_string(),     Type::String);
    return_types.insert("q_str_count".to_string(),      Type::Int);
    return_types.insert("q_str_center".to_string(),     Type::String);
    return_types.insert("q_str_zfill".to_string(),      Type::String);
    return_types.insert("q_str_find".to_string(),       Type::Int);
    return_types.insert("q_str_rfind".to_string(),      Type::Int);
    return_types.insert("q_str_eq".to_string(),         Type::Int);
    // String dot-methods
    return_types.insert("q_str_len".to_string(),         Type::Int);
    return_types.insert("q_str_contains".to_string(),    Type::Int);
    return_types.insert("q_str_starts_with".to_string(), Type::Int);
    return_types.insert("q_str_ends_with".to_string(),   Type::Int);
    return_types.insert("q_str_to_upper".to_string(),    Type::String);
    return_types.insert("q_str_to_lower".to_string(),    Type::String);
    return_types.insert("q_str_trim".to_string(),        Type::String);
    return_types.insert("q_str_replace".to_string(),     Type::String);
    return_types.insert("q_str_split_first".to_string(), Type::String);
    return_types.insert("q_str_split_last".to_string(),  Type::String);
    return_types.insert("q_str_index".to_string(),       Type::Int);
    return_types.insert("q_str_substr".to_string(),      Type::String);
    return_types.insert("q_str_repeat".to_string(),      Type::String);
    return_types.insert("q_str_parse_int".to_string(),   Type::I64);
    return_types.insert("q_str_parse_float".to_string(), Type::Float);
    return_types.insert("q_str_from_int".to_string(),    Type::String);
    return_types.insert("q_str_from_float".to_string(),  Type::String);
    // Number dot-methods
    return_types.insert("q_num_abs".to_string(),   Type::Float);
    return_types.insert("q_num_sqrt".to_string(),  Type::Float);
    return_types.insert("q_num_pow".to_string(),   Type::Float);
    return_types.insert("q_num_floor".to_string(), Type::Float);
    return_types.insert("q_num_ceil".to_string(),  Type::Float);
    return_types.insert("q_num_round".to_string(), Type::Float);
    return_types.insert("q_num_min".to_string(),   Type::Float);
    return_types.insert("q_num_max".to_string(),   Type::Float);
    return_types.insert("q_num_clamp".to_string(), Type::Float);
    return_types.insert("q_print_help".to_string(), Type::Int);
        return_types.insert("q_alloc".to_string(), Type::I64);
    return_types.insert("q_realloc".to_string(), Type::I64);
    return_types.insert("q_read_i64".to_string(), Type::I64);
    return_types.insert("q_read_f64".to_string(), Type::Float);
    return_types.insert("q_file_read_text".to_string(),   Type::String);
    return_types.insert("q_file_write_text".to_string(),  Type::Int);
    return_types.insert("q_file_append_text".to_string(), Type::Int);
    return_types.insert("q_file_exists".to_string(),      Type::Int);
    return_types.insert("q_file_line_count".to_string(),  Type::Int);
    return_types.insert("q_csv_read".to_string(),         Type::Int);
    return_types.insert("q_csv_read_col".to_string(),     Type::Int);

    // String methods
    return_types.insert("q_str_len".to_string(),         Type::Int);
    return_types.insert("q_str_contains".to_string(),    Type::Int);
    return_types.insert("q_str_starts_with".to_string(), Type::Int);
    return_types.insert("q_str_ends_with".to_string(),   Type::Int);
    return_types.insert("q_str_to_upper".to_string(),    Type::String);
    return_types.insert("q_str_to_lower".to_string(),    Type::String);
    return_types.insert("q_str_trim".to_string(),        Type::String);
    return_types.insert("q_str_replace".to_string(),     Type::String);
    return_types.insert("q_str_split_first".to_string(), Type::String);
    return_types.insert("q_str_split_last".to_string(),  Type::String);
    return_types.insert("q_str_index".to_string(),       Type::Int);
    return_types.insert("q_str_substr".to_string(),      Type::String);
    return_types.insert("q_str_parse_int".to_string(),   Type::I64);
    return_types.insert("q_str_parse_float".to_string(), Type::Float);
    return_types.insert("q_str_from_int".to_string(),    Type::String);
    return_types.insert("q_str_from_float".to_string(),  Type::String);
    return_types.insert("q_str_repeat".to_string(),      Type::String);
    return_types.insert("__q_to_string".to_string(), Type::String);
    return_types.insert("__q_to_string_f".to_string(), Type::String);
    return_types.insert("__q_strcat".to_string(), Type::String);
    return_types.insert("q_strmap_new".to_string(),      Type::I64);
    return_types.insert("q_strmap_get".to_string(),      Type::Int);
    return_types.insert("q_strmap_contains".to_string(), Type::Int);
    return_types.insert("q_strmap_len".to_string(),      Type::Int);
    let mut struct_field_types: HashMap<String, HashMap<String, Type>> = HashMap::new();
    let mut mir_trait_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut mir_trait_method_names: HashMap<String, Vec<String>> = HashMap::new();
    let mut mir_param_types: HashMap<String, Vec<Type>> = HashMap::new();
    let mut traits: Vec<MirTraitDef> = Vec::new();
    // Full AST trait definitions (including default method bodies),
    // collected so we can synthesize {Struct}_{method} functions for any
    // trait default a struct's impl block didn't override — this is what
    // makes trait defaults act as inherited behavior at the codegen level.
    let mut raw_trait_defs: HashMap<String, crate::ast::Trait> = HashMap::new();
    let mut trait_impls: Vec<(String, String)> = Vec::new();
    for item in &program.items {
        match item {
            Item::Function(func) => {
                let rt = func.return_type.clone().unwrap_or_else(|| {
                    if body_has_return_value(&func.body) { Type::Inferred } else { Type::Void }
                });
                return_types.insert(func.name.clone(), rt);
                mir_param_types.insert(func.name.clone(), func.params.iter().map(|p| p.ty.clone()).collect());
            }
            Item::ExternBlock(block) => {
                for ext_fn in &block.functions {
                    return_types.insert(
                        ext_fn.name.clone(),
                        ext_fn.return_type.clone().unwrap_or(Type::Void),
                    );
                }
            }
            Item::Struct(s) => {
                let fields: HashMap<String, Type> = s.fields.iter()
                    .map(|f| (f.name.clone(), f.ty.clone()))
                    .collect();
                struct_field_types.insert(s.name.clone(), fields);
            }
            Item::Trait(trait_def) => {
                mir_trait_names.insert(trait_def.name.clone());
                let method_names: Vec<String> = trait_def.methods.iter().map(|m| m.name.clone()).collect();
                mir_trait_method_names.insert(trait_def.name.clone(), method_names.clone());
                let mir_methods = trait_def.methods.iter().map(|m| MirTraitMethodSig {
                    name: m.name.clone(),
                    param_types: m.params.iter().skip(1).map(|p| p.ty.clone()).collect(),
                    return_type: m.return_type.clone().unwrap_or(Type::Void),
                }).collect();
                traits.push(MirTraitDef { name: trait_def.name.clone(), methods: mir_methods });
                raw_trait_defs.insert(trait_def.name.clone(), trait_def.clone());
            }
            Item::Impl(impl_block) => {
                // Record (struct, trait) pairs for vtable construction in codegen.
                if let Some(tn) = &impl_block.trait_name {
                    trait_impls.push((impl_block.type_name.clone(), tn.clone()));
                }
                // Register each method's return type under its mangled
                // name (e.g. `Point_area`) so the MethodCall lowering
                // can look it up via function_return_types.
                for method in &impl_block.methods {
                    let mangled = format!("{}_{}", impl_block.type_name, method.name);
                    let rt = method.return_type.clone().unwrap_or_else(|| {
                        if body_has_return_value(&method.body) { Type::Inferred } else { Type::Void }
                    });
                    return_types.insert(mangled, rt);
                }
            }
            _ => {}
        }
    }

    // Pre-pass: infer untyped (Inferred) parameter types from call-site
    // arguments. This matters most for Python-style code (`def greet(name):`
    // has no type annotation at all), where a parameter that's never given
    // an explicit type would otherwise default to Inferred -> int64_t in C,
    // even when every call site only ever passes it a string. Without this,
    // `greet("World")` would corrupt the string into q_int_to_str(garbage)
    // inside greet's body. This is necessarily a heuristic (Quantum doesn't
    // have a real cross-function type-inference pass), so it only handles
    // the common case: a direct call `fn_name(arg0, arg1, ...)` where an
    // argument is a literal whose type is unambiguous. Mismatched calls
    // (e.g. one call site passes a string, another passes an int) aren't
    // detected as an error — the first literal type found wins.
    let mut inferred_param_types: HashMap<String, Vec<Option<Type>>> = HashMap::new();

    /// Best-effort literal type of an expression, resolving simple
    /// variable references via `var_types` (tracks `let`/assign bindings
    /// to literals seen so far, in source order, within the current
    /// function only — this is a single-function-deep heuristic, not a
    /// real dataflow analysis).
    fn literal_type_of(expr: &Expr, var_types: &HashMap<String, Type>) -> Option<Type> {
        match expr {
            Expr::Literal { value: Literal::String(_), .. } => Some(Type::String),
            Expr::Literal { value: Literal::Int(_), .. } => Some(Type::Int),
            Expr::Literal { value: Literal::Float(_), .. } => Some(Type::Float),
            Expr::Literal { value: Literal::Bool(_), .. } => Some(Type::Bool),
            Expr::Ident { name, .. } => var_types.get(name).cloned(),
            _ => None,
        }
    }

    fn record_call(expr: &Expr, var_types: &HashMap<String, Type>, out: &mut HashMap<String, Vec<Option<Type>>>) {
        if let Expr::Call { func, args, .. } = expr {
            if let Expr::Ident { name, .. } = func.as_ref() {
                let entry = out.entry(name.clone()).or_insert_with(|| vec![None; args.len()]);
                if entry.len() < args.len() {
                    entry.resize(args.len(), None);
                }
                for (i, arg) in args.iter().enumerate() {
                    if entry[i].is_none() {
                        entry[i] = literal_type_of(arg, var_types);
                    }
                }
            }
        }
    }

    fn scan_stmt_for_calls(
        stmt: &Stmt,
        var_types: &mut HashMap<String, Type>,
        out: &mut HashMap<String, Vec<Option<Type>>>,
    ) {
        match stmt {
            Stmt::Expr { expr, .. } => record_call(expr, var_types, out),
            Stmt::Let { name, init: Some(expr), .. } => {
                record_call(expr, var_types, out);
                if let Some(t) = literal_type_of(expr, var_types) {
                    var_types.insert(name.clone(), t);
                }
            }
            Stmt::Assign { target, value, .. } => {
                record_call(value, var_types, out);
                if let Expr::Ident { name, .. } = target {
                    if let Some(t) = literal_type_of(value, var_types) {
                        var_types.insert(name.clone(), t);
                    }
                }
            }
            Stmt::Return { value: Some(expr), .. } => record_call(expr, var_types, out),
            Stmt::While { body, .. } | Stmt::For { body, .. } | Stmt::Loop { body, .. } | Stmt::DoWhile { body, .. } => {
                for s in &body.stmts { scan_stmt_for_calls(s, var_types, out); }
            }
            _ => {}
        }
    }

    for item in &program.items {
        if let Item::Function(func) = item {
            let mut var_types: HashMap<String, Type> = HashMap::new();
            // Seed with the function's own already-typed parameters, so a
            // typed/already-resolved param forwarded directly to another
            // call (`fn wrapper(s: string) { inner(s) }`) is tracked too.
            for p in &func.params {
                if !matches!(p.ty, Type::Inferred) {
                    var_types.insert(p.name.clone(), p.ty.clone());
                }
            }
            for stmt in &func.body.stmts {
                scan_stmt_for_calls(stmt, &mut var_types, &mut inferred_param_types);
            }
            // The function body's trailing expression (e.g. a bare
            // `greet("World")` as the only/last statement, which the
            // parser classifies as the block's trailing value rather than
            // a Stmt::Expr — see parse_block's trailing-expression
            // heuristic) also needs scanning; otherwise a call that
            // happens to be the last thing in a function body is silently
            // skipped, exactly the common case for short Python-style
            // functions.
            if let Some(expr) = &func.body.expr {
                record_call(expr, &var_types, &mut inferred_param_types);
            }
        }
    }

    // Register inherited trait-default methods' return types into
    // return_types BEFORE any function body is lowered — this must
    // happen here, not after the lowering loop, because main() (or any
    // other caller of c.describe()) needs to see Circle_describe's
    // return type when ITS OWN body gets lowered, which happens in the
    // very next loop below.
    for (struct_name, trait_name) in &trait_impls {
        if let Some(trait_def) = raw_trait_defs.get(trait_name) {
            for tm in &trait_def.methods {
                if tm.default_body.is_none() { continue; }
                let mangled = format!("{}_{}", struct_name, tm.name);
                // Only register if the struct's own impl didn't already
                // declare this method (which would have its own entry).
                return_types.entry(mangled).or_insert_with(|| tm.return_type.clone().unwrap_or(Type::Void));
            }
        }
    }

    for item in program.items {
        match item {
            Item::Function(mut func) => {
                // Apply any inferred parameter types found above, but only
                // for parameters that are still Inferred (an explicit
                // annotation always wins over a call-site guess).
                if let Some(arg_types) = inferred_param_types.get(&func.name) {
                    for (i, param) in func.params.iter_mut().enumerate() {
                        if matches!(param.ty, Type::Inferred) {
                            if let Some(Some(t)) = arg_types.get(i) {
                                param.ty = t.clone();
                            }
                        }
                    }
                }
                functions.push(lower_function(func, &return_types, &struct_field_types, &mir_trait_names, &mir_param_types, &mir_trait_method_names)?);
            }
            Item::Const(c) => {
                if let Some(constant) = expr_to_constant(&c.value) {
                    globals.push(MirGlobal {
                        name: c.name,
                        ty: c.ty,
                        value: constant,
                    });
                }
            }
            Item::ExternBlock(block) => {
                externs.extend(block.functions);
            }
            Item::Struct(s) => {
                let fields = s.fields.iter()
                    .map(|f| (f.name.clone(), f.ty.clone()))
                    .collect();
                let _ = &struct_field_types; // silence unused warning
                structs.push(MirStructDef { name: s.name.clone(), fields });
            }
            Item::Impl(impl_block) => {
                // Desugar methods into free functions. Each method
                //   `impl Point { fn area(self) -> int { ... } }`
                // becomes a function named `Point_area` with `self`
                // rewritten to an explicit `Point`-typed first param.
                // This lets all existing codegen work unchanged — a
                // method call `p.area()` simply becomes `Point_area(p)`
                // at the call site (handled in MethodCall lowering below).
                for mut method in impl_block.methods {
                    // Rename: `area` -> `Point_area`
                    let mangled_name = format!("{}_{}", impl_block.type_name, method.name);
                    method.name = mangled_name;

                    // Ensure the first parameter is typed as the struct.
                    // If the method wrote `fn area(self)` with Inferred
                    // type (or used `self: Point` explicitly), normalize
                    // both to `self: Point` so codegen sees the right type.
                    if let Some(first_param) = method.params.first_mut() {
                        if matches!(first_param.ty, Type::Inferred)
                            || first_param.name == "self"
                        {
                            first_param.ty = Type::Named(impl_block.type_name.clone());
                            first_param.name = "self_".to_string(); // avoid C `self` keyword
                        }
                    }

                    // Also fix any `self` references inside the body to
                    // use `self_` so the C output is valid. We do this
                    // by registering a call-site inference hint (the
                    // param rename handles the declaration; variable
                    // references in the body just use the local name,
                    // which already is `self_` after the param rename).
                    functions.push(lower_function(method, &return_types, &struct_field_types, &mir_trait_names, &mir_param_types, &mir_trait_method_names)?);
                }
            }
            _ => {}
        }
    }

    // Synthesize a real {Struct}_{method} function for every trait
    // default method a struct didn't override in its own impl block —
    // this is what makes a trait default act as inherited behavior at
    // runtime, not just a typechecker-level fiction. Each synthesized
    // function's body is the trait's default body, lowered with `self`
    // typed as the concrete struct (same as any other method).
    let existing_fn_names: std::collections::HashSet<String> =
        functions.iter().map(|f| f.name.clone()).collect();
    for (struct_name, trait_name) in &trait_impls {
        if let Some(trait_def) = raw_trait_defs.get(trait_name) {
            for tm in &trait_def.methods {
                let default_body = match &tm.default_body {
                    Some(b) => b,
                    None => continue, // no default — struct was required to provide its own
                };
                let mangled = format!("{}_{}", struct_name, tm.name);
                if existing_fn_names.contains(&mangled) {
                    continue; // struct overrode this method explicitly
                }
                // Build a synthetic Function from the trait's default,
                // with self typed as the concrete struct and the name
                // mangled to match what call sites expect.
                let mut synthetic_method = crate::ast::Function {
                    id: 0,
                    name: mangled.clone(),
                    type_params: vec![],
                    params: tm.params.clone(),
                    return_type: tm.return_type.clone(),
                    body: default_body.clone(),
                    is_pub: true,
                    is_async: false,
                    span: default_body.span,
                };
                if let Some(first) = synthetic_method.params.first_mut() {
                    if matches!(first.ty, Type::Inferred) {
                        first.ty = Type::Named(struct_name.clone());
                    }
                }
                functions.push(lower_function(synthetic_method, &return_types, &struct_field_types, &mir_trait_names, &mir_param_types, &mir_trait_method_names)?);
            }
        }
    }

    let safe_mode = program.safe_mode;
    Ok(MirProgram { functions, globals, externs, structs, traits, trait_impls, safe_mode })
}

/// Returns true if any statement in `block` (or nested blocks) is a
/// `return <expr>` with a non-unit value. Used to infer whether an
/// unannotated function has a return type.
fn body_has_return_value(block: &crate::ast::Block) -> bool {
    for stmt in &block.stmts {
        match stmt {
            Stmt::Return { value: Some(_), .. } => return true,
            Stmt::While { body, .. } | Stmt::For { body, .. } | Stmt::Loop { body, .. } | Stmt::DoWhile { body, .. } => {
                if body_has_return_value(body) { return true; }
            }
            _ => {}
        }
    }
    if let Some(expr) = &block.expr {
        if let Expr::If { then_block, else_block, .. } = expr.as_ref() {
            if body_has_return_value(then_block) { return true; }
            if let Some(eb) = else_block {
                if body_has_return_value(eb) { return true; }
            }
        }
    }
    false
}

fn lower_function(func: Function, return_types: &HashMap<String, Type>, struct_field_types: &HashMap<String, HashMap<String, Type>>, trait_names: &std::collections::HashSet<String>, param_types: &HashMap<String, Vec<Type>>, trait_method_names: &HashMap<String, Vec<String>>) -> Result<MirFunction, String> {
    // Resolve Named(X) -> TraitObject(X) when X is a known trait name
    let resolve_ty = |ty: &Type| -> Type {
        if let Type::Named(n) = ty {
            if trait_names.contains(n.as_str()) {
                return Type::TraitObject(n.clone());
            }
        }
        ty.clone()
    };

    let mut builder = MirBuilder::new();
    builder.function_return_types = return_types.clone();
    builder.struct_field_types = struct_field_types.clone();
    builder.function_param_types = param_types.iter()
        .map(|(k, v)| (k.clone(), v.iter().map(|ty| resolve_ty(ty)).collect()))
        .collect();
    builder.trait_method_names = trait_method_names.clone();
    builder.trait_names_set = trait_names.clone();

    // Create function
    // Reset builder state for this function
    builder.local_counter = 0;
    builder.block_counter = 0;
    builder.locals_map.clear();

    // When no return type annotation is given, try to infer from `return`
    // statements in the body. Default to Inferred (→ int64_t in C) rather
    // than Void so untyped functions like Python's `def add(a, b): return a+b`
    // work without annotation.
    let return_type = func.return_type.clone().unwrap_or_else(|| {
        // Quick scan: does the body have any return statements with values?
        if body_has_return_value(&func.body) {
            Type::Inferred
        } else {
            Type::Void
        }
    });
    let mir_func = MirFunction {
        name: func.name.clone(),
        params: Vec::new(),
        locals: Vec::new(),
        basic_blocks: Vec::new(),
        return_type,
    };
    builder.current_function = Some(mir_func);

    // Register parameters as locals (before body so they resolve correctly)
    let mut param_ids = Vec::new();
    for param in &func.params {
        let id = builder.new_local(param.name.clone(), resolve_ty(&param.ty), param.mutable);
        param_ids.push(id);
    }

    // Create entry block
    let entry_block = builder.new_block();

    // Lower function body
    lower_block(&mut builder, &func.body, entry_block)?;

    // Retrieve the fully built function
    let mut result = builder.current_function.take()
        .ok_or_else(|| "Failed to build function".to_string())?;

    // Populate params from the registered locals
    result.params = param_ids.iter().zip(&func.params).map(|(&id, param)| {
        MirLocal { id, name: param.name.clone(), ty: resolve_ty(&param.ty), mutable: param.mutable }
    }).collect();

    Ok(result)
}

/// Lower a single `if`/`ternary` branch block (then or else), redirecting
/// its trailing value into `result_temp` and jumping to `merge_bb`, instead
/// of letting `lower_block`'s normal trailing-expression handling turn it
/// into a genuine function-level `Return`. This is what makes `if`-as-an-
/// expression (and the ternary operator, which desugars to this) actually
/// carry a value out to the enclosing `let`/expression context, rather than
/// silently returning from the whole function and discarding the value.
///
/// Statements before the trailing expression are lowered normally via the
/// existing `lower_stmt` loop (copied here rather than calling `lower_block`
/// directly, since we need to intercept only the final trailing-expression
/// step).
fn lower_then_or_else_block(
    builder: &mut MirBuilder,
    block: &Block,
    start_block: usize,
    result_temp: usize,
    merge_bb: usize,
) -> Result<usize, String> {
    let mut current_block = start_block;
    for stmt in &block.stmts {
        lower_stmt(builder, stmt, &mut current_block)?;
    }

    if let Some(expr) = &block.expr {
        // Has a trailing value (e.g. the `100` in `if cond { 100 }`) —
        // evaluate it and assign into the shared result temp.
        let operand = lower_expr_to_operand(builder, expr, &mut current_block)?;
        if let Some(func) = &mut builder.current_function {
            if let Some(bb) = func.basic_blocks.get_mut(current_block) {
                bb.statements.push(MirStatement::Assign {
                    place: MirPlace::Local(result_temp),
                    rvalue: MirRvalue::Use(operand),
                });
                if matches!(bb.terminator, MirTerminator::Unreachable | MirTerminator::Return(_)) {
                    bb.terminator = MirTerminator::Goto(merge_bb);
                }
            }
        }
    } else {
        // No trailing value (e.g. `if cond { println("x") }` used as a
        // statement, not an expression) — nothing to assign, just continue
        // to merge_bb.
        if let Some(func) = &mut builder.current_function {
            if let Some(bb) = func.basic_blocks.get_mut(current_block) {
                if matches!(bb.terminator, MirTerminator::Unreachable | MirTerminator::Return(None)) {
                    bb.terminator = MirTerminator::Goto(merge_bb);
                }
            }
        }
    }

    Ok(current_block)
}

fn lower_block(
    builder: &mut MirBuilder,
    block: &Block,
    start_block: usize,
) -> Result<usize, String> {
    lower_block_inner(builder, block, start_block, false)
}

/// Lowers a block used as a loop body (while/for/loop). The key
/// difference from a function-body block: a trailing expression here is
/// evaluated only for its side effects and then falls through to the
/// loop's back-edge — it must NOT become a `Return`, even if the
/// expression happens to produce a value (e.g. an `if` without an else,
/// used as the last statement in the loop body, like `if i >= 3 { break }`).
/// Without this distinction, lower_block's normal "trailing expression
/// becomes the function's return value" behavior incorrectly turns the
/// loop body's natural fallthrough into a real `return`, orphaning the
/// loop-back edge entirely.
fn lower_loop_body(
    builder: &mut MirBuilder,
    block: &Block,
    start_block: usize,
) -> Result<usize, String> {
    lower_block_inner(builder, block, start_block, true)
}

fn lower_block_inner(
    builder: &mut MirBuilder,
    block: &Block,
    start_block: usize,
    is_loop_body: bool,
) -> Result<usize, String> {
    let mut current_block = start_block;
    for stmt in &block.stmts {
        lower_stmt(builder, stmt, &mut current_block)?;
    }

    if let Some(expr) = &block.expr {
        let operand = lower_expr_to_operand(builder, expr, &mut current_block)?;

        if let Some(func) = &mut builder.current_function {
            if let Some(bb) = func.basic_blocks.get_mut(current_block) {
                if is_loop_body {
                    // Evaluate for side effects only; leave the
                    // terminator as whatever the expression's own
                    // lowering set (e.g. an if/break already set Goto to
                    // the break target — don't clobber that). Only
                    // default to Unreachable (so the loop-back-edge
                    // override in While/For/Loop's caller can detect
                    // "fell through naturally" and wire up the back edge)
                    // if nothing else already terminated this block.
                    if matches!(bb.terminator, MirTerminator::Unreachable) {
                        // leave as Unreachable — the caller (Stmt::While /
                        // Stmt::For / Stmt::Loop) checks for exactly this
                        // and overrides it with the loop-back Goto.
                    }
                } else {
                    bb.terminator = MirTerminator::Return(Some(operand));
                }
            }
        }
    } else {
        if let Some(func) = &mut builder.current_function {
            if let Some(bb) = func.basic_blocks.get_mut(current_block) {
                if matches!(bb.terminator, MirTerminator::Unreachable) {
                    if is_loop_body {
                        // leave as Unreachable for the loop-back-edge
                        // override, same reasoning as above.
                    } else {
                        bb.terminator = MirTerminator::Return(None);
                    }
                }
            }
        }
    }

    Ok(current_block)
}

fn lower_stmt(
    builder: &mut MirBuilder,
    stmt: &Stmt,
    current_block: &mut usize,
) -> Result<(), String> {
    match stmt {
        Stmt::Let { name, ty, init, mutable, .. } => {
            // If no explicit type annotation, try to infer a concrete type from
            // a simple literal initializer (int/float/bool/string). This avoids
            // declaring e.g. `let pi = 3.14159` as int64_t (Type::Inferred's C
            // representation), which would truncate the float and break
            // `.to_string()` type dispatch.
            let inferred_from_literal = init.as_ref().and_then(|e| match e {
                Expr::Literal { value: Literal::Int(_), .. } => Some(Type::Int),
                Expr::Literal { value: Literal::Float(_), .. } => Some(Type::Float),
                Expr::Literal { value: Literal::Bool(_), .. } => Some(Type::Bool),
                Expr::Literal { value: Literal::String(_), .. } => Some(Type::String),
                // Array literal — infer element type from the first element
                // (mixed-type arrays aren't supported) and length from the
                // literal's element count, e.g. `[1,2,3,4,5]` → Array(Int, 5).
                Expr::Array { elements, .. } => {
                    let elem_ty = elements.first().and_then(|first| match first {
                        Expr::Literal { value: Literal::Int(_), .. } => Some(Type::Int),
                        Expr::Literal { value: Literal::Float(_), .. } => Some(Type::Float),
                        Expr::Literal { value: Literal::Bool(_), .. } => Some(Type::Bool),
                        Expr::Literal { value: Literal::String(_), .. } => Some(Type::String),
                        _ => None,
                    }).unwrap_or(Type::Int);
                    Some(Type::Array(Box::new(elem_ty), Some(elements.len())))
                }
                // A function/extern call — look up its known return type so
                // `let p = get_pi()` gets `double` instead of defaulting to
                // int64_t and truncating the result.
                Expr::Call { func, .. } => {
                    if let Expr::Ident { name, .. } = func.as_ref() {
                        builder.function_return_types.get(name).cloned()
                            .filter(|t| !matches!(t, Type::Inferred | Type::Void))
                    } else {
                        None
                    }
                }
                // Plain variable reference — propagate the referenced
                // variable's declared type, so `let mut result = s` (where
                // `s` is a known String parameter) correctly makes `result`
                // a String too, instead of defaulting to Inferred/int64_t.
                // Without this, later `.len()` calls on `result` couldn't
                // tell it should use strlen() instead of sizeof-arithmetic.
                Expr::Ident { name, .. } => {
                    builder.lookup_local(name)
                        .and_then(|id| builder.current_function.as_ref()
                            .and_then(|f| f.locals.iter().find(|l| l.id == id)))
                        .map(|l| l.ty.clone())
                        .filter(|t| !matches!(t, Type::Inferred))
                }
                // `x as Type` — the cast's target type IS the result type,
                // by definition. Without this, `let b = a as float` would
                // fall through to Inferred/int64_t and immediately
                // truncate the just-cast float right back to an int.
                Expr::Cast { target_type, .. } => Some(target_type.clone()),
                Expr::StructInit { name, .. } => Some(Type::Named(name.clone())),
                // Binary expression — if either side is recognizably
                // float-producing, the whole expression is float too.
                // Covers the common `(a as float) / (b as float)` pattern,
                // where the top-level expression isn't itself a Cast but
                // its operands are.
                Expr::Binary { op, left, right, .. } => {
                    if binop_produces_bool(*op) {
                        Some(Type::Bool)
                    } else if expr_is_float_typed(builder, left) || expr_is_float_typed(builder, right) {
                        Some(Type::Float)
                    } else {
                        None
                    }
                }
                Expr::Unary { op: UnOp::Not, .. } => Some(Type::Bool),
                _ => None,
            });
            let var_type = ty.clone()
                .or(inferred_from_literal)
                .unwrap_or(Type::Inferred);
            let local_id = builder.new_local(name.clone(), var_type.clone(), *mutable);

            if let Some(init_expr) = init {
                let rvalue = lower_expr_to_rvalue(builder, init_expr, current_block)?;

                if let Some(func) = &mut builder.current_function {
                    if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                        bb.statements.push(MirStatement::Assign {
                            place: MirPlace::Local(local_id),
                            rvalue,
                        });
                    }
                }
            }

            Ok(())
        }

        Stmt::Assign { target, value, .. } => {
            // Python-style implicit declaration: if the assignment target is
            // a bare identifier that isn't in scope yet, auto-introduce it as
            // a mutable local (equivalent to `let mut name = value`).
            // This lets users write `x = 42` instead of `let x = 42`.
            if let Expr::Ident { name, .. } = target {
                if builder.lookup_local(name).is_none() {
                    // Infer type from the RHS literal (or call return type) if possible
                    let inferred_ty = match value {
                        Expr::Literal { value: crate::ast::Literal::Int(_), .. } => crate::ast::Type::Int,
                        Expr::Literal { value: crate::ast::Literal::Float(_), .. } => crate::ast::Type::Float,
                        Expr::Literal { value: crate::ast::Literal::Bool(_), .. } => crate::ast::Type::Bool,
                        Expr::Literal { value: crate::ast::Literal::String(_), .. } => crate::ast::Type::String,
                        Expr::Call { func, .. } => {
                            if let Expr::Ident { name, .. } = func.as_ref() {
                                builder.function_return_types.get(name).cloned()
                                    .filter(|t| !matches!(t, crate::ast::Type::Inferred | crate::ast::Type::Void))
                                    .unwrap_or(crate::ast::Type::Inferred)
                            } else {
                                crate::ast::Type::Inferred
                            }
                        }
                        // Propagate type from another known variable
                        // (e.g. `result = s` where `s` is a String param).
                        Expr::Ident { name: ref_name, .. } => {
                            builder.lookup_local(ref_name)
                                .and_then(|id| builder.current_function.as_ref()
                                    .and_then(|f| f.locals.iter().find(|l| l.id == id)))
                                .map(|l| l.ty.clone())
                                .filter(|t| !matches!(t, crate::ast::Type::Inferred))
                                .unwrap_or(crate::ast::Type::Inferred)
                        }
                        // `x as Type` — same reasoning as the Stmt::Let case.
                        Expr::Cast { target_type, .. } => target_type.clone(),
                        Expr::Binary { op, left, right, .. } => {
                            if binop_produces_bool(*op) {
                                crate::ast::Type::Bool
                            } else if expr_is_float_typed(builder, left) || expr_is_float_typed(builder, right) {
                                crate::ast::Type::Float
                            } else {
                                crate::ast::Type::Inferred
                            }
                        }
                        Expr::Unary { op: UnOp::Not, .. } => crate::ast::Type::Bool,
                        _ => crate::ast::Type::Inferred,
                    };
                    let local_id = builder.new_local(name.clone(), inferred_ty, true);
                    let rvalue = lower_expr_to_rvalue(builder, value, current_block)?;
                    if let Some(func) = &mut builder.current_function {
                        if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                            bb.statements.push(MirStatement::Assign {
                                place: MirPlace::Local(local_id),
                                rvalue,
                            });
                        }
                    }
                    return Ok(());
                }
            }

            let place = lower_expr_to_place(builder, target, current_block)?;
            let rvalue = lower_expr_to_rvalue(builder, value, current_block)?;

            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                    bb.statements.push(MirStatement::Assign { place, rvalue });
                }
            }

            Ok(())
        }

        Stmt::Expr { expr, .. } => {
            // Evaluate expression for side effects
            lower_expr_to_operand(builder, expr, current_block)?;
            Ok(())
        }

        Stmt::Return { value, .. } => {
            let operand = if let Some(expr) = value {
                Some(lower_expr_to_operand(builder, expr, current_block)?)
            } else {
                None
            };

            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                    bb.terminator = MirTerminator::Return(operand);
                }
            }

            Ok(())
        }

        Stmt::While { condition, body, .. } => {
            let cond_block = builder.new_block();
            let body_block = builder.new_block();
            let exit_block = builder.new_block();

            // Jump to condition
            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                    bb.terminator = MirTerminator::Goto(cond_block);
                }
            }

            // Condition
            let mut cond_block_mut = cond_block;
            let cond_operand = lower_expr_to_operand(builder, condition, &mut cond_block_mut)?;
            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(cond_block_mut) {
                    bb.terminator = MirTerminator::If {
                        condition: cond_operand,
                        then_block: body_block,
                        else_block: exit_block,
                    };
                }
            }

            // `continue` jumps back to cond_block (re-checks the loop
            // condition); `break` jumps to exit_block. Pushed before
            // lowering the body so any break/continue inside (including
            // inside nested if/else, but NOT inside a nested loop, which
            // pushes its own targets) resolves to this loop's blocks.
            builder.loop_stack.push((cond_block, exit_block));
            let body_final = lower_loop_body(builder, body, body_block)?;
            builder.loop_stack.pop();

            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(body_final) {
                    if matches!(bb.terminator, MirTerminator::Unreachable | MirTerminator::Return(None)) {
                        bb.terminator = MirTerminator::Goto(cond_block);
                    }
                }
            }

            *current_block = exit_block;

            Ok(())
        }

        Stmt::DoWhile { body, condition, .. } => {
            // `do { body } while cond` — same block shape as `while`, but
            // execution jumps straight into body_block first (skipping
            // the initial condition check entirely), and only reaches
            // cond_block after the body's first run. This guarantees the
            // body always executes at least once, which is the entire
            // point of do-while versus a plain while loop.
            let body_block = builder.new_block();
            let cond_block = builder.new_block();
            let exit_block = builder.new_block();

            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                    bb.terminator = MirTerminator::Goto(body_block);
                }
            }

            // `continue` jumps to cond_block (re-checks the condition,
            // same as while); `break` jumps to exit_block.
            builder.loop_stack.push((cond_block, exit_block));
            let body_final = lower_loop_body(builder, body, body_block)?;
            builder.loop_stack.pop();

            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(body_final) {
                    if matches!(bb.terminator, MirTerminator::Unreachable | MirTerminator::Return(None)) {
                        bb.terminator = MirTerminator::Goto(cond_block);
                    }
                }
            }

            // Condition is checked AFTER the body, at cond_block — if
            // true, loop back to body_block; if false, fall through to
            // exit_block. This is the only structural difference from a
            // plain while loop.
            let mut cond_block_mut = cond_block;
            let cond_operand = lower_expr_to_operand(builder, condition, &mut cond_block_mut)?;
            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(cond_block_mut) {
                    bb.terminator = MirTerminator::If {
                        condition: cond_operand,
                        then_block: body_block,
                        else_block: exit_block,
                    };
                }
            }

            *current_block = exit_block;

            Ok(())
        }

        Stmt::For { var, iter, body, span } => {
            // Desugar `for i in start..end { body }` (or `..=`) into:
            //   let mut i = start;
            //   while i < end /* or <= for inclusive */ {
            //       body
            //       i = i + 1
            //   }
            // Only range iteration is currently supported.
            let (start_expr, end_expr, inclusive) = match iter {
                Expr::Binary { op: BinOp::Range, left, right, .. } => {
                    (left.as_ref().clone(), right.as_ref().clone(), false)
                }
                Expr::Binary { op: BinOp::RangeInclusive, left, right, .. } => {
                    (left.as_ref().clone(), right.as_ref().clone(), true)
                }
                Expr::Range { start, end, inclusive, .. } => {
                    (start.as_ref().clone(), end.as_ref().clone(), *inclusive)
                }
                _ => {
                    // Unsupported iterable (arrays, etc.) — skip for now
                    return Ok(());
                }
            };

            // 1. let mut <var> = start
            let loop_var = builder.new_local(var.clone(), Type::Int, true);
            let start_rvalue = lower_expr_to_rvalue(builder, &start_expr, current_block)?;
            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                    bb.statements.push(MirStatement::Assign {
                        place: MirPlace::Local(loop_var),
                        rvalue: start_rvalue,
                    });
                }
            }

            // 2. Set up condition / body / increment / exit blocks.
            // `inc_block` exists as a separate step (not just appended
            // after the body) so `continue` can jump there directly —
            // jumping straight to cond_block instead would skip
            // incrementing the loop variable and infinite-loop.
            let cond_block = builder.new_block();
            let body_block = builder.new_block();
            let inc_block = builder.new_block();
            let exit_block = builder.new_block();

            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                    bb.terminator = MirTerminator::Goto(cond_block);
                }
            }

            // 3. Condition: i < end  (or i <= end if inclusive)
            let mut cond_block_mut = cond_block;
            let end_operand = lower_expr_to_operand(builder, &end_expr, &mut cond_block_mut)?;
            let cmp_op = if inclusive { BinOp::Le } else { BinOp::Lt };
            let cond_temp = builder.new_local(format!("__for_cond_{}", loop_var), Type::Bool, false);
            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(cond_block_mut) {
                    bb.statements.push(MirStatement::Assign {
                        place: MirPlace::Local(cond_temp),
                        rvalue: MirRvalue::BinaryOp(cmp_op, MirOperand::Copy(MirPlace::Local(loop_var)), end_operand),
                    });
                    bb.terminator = MirTerminator::If {
                        condition: MirOperand::Copy(MirPlace::Local(cond_temp)),
                        then_block: body_block,
                        else_block: exit_block,
                    };
                }
            }

            // 4. Body — `continue` targets inc_block (not cond_block
            // directly), `break` targets exit_block.
            builder.loop_stack.push((inc_block, exit_block));
            let body_end_block = lower_loop_body(builder, body, body_block)?;
            builder.loop_stack.pop();

            // Body's natural fallthrough (no break/continue/return hit)
            // also goes to inc_block.
            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(body_end_block) {
                    if matches!(bb.terminator, MirTerminator::Unreachable | MirTerminator::Return(None)) {
                        bb.terminator = MirTerminator::Goto(inc_block);
                    }
                }
            }

            // 4b. inc_block: increment i, then loop back to cond_block.
            // This is now its own block (reached via fallthrough from the
            // body OR directly via `continue`) rather than appended onto
            // whichever block the body happened to end in.
            let inc_temp = builder.new_local(format!("__for_inc_{}", loop_var), Type::Int, false);
            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(inc_block) {
                    bb.statements.push(MirStatement::Assign {
                        place: MirPlace::Local(inc_temp),
                        rvalue: MirRvalue::BinaryOp(
                            BinOp::Add,
                            MirOperand::Copy(MirPlace::Local(loop_var)),
                            MirOperand::Constant(MirConstant::Int(1)),
                        ),
                    });
                    bb.statements.push(MirStatement::Assign {
                        place: MirPlace::Local(loop_var),
                        rvalue: MirRvalue::Use(MirOperand::Copy(MirPlace::Local(inc_temp))),
                    });
                    bb.terminator = MirTerminator::Goto(cond_block);
                }
            }

            // 5. Continue lowering subsequent statements in exit_block
            *current_block = exit_block;

            Ok(())
        }

        Stmt::Loop { body, .. } => {
            // `loop { ... }` — an unconditional, infinite loop. Desugars
            // to the same block structure as `while true { ... }`, but
            // without ever needing to lower a condition expression.
            let body_block = builder.new_block();
            let exit_block = builder.new_block();

            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                    bb.terminator = MirTerminator::Goto(body_block);
                }
            }

            // `continue` jumps back to the top of the body (there's no
            // separate condition block to re-check, since `loop` has no
            // condition); `break` jumps to exit_block.
            builder.loop_stack.push((body_block, exit_block));
            let body_final = lower_loop_body(builder, body, body_block)?;
            builder.loop_stack.pop();

            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(body_final) {
                    if matches!(bb.terminator, MirTerminator::Unreachable | MirTerminator::Return(None)) {
                        bb.terminator = MirTerminator::Goto(body_block);
                    }
                }
            }

            *current_block = exit_block;

            Ok(())
        }

        Stmt::Break { .. } => {
            // Jump to the innermost enclosing loop's exit block. A
            // `break` outside any loop is a typechecker-level error (not
            // currently enforced — see the README's known-limitations
            // section), so this silently does nothing if loop_stack is
            // empty rather than panicking.
            if let Some(&(_, break_target)) = builder.loop_stack.last() {
                if let Some(func) = &mut builder.current_function {
                    if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                        bb.terminator = MirTerminator::Goto(break_target);
                    }
                }
                // Start a fresh, unreachable block for any statements that
                // textually follow the break in the same block (dead code,
                // but still needs somewhere to lower into without
                // corrupting the break's own terminator).
                *current_block = builder.new_block();
            }
            Ok(())
        }

        Stmt::Continue { .. } => {
            // Jump to the innermost enclosing loop's continue target
            // (the condition-check block for while/loop, or the
            // increment block for for-loops — see the loop_stack push
            // sites above for which one each loop kind uses).
            if let Some(&(continue_target, _)) = builder.loop_stack.last() {
                if let Some(func) = &mut builder.current_function {
                    if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                        bb.terminator = MirTerminator::Goto(continue_target);
                    }
                }
                *current_block = builder.new_block();
            }
            Ok(())
        }
    }
}

/// Best-effort check for whether an expression is recognizably
/// float-typed: a float literal, a cast to a float type, or a reference to
/// a variable already known to be declared Float/F32/F64. This is a
/// pattern-matching heuristic, not a real type lookup (MIR lowering runs
/// independently of the typechecker and doesn't have access to its
/// resolved types — see the longer note where this is used), but it's
/// enough to catch the common cases that previously caused float
/// expressions to be silently truncated back to int64_t: a `let`
/// initializer, an implicit-assign target, or — the case this helper was
/// specifically added for — a `return` expression's destination temp,
/// none of which have type annotations to fall back on.
/// Operators that always produce a Bool result, regardless of operand
/// types — comparisons and logical and/or. Used to give the destination
/// temp the correct type at every Binary-expression lowering site, the
/// same architectural gap that previously caused float results to be
/// truncated (see expr_is_float_typed above): without this, `a && b`'s
/// result temp defaulted to Inferred/int64_t, which happened to print the
/// right 0/1 value but dispatched to the wrong println variant (the
/// generic string-ish path instead of println_bool), showing a raw 0/1
/// instead of true/false.
fn binop_produces_bool(op: BinOp) -> bool {
    matches!(op, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
        | BinOp::And | BinOp::Or)
}

fn expr_is_float_typed(builder: &MirBuilder, expr: &Expr) -> bool {
    match expr {
        Expr::Literal { value: Literal::Float(_), .. } => true,
        Expr::Cast { target_type, .. } => matches!(target_type, Type::Float | Type::F32 | Type::F64),
        Expr::Ident { name, .. } => {
            builder.lookup_local(name)
                .and_then(|id| builder.current_function.as_ref()
                    .and_then(|f| f.locals.iter().find(|l| l.id == id)))
                .map(|l| matches!(l.ty, Type::Float | Type::F32 | Type::F64))
                .unwrap_or(false)
        }
        _ => false,
    }
}

fn lower_expr_to_operand(
    builder: &mut MirBuilder,
    expr: &Expr,
    current_block: &mut usize,
) -> Result<MirOperand, String> {
    match expr {
        Expr::Literal { value, .. } => {
            Ok(MirOperand::Constant(literal_to_constant(value)))
        }

        Expr::Ident { name, .. } => {
            if let Some(local_id) = builder.lookup_local(name) {
                Ok(MirOperand::Copy(MirPlace::Local(local_id)))
            } else {
                Err(format!("Undefined variable: {}", name))
            }
        }

        Expr::Binary { op, left, right, .. } => {
            // Bool-producing operators (comparisons, &&/||) always yield
            // Bool regardless of operand types — check this first since
            // it's unambiguous, unlike the float-operand heuristic below.
            // Infer Float for the destination temp when either operand is
            // recognizably float-typed (float literal, cast-to-float, or a
            // known-float variable) — without this, expressions like
            // `(total as float) / (len as float)` used directly in a
            // `return` statement (which has no `let`-style type
            // annotation to fall back on) would have their correct double
            // result silently truncated back to int64_t right here.
            let dest_ty = if binop_produces_bool(*op) {
                Type::Bool
            } else if expr_is_float_typed(builder, left) || expr_is_float_typed(builder, right) {
                Type::Float
            } else {
                Type::Inferred
            };
            let temp = builder.new_local(
                format!("_tmp{}", builder.local_counter),
                dest_ty,
                false
            );

            let left_op = lower_expr_to_operand(builder, left, current_block)?;
            let right_op = lower_expr_to_operand(builder, right, current_block)?;

            let rvalue = MirRvalue::BinaryOp(*op, left_op, right_op);

            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                    bb.statements.push(MirStatement::Assign {
                        place: MirPlace::Local(temp),
                        rvalue,
                    });
                }
            }

            Ok(MirOperand::Copy(MirPlace::Local(temp)))
        }

        Expr::Call { func, args, .. } => {
            if let Expr::Ident { name, .. } = func.as_ref() {
                // Look up expected parameter types to detect trait-object params.
                let expected_params = builder.function_param_types
                    .get(name.as_str())
                    .cloned()
                    .unwrap_or_default();

                // Lower each argument, wrapping in a VTableConstruct if
                // the expected param is TraitObject AND the actual arg is
                // a concrete Named struct (not already a TraitObject).
                let mut arg_operands: Vec<MirOperand> = Vec::with_capacity(args.len());
                for (i, arg) in args.iter().enumerate() {
                    let raw = lower_expr_to_operand(builder, arg, current_block)?;
                    let expected = expected_params.get(i);
                    if let Some(Type::TraitObject(trait_name)) = expected {
                        // Get concrete struct name from the arg's declared type
                        let struct_name = if let MirOperand::Copy(MirPlace::Local(id))
                                            | MirOperand::Move(MirPlace::Local(id)) = &raw {
                            builder.current_function.as_ref()
                                .and_then(|f| f.locals.iter().chain(f.params.iter()).find(|l| l.id == *id))
                                .and_then(|l| if let Type::Named(s) = &l.ty { Some(s.clone()) } else { None })
                        } else { None };

                        if let Some(sname) = struct_name {
                            let method_names = builder.trait_method_names
                                .get(trait_name.as_str())
                                .cloned()
                                .unwrap_or_default();
                            let vtable_temp = builder.new_local(
                                format!("_vtbl{}", builder.local_counter),
                                Type::TraitObject(trait_name.clone()),
                                false,
                            );
                            if let Some(fd) = &mut builder.current_function {
                                if let Some(bb) = fd.basic_blocks.get_mut(*current_block) {
                                    bb.statements.push(MirStatement::Assign {
                                        place: MirPlace::Local(vtable_temp),
                                        rvalue: MirRvalue::VTableConstruct {
                                            trait_name: trait_name.clone(),
                                            struct_name: sname,
                                            data_operand: raw,
                                            method_names,
                                        },
                                    });
                                }
                            }
                            arg_operands.push(MirOperand::Copy(MirPlace::Local(vtable_temp)));
                            continue;
                        }
                    }
                    arg_operands.push(raw);
                }

                // Look up the real return type so non-void calls get a
                // correctly-typed destination temp (e.g. float-returning
                // functions don't get truncated into an int64_t slot).
                // For void-returning functions, keep Inferred rather than
                // Void — `void` isn't a valid C variable type, and the
                // codegen layer already has its own separate void-call
                // handling (it skips assigning the result at all when the
                // function is known void) that doesn't depend on this local's
                // declared type.
                let dest_ty = builder.function_return_types
                    .get(name)
                    .cloned()
                    .filter(|t| !matches!(t, Type::Void))
                    .unwrap_or(Type::Inferred);
                let temp = builder.new_local(
                    format!("_tmp{}", builder.local_counter),
                    dest_ty,
                    false
                );

                let next_block = builder.new_block();
                let cb = *current_block;

                if let Some(func_data) = &mut builder.current_function {
                    if let Some(bb) = func_data.basic_blocks.get_mut(cb) {
                        bb.terminator = MirTerminator::Call {
                            func: name.clone(),
                            args: arg_operands,
                            destination: Some(MirPlace::Local(temp)),
                            target: next_block,
                        };
                    }
                }

                // Advance current block to the continuation block
                *current_block = next_block;

                Ok(MirOperand::Copy(MirPlace::Local(temp)))
            } else {
                Err("Indirect calls not yet supported".to_string())
            }
        }

        Expr::Cast { expr: inner, target_type, .. } => {
            // `x as float` / `y as int` — evaluate the inner expression
            // then wrap it in MirRvalue::Cast (which codegen already knows
            // how to render as a C-style cast, `((double)(x))`). This is
            // the only way to convert between numeric types, since
            // Quantum has no implicit int/float promotion.
            let operand = lower_expr_to_operand(builder, inner, current_block)?;
            let temp = builder.new_local(
                format!("_tmp{}", builder.local_counter),
                target_type.clone(),
                false
            );
            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                    bb.statements.push(MirStatement::Assign {
                        place: MirPlace::Local(temp),
                        rvalue: MirRvalue::Cast(operand, target_type.clone()),
                    });
                }
            }
            Ok(MirOperand::Copy(MirPlace::Local(temp)))
        }

        Expr::Ternary { condition, then_expr, else_expr, span } => {
            // Desugar `cond ? a : b` into the equivalent `if cond { a } else { b }`
            // and reuse the existing (already correctly-fixed) If lowering
            // machinery rather than duplicating the branch/merge-block logic.
            let then_block = Block {
                stmts: Vec::new(),
                expr: Some(then_expr.clone()),
                span: then_expr.span(),
            };
            let else_block = Block {
                stmts: Vec::new(),
                expr: Some(else_expr.clone()),
                span: else_expr.span(),
            };
            let desugared = Expr::If {
                condition: condition.clone(),
                then_block,
                else_block: Some(else_block),
                span: *span,
            };
            lower_expr_to_operand(builder, &desugared, current_block)
        }

        Expr::If { condition, then_block, else_block, .. } => {
            // Lower if-expression as control flow
            let then_bb = builder.new_block();
            let else_bb = builder.new_block();
            let merge_bb = builder.new_block();

            // Pre-declare the merge-result temp now (before lowering
            // either branch) so both branches can assign into the same
            // local before jumping to merge_bb. This is what makes
            // `let y = if cond { 100 } else { 200 }` actually carry 100 or
            // 200 into `y` — previously this temp was declared but never
            // assigned by either branch, always reading as garbage/zero.
            let result_temp = builder.new_local(
                format!("_if_tmp{}", builder.local_counter),
                Type::Inferred,
                false,
            );

            // Evaluate condition in current block
            let cond_op = lower_expr_to_operand(builder, condition, current_block)?;
            let cb = *current_block;
            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(cb) {
                    if matches!(bb.terminator, MirTerminator::Unreachable) {
                        bb.terminator = MirTerminator::If {
                            condition: cond_op,
                            then_block: then_bb,
                            else_block: else_bb,
                        };
                    }
                }
            }

            // Lower then block. lower_block's own trailing-expression
            // handling sets Return(Some(operand)) on whatever block it
            // ends on — for a nested if/ternary branch (as opposed to a
            // genuine function body) that's never what we want: the value
            // needs to flow into result_temp and continue to merge_bb,
            // not actually return from the enclosing function. We rewrite
            // any such terminator we find into an assignment + Goto.
            let then_final = lower_then_or_else_block(builder, then_block, then_bb, result_temp, merge_bb)?;
            let _ = then_final;

            // Lower else block if present (mirrors the then-block handling above)
            if let Some(else_b) = else_block {
                let else_final = lower_then_or_else_block(builder, else_b, else_bb, result_temp, merge_bb)?;
                let _ = else_final;
            } else {
                // No else — connect the else_bb directly to merge
                if let Some(func) = &mut builder.current_function {
                    if let Some(bb) = func.basic_blocks.get_mut(else_bb) {
                        bb.terminator = MirTerminator::Goto(merge_bb);
                    }
                }
            }

            *current_block = merge_bb;
            Ok(MirOperand::Copy(MirPlace::Local(result_temp)))
        }

        Expr::MethodCall { receiver, method, args, .. } => {
            match method.as_str() {
                "to_string" => {
                    // Lower the receiver, then call the appropriate q_*_to_str
                    // helper based on its type. Locals default to Inferred
                    // (int64_t in C), so q_int_to_str is the safe default;
                    // float locals are explicitly typed and get q_float_to_str.
                    let recv_op = lower_expr_to_operand(builder, receiver, current_block)?;
                    let recv_ty = match &recv_op {
                        MirOperand::Copy(MirPlace::Local(id)) | MirOperand::Move(MirPlace::Local(id)) => {
                            builder.current_function.as_ref()
                                .and_then(|f| f.locals.iter().find(|l| l.id == *id))
                                .map(|l| l.ty.clone())
                        }
                        MirOperand::Constant(MirConstant::Float(_)) => Some(Type::Float),
                        MirOperand::Constant(MirConstant::Bool(_)) => Some(Type::Bool),
                        MirOperand::Constant(MirConstant::Int(_)) => Some(Type::Int),
                        _ => None,
                    };
                    let helper = match recv_ty {
                        Some(Type::Float) | Some(Type::F32) | Some(Type::F64) => "q_float_to_str",
                        Some(Type::Bool) => "q_bool_to_str",
                        _ => "q_int_to_str",
                    };

                    let temp = builder.new_local(
                        format!("_tmp{}", builder.local_counter),
                        Type::String,
                        false
                    );
                    let next_block = builder.new_block();
                    let cb = *current_block;
                    if let Some(func_data) = &mut builder.current_function {
                        if let Some(bb) = func_data.basic_blocks.get_mut(cb) {
                            bb.terminator = MirTerminator::Call {
                                func: helper.to_string(),
                                args: vec![recv_op],
                                destination: Some(MirPlace::Local(temp)),
                                target: next_block,
                            };
                        }
                    }
                    *current_block = next_block;
                    Ok(MirOperand::Copy(MirPlace::Local(temp)))
                }
                "len" => {
                    // Determine if the receiver is a string (declared
                    // Type::String, or a string literal) — use strlen() for
                    // strings, array-length arithmetic otherwise.
                    let recv_is_string = match receiver.as_ref() {
                        Expr::Literal { value: Literal::String(_), .. } => true,
                        Expr::Ident { name, .. } => {
                            builder.lookup_local(name)
                                .and_then(|id| builder.current_function.as_ref()
                                    .and_then(|f| f.locals.iter().find(|l| l.id == id)))
                                .map(|l| l.ty == Type::String)
                                .unwrap_or(false)
                        }
                        _ => false,
                    };

                    if recv_is_string {
                        let recv_op = lower_expr_to_operand(builder, receiver, current_block)?;
                        let temp = builder.new_local(
                            format!("_tmp{}", builder.local_counter),
                            Type::Int,
                            false
                        );
                        let next_block = builder.new_block();
                        let cb = *current_block;
                        if let Some(func_data) = &mut builder.current_function {
                            if let Some(bb) = func_data.basic_blocks.get_mut(cb) {
                                bb.terminator = MirTerminator::Call {
                                    func: "q_strlen".to_string(),
                                    args: vec![recv_op],
                                    destination: Some(MirPlace::Local(temp)),
                                    target: next_block,
                                };
                            }
                        }
                        *current_block = next_block;
                        return Ok(MirOperand::Copy(MirPlace::Local(temp)));
                    }

                    let place = lower_expr_to_place(builder, receiver, current_block)?;
                    let temp = builder.new_local(
                        format!("_tmp{}", builder.local_counter),
                        Type::Int,
                        false
                    );
                    if let Some(func) = &mut builder.current_function {
                        if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                            bb.statements.push(MirStatement::Assign {
                                place: MirPlace::Local(temp),
                                rvalue: MirRvalue::Len(place),
                            });
                        }
                    }
                    Ok(MirOperand::Copy(MirPlace::Local(temp)))
                }
                "unwrap" | "expect" => {
                    // `result.unwrap()` — extract value or abort with
                    // a beginner-friendly error message explaining what
                    // went wrong and how to fix it.
                    let recv_op = lower_expr_to_operand(builder, receiver, current_block)?;
                    let src_id = if let MirOperand::Copy(MirPlace::Local(id))
                                    | MirOperand::Move(MirPlace::Local(id)) = &recv_op { *id }
                    else { return Err("unwrap: expected local".to_string()); };

                    let recv_ty = builder.current_function.as_ref()
                        .and_then(|f| f.locals.iter().chain(f.params.iter()).find(|l| l.id == src_id))
                        .map(|l| l.ty.clone());

                    // Determine field name and panic message based on type
                    let (check_field, val_ty, panic_msg) = match &recv_ty {
                        Some(Type::Result(ok_ty, _)) => (
                            "is_ok", *ok_ty.clone(),
                            "called .unwrap() on an Err value — the operation failed. Check .is_ok first, or use a match expression to handle both Ok and Err cases."
                        ),
                        Some(Type::Option(inner)) => (
                            "has_value", *inner.clone(),
                            "called .unwrap() on a None value — the optional was empty. Check .has_value first, or use a match expression to handle both Some and None cases."
                        ),
                        _ => ("is_ok", Type::Inferred,
                              "called .unwrap() on an empty value")
                    };

                    let check_temp = builder.new_local(format!("_unwrap_ok{}", builder.local_counter), Type::Bool, false);
                    let val_temp = builder.new_local(format!("_unwrap_val{}", builder.local_counter), val_ty, false);
                    let ok_block = builder.new_block();
                    let fail_block = builder.new_block();
                    let cont_block = builder.new_block();

                    if let Some(func) = &mut builder.current_function {
                        if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                            bb.statements.push(MirStatement::Assign {
                                place: MirPlace::Local(check_temp),
                                rvalue: MirRvalue::Use(MirOperand::Copy(
                                    MirPlace::Field(Box::new(MirPlace::Local(src_id)), check_field.to_string())
                                )),
                            });
                            bb.terminator = MirTerminator::If {
                                condition: MirOperand::Copy(MirPlace::Local(check_temp)),
                                then_block: ok_block,
                                else_block: fail_block,
                            };
                        }
                        // fail_block: abort with clear message
                        if let Some(bb) = func.basic_blocks.get_mut(fail_block) {
                            bb.terminator = MirTerminator::Call {
                                func: "__quantum_panic".to_string(),
                                args: vec![MirOperand::Constant(MirConstant::String(panic_msg.to_string()))],
                                destination: None,
                                target: cont_block,
                            };
                        }
                        // ok_block: extract value
                        if let Some(bb) = func.basic_blocks.get_mut(ok_block) {
                            bb.statements.push(MirStatement::Assign {
                                place: MirPlace::Local(val_temp),
                                rvalue: MirRvalue::Use(MirOperand::Copy(
                                    MirPlace::Field(Box::new(MirPlace::Local(src_id)), "value".to_string())
                                )),
                            });
                            bb.terminator = MirTerminator::Goto(cont_block);
                        }
                    }
                    *current_block = cont_block;
                    Ok(MirOperand::Copy(MirPlace::Local(val_temp)))
                }
                "is_ok" | "is_err" | "is_some" | "is_none" => {
                    let recv_op = lower_expr_to_operand(builder, receiver, current_block)?;
                    let src_id = if let MirOperand::Copy(MirPlace::Local(id))
                                    | MirOperand::Move(MirPlace::Local(id)) = &recv_op { *id }
                    else { return Err(format!(".{}(): expected local", method)); };
                    let field_name = if method == "is_some" || method == "is_none" {
                        "has_value"
                    } else { "is_ok" };
                    let negate = method == "is_err" || method == "is_none";
                    let temp = builder.new_local(format!("_chk{}", builder.local_counter), Type::Bool, false);
                    let neg_temp = if negate {
                        Some(builder.new_local(format!("_neg{}", builder.local_counter), Type::Bool, false))
                    } else { None };
                    if let Some(func) = &mut builder.current_function {
                        if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                            bb.statements.push(MirStatement::Assign {
                                place: MirPlace::Local(temp),
                                rvalue: MirRvalue::Use(MirOperand::Copy(
                                    MirPlace::Field(Box::new(MirPlace::Local(src_id)), field_name.to_string())
                                )),
                            });
                            if let Some(nt) = neg_temp {
                                bb.statements.push(MirStatement::Assign {
                                    place: MirPlace::Local(nt),
                                    rvalue: MirRvalue::UnaryOp(crate::ast::UnOp::Not, MirOperand::Copy(MirPlace::Local(temp))),
                                });
                                return Ok(MirOperand::Copy(MirPlace::Local(nt)));
                            }
                        }
                    }
                    Ok(MirOperand::Copy(MirPlace::Local(temp)))
                }
                "unwrap_or" => {
                    // `result.unwrap_or(default)` — return value if Ok/Some, else return default
                    let recv_op = lower_expr_to_operand(builder, receiver, current_block)?;
                    let default_op = if let Some(first_arg) = args.first() {
                        lower_expr_to_operand(builder, first_arg, current_block)?
                    } else {
                        return Err("unwrap_or requires a default value argument".to_string());
                    };
                    let src_id = if let MirOperand::Copy(MirPlace::Local(id))
                                    | MirOperand::Move(MirPlace::Local(id)) = &recv_op { *id }
                    else { return Err("unwrap_or: expected local".to_string()); };

                    let recv_ty = builder.current_function.as_ref()
                        .and_then(|f| f.locals.iter().chain(f.params.iter()).find(|l| l.id == src_id))
                        .map(|l| l.ty.clone());
                    let check_field = match &recv_ty {
                        Some(Type::Option(_)) => "has_value",
                        _ => "is_ok",
                    };

                    let check_temp = builder.new_local(format!("_uor_ok{}", builder.local_counter), Type::Bool, false);
                    let result_temp = builder.new_local(format!("_uor_val{}", builder.local_counter), Type::Inferred, false);
                    let ok_block = builder.new_block();
                    let else_block = builder.new_block();
                    let merge_block = builder.new_block();

                    if let Some(func) = &mut builder.current_function {
                        if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                            bb.statements.push(MirStatement::Assign {
                                place: MirPlace::Local(check_temp),
                                rvalue: MirRvalue::Use(MirOperand::Copy(
                                    MirPlace::Field(Box::new(MirPlace::Local(src_id)), check_field.to_string())
                                )),
                            });
                            bb.terminator = MirTerminator::If {
                                condition: MirOperand::Copy(MirPlace::Local(check_temp)),
                                then_block: ok_block,
                                else_block,
                            };
                        }
                        if let Some(bb) = func.basic_blocks.get_mut(ok_block) {
                            bb.statements.push(MirStatement::Assign {
                                place: MirPlace::Local(result_temp),
                                rvalue: MirRvalue::Use(MirOperand::Copy(
                                    MirPlace::Field(Box::new(MirPlace::Local(src_id)), "value".to_string())
                                )),
                            });
                            bb.terminator = MirTerminator::Goto(merge_block);
                        }
                        if let Some(bb) = func.basic_blocks.get_mut(else_block) {
                            bb.statements.push(MirStatement::Assign {
                                place: MirPlace::Local(result_temp),
                                rvalue: MirRvalue::Use(default_op),
                            });
                            bb.terminator = MirTerminator::Goto(merge_block);
                        }
                    }
                    *current_block = merge_block;
                    Ok(MirOperand::Copy(MirPlace::Local(result_temp)))
                }
                _ => {
                    let recv_op = lower_expr_to_operand(builder, receiver, current_block)?;
                    let recv_ty = if let MirOperand::Copy(MirPlace::Local(id))
                                    | MirOperand::Move(MirPlace::Local(id)) = &recv_op {
                        builder.current_function.as_ref()
                            .and_then(|f| f.locals.iter().chain(f.params.iter()).find(|l| l.id == *id))
                            .map(|l| l.ty.clone())
                    } else { None };

                    if let Some(Type::TraitObject(trait_name)) = &recv_ty {
                        // Dynamic dispatch: call through the vtable function pointer.
                        // dest_ty: find return type from any impl of this method.
                        let dest_ty = builder.function_return_types.iter()
                            .find(|(k, _)| k.ends_with(&format!("_{}", method)))
                            .map(|(_, v)| v.clone())
                            .filter(|t| !matches!(t, Type::Void | Type::Inferred))
                            .unwrap_or(Type::Inferred);
                        let dyn_func = format!("__dyn_dispatch__{}", method);
                        let mut all_args = vec![recv_op];
                        for arg in args {
                            all_args.push(lower_expr_to_operand(builder, arg, current_block)?);
                        }
                        let temp = builder.new_local(format!("_tmp{}", builder.local_counter), dest_ty, false);
                        let next_block = builder.new_block();
                        let cb = *current_block;
                        if let Some(fd) = &mut builder.current_function {
                            if let Some(bb) = fd.basic_blocks.get_mut(cb) {
                                bb.terminator = MirTerminator::Call {
                                    func: dyn_func, args: all_args,
                                    destination: Some(MirPlace::Local(temp)), target: next_block,
                                };
                            }
                        }
                        *current_block = next_block;
                        return Ok(MirOperand::Copy(MirPlace::Local(temp)));
                    }

                    // ── Built-in dot-methods for all types ──────────────────────
                    match method.as_str() {
                        "upper"       => return emit_method_call(builder, current_block, receiver, args, "q_str_to_upper",    Type::String),
                        "lower"       => return emit_method_call(builder, current_block, receiver, args, "q_str_to_lower",    Type::String),
                        "trim"        => return emit_method_call(builder, current_block, receiver, args, "q_str_trim",        Type::String),
                        "contains"    => return emit_method_call(builder, current_block, receiver, args, "q_str_contains",    Type::Int),
                        "starts_with" => return emit_method_call(builder, current_block, receiver, args, "q_str_starts_with", Type::Int),
                        "ends_with"   => return emit_method_call(builder, current_block, receiver, args, "q_str_ends_with",   Type::Int),
                        "replace"     => return emit_method_call(builder, current_block, receiver, args, "q_str_replace",     Type::String),
                        "before"      => return emit_method_call(builder, current_block, receiver, args, "q_str_split_first", Type::String),
                        "after"       => return emit_method_call(builder, current_block, receiver, args, "q_str_split_last",  Type::String),
                        "substr"      => return emit_method_call(builder, current_block, receiver, args, "q_str_substr",      Type::String),
                        "repeat"      => return emit_method_call(builder, current_block, receiver, args, "q_str_repeat",      Type::String),
                        "char_at"     => return emit_method_call(builder, current_block, receiver, args, "q_str_index",       Type::Int),
                        "parse_int"   => return emit_method_call(builder, current_block, receiver, args, "q_str_parse_int",   Type::I64),
                        "parse_float" => return emit_method_call(builder, current_block, receiver, args, "q_str_parse_float", Type::Float),
                        "is_empty"    => return emit_method_call(builder, current_block, receiver, args, "q_str_len",         Type::Int),
                        "abs"   => return emit_method_call(builder, current_block, receiver, args, "q_num_abs",   Type::Float),
                        "sqrt"  => return emit_method_call(builder, current_block, receiver, args, "q_num_sqrt",  Type::Float),
                        "pow"   => return emit_method_call(builder, current_block, receiver, args, "q_num_pow",   Type::Float),
                        "floor" => return emit_method_call(builder, current_block, receiver, args, "q_num_floor", Type::Float),
                        "ceil"  => return emit_method_call(builder, current_block, receiver, args, "q_num_ceil",  Type::Float),
                        "round" => return emit_method_call(builder, current_block, receiver, args, "q_num_round", Type::Float),
                        "min"   => return emit_method_call(builder, current_block, receiver, args, "q_num_min",   Type::Float),
                        "max"   => return emit_method_call(builder, current_block, receiver, args, "q_num_max",   Type::Float),
                        "clamp" => return emit_method_call(builder, current_block, receiver, args, "q_num_clamp", Type::Float),
                        "help"  => return emit_method_call(builder, current_block, receiver, args, "q_print_help", Type::Int),
                        // Int methods
                        "is_even"    => return emit_method_call(builder, current_block, receiver, args, "q_int_is_even",   Type::Int),
                        "is_odd"     => return emit_method_call(builder, current_block, receiver, args, "q_int_is_odd",    Type::Int),
                        "to_float"   => return emit_method_call(builder, current_block, receiver, args, "q_int_to_float",  Type::Float),
                        "to_string"  => return emit_method_call(builder, current_block, receiver, args, "q_int_to_str",    Type::String),
                        "to_str"     => return emit_method_call(builder, current_block, receiver, args, "q_int_to_str",    Type::String),
                        "bit_count"  => return emit_method_call(builder, current_block, receiver, args, "q_int_bit_count", Type::Int),
                        // Float-specific
                        "to_int"     => return emit_method_call(builder, current_block, receiver, args, "q_float_to_int",      Type::I64),
                        "is_nan"     => return emit_method_call(builder, current_block, receiver, args, "q_float_is_nan",      Type::Int),
                        "is_infinite"=> return emit_method_call(builder, current_block, receiver, args, "q_float_is_infinite", Type::Int),
                        "is_finite"  => return emit_method_call(builder, current_block, receiver, args, "q_float_is_finite",   Type::Int),
                        "is_positive"=> return emit_method_call(builder, current_block, receiver, args, "q_float_is_positive", Type::Int),
                        "is_negative"=> return emit_method_call(builder, current_block, receiver, args, "q_float_is_negative", Type::Int),
                        // Bool methods
                        "to_int"     => return emit_method_call(builder, current_block, receiver, args, "q_bool_to_int",  Type::I64),
                        "not"        => return emit_method_call(builder, current_block, receiver, args, "q_bool_not",     Type::Int),
                        // Char methods
                        "is_alpha"   => return emit_method_call(builder, current_block, receiver, args, "q_char_is_alpha", Type::Int),
                        "is_digit"   => return emit_method_call(builder, current_block, receiver, args, "q_char_is_digit", Type::Int),
                        "is_upper"   => return emit_method_call(builder, current_block, receiver, args, "q_char_is_upper", Type::Int),
                        "is_lower"   => return emit_method_call(builder, current_block, receiver, args, "q_char_is_lower", Type::Int),
                        "is_space"   => return emit_method_call(builder, current_block, receiver, args, "q_char_is_space", Type::Int),
                        // Extra string methods
                        "capitalize" => return emit_method_call(builder, current_block, receiver, args, "q_str_capitalize", Type::String),
                        "title"      => return emit_method_call(builder, current_block, receiver, args, "q_str_title",      Type::String),
                        "lstrip"     => return emit_method_call(builder, current_block, receiver, args, "q_str_lstrip",     Type::String),
                        "rstrip"     => return emit_method_call(builder, current_block, receiver, args, "q_str_rstrip",     Type::String),
                        "count"      => return emit_method_call(builder, current_block, receiver, args, "q_str_count",      Type::Int),
                        "center"     => return emit_method_call(builder, current_block, receiver, args, "q_str_center",     Type::String),
                        "zfill"      => return emit_method_call(builder, current_block, receiver, args, "q_str_zfill",      Type::String),
                        "find"       => return emit_method_call(builder, current_block, receiver, args, "q_str_find",       Type::Int),
                        "rfind"      => return emit_method_call(builder, current_block, receiver, args, "q_str_rfind",      Type::Int),
                        "eq"         => return emit_method_call(builder, current_block, receiver, args, "q_str_eq",         Type::Int),
                        "type_of" | "type_name" => {
                            return Ok(MirOperand::Constant(MirConstant::String("value".to_string())));
                        }
                        _ => {} // fall through to struct dispatch
                    }
                    let struct_name = if let Some(Type::Named(s)) = recv_ty { Some(s) } else { None };
                    let func_name = if let Some(sname) = struct_name {
                        format!("{}_{}", sname, method)
                    } else {
                        return Err(format!(
                            "Cannot call method .{}() — receiver type is not a known struct. \
                             Built-in methods are: to_string, len",
                            method
                        ));
                    };

                    // Build the full argument list: receiver first, then
                    // any explicit arguments (same as C `Point_area(self, x, y)`)
                    let mut all_args = vec![recv_op];
                    for arg in args {
                        all_args.push(lower_expr_to_operand(builder, arg, current_block)?);
                    }

                    // Look up the method's return type from the pre-computed
                    // function_return_types map (keyed by the mangled name).
                    let dest_ty = builder.function_return_types
                        .get(&func_name)
                        .cloned()
                        .filter(|t| !matches!(t, Type::Void))
                        .unwrap_or(Type::Inferred);

                    let temp = builder.new_local(
                        format!("_tmp{}", builder.local_counter),
                        dest_ty,
                        false
                    );
                    let next_block = builder.new_block();
                    let cb = *current_block;
                    if let Some(func_data) = &mut builder.current_function {
                        if let Some(bb) = func_data.basic_blocks.get_mut(cb) {
                            bb.terminator = MirTerminator::Call {
                                func: func_name,
                                args: all_args,
                                destination: Some(MirPlace::Local(temp)),
                                target: next_block,
                            };
                        }
                    }
                    *current_block = next_block;
                    Ok(MirOperand::Copy(MirPlace::Local(temp)))
                }
            }
        }

        Expr::FieldAccess { expr: base_expr, field, .. } => {
            // Reading `p.x` — lower the base to a place, build the field
            // place, and infer the destination temp's type from the
            // struct's declared field type (so e.g. a string field reads
            // back as String, not Inferred/int64_t, the same reasoning as
            // the array-element-type inference for Index reads above).
            let base_place = lower_expr_to_place(builder, base_expr, current_block)?;
            let place = MirPlace::Field(Box::new(base_place.clone()), field.clone());

            let field_ty = if let MirPlace::Local(base_id) = &base_place {
                builder.current_function.as_ref()
                    .and_then(|f| f.locals.iter().chain(f.params.iter()).find(|l| l.id == *base_id))
                    .and_then(|l| match &l.ty {
                        Type::Named(struct_name) => builder.struct_field_types
                            .get(struct_name)
                            .and_then(|fields| fields.get(field))
                            .cloned(),
                        // Result<T,E> known field types
                        Type::Result(ok_ty, err_ty) => match field.as_str() {
                            "is_ok" => Some(Type::Bool),
                            "value" => Some(*ok_ty.clone()),
                            "error" => Some(*err_ty.clone()),
                            _ => None,
                        },
                        // Option<T> known field types
                        Type::Option(inner_ty) => match field.as_str() {
                            "has_value" => Some(Type::Bool),
                            "value" => Some(*inner_ty.clone()),
                            _ => None,
                        },
                        _ => None,
                    })
            } else {
                None
            };

            let temp = builder.new_local(
                format!("_tmp{}", builder.local_counter),
                field_ty.unwrap_or(Type::Inferred),
                false
            );
            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                    bb.statements.push(MirStatement::Assign {
                        place: MirPlace::Local(temp),
                        rvalue: MirRvalue::Use(MirOperand::Copy(place)),
                    });
                }
            }
            Ok(MirOperand::Copy(MirPlace::Local(temp)))
        }

        Expr::StructInit { name, fields, .. } => {
            // `Point { x: 10, y: 20 }` — lower each field's value
            // expression and construct a StructInit rvalue, which codegen
            // renders as a C designated initializer.
            let mut field_ops = Vec::with_capacity(fields.len());
            for (field_name, field_expr) in fields {
                let op = lower_expr_to_operand(builder, field_expr, current_block)?;
                field_ops.push((field_name.clone(), op));
            }
            let temp = builder.new_local(
                format!("_tmp{}", builder.local_counter),
                Type::Named(name.clone()),
                false
            );
            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                    bb.statements.push(MirStatement::Assign {
                        place: MirPlace::Local(temp),
                        rvalue: MirRvalue::StructInit(name.clone(), field_ops),
                    });
                }
            }
            Ok(MirOperand::Copy(MirPlace::Local(temp)))
        }

        Expr::Some { value, .. } => {
            let val_op = lower_expr_to_operand(builder, value, current_block)?;
            let temp = builder.new_local(format!("_opt{}", builder.local_counter), Type::Option(Box::new(Type::Inferred)), false);
            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                    bb.statements.push(MirStatement::Assign { place: MirPlace::Local(temp), rvalue: MirRvalue::MakeSome(val_op) });
                }
            }
            Ok(MirOperand::Copy(MirPlace::Local(temp)))
        }

        Expr::None { .. } => {
            let temp = builder.new_local(format!("_opt{}", builder.local_counter), Type::Option(Box::new(Type::Inferred)), false);
            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                    bb.statements.push(MirStatement::Assign { place: MirPlace::Local(temp), rvalue: MirRvalue::MakeNone });
                }
            }
            Ok(MirOperand::Copy(MirPlace::Local(temp)))
        }

        Expr::Ok { value, .. } => {
            let val_op = lower_expr_to_operand(builder, value, current_block)?;
            let temp = builder.new_local(format!("_res{}", builder.local_counter), Type::Result(Box::new(Type::Inferred), Box::new(Type::Inferred)), false);
            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                    bb.statements.push(MirStatement::Assign { place: MirPlace::Local(temp), rvalue: MirRvalue::MakeOk(val_op) });
                }
            }
            Ok(MirOperand::Copy(MirPlace::Local(temp)))
        }

        Expr::Err { value, .. } => {
            let val_op = lower_expr_to_operand(builder, value, current_block)?;
            let temp = builder.new_local(format!("_res{}", builder.local_counter), Type::Result(Box::new(Type::Inferred), Box::new(Type::Inferred)), false);
            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                    bb.statements.push(MirStatement::Assign { place: MirPlace::Local(temp), rvalue: MirRvalue::MakeErr(val_op) });
                }
            }
            Ok(MirOperand::Copy(MirPlace::Local(temp)))
        }

        Expr::Try { expr: inner, .. } => {
            // `x?` — if Result is Err or Option is None, return it early;
            // otherwise extract the inner value.
            let inner_op = lower_expr_to_operand(builder, inner, current_block)?;
            let src_id = if let MirOperand::Copy(MirPlace::Local(id)) | MirOperand::Move(MirPlace::Local(id)) = &inner_op {
                *id
            } else {
                return Err("? operator: expected local variable result".to_string());
            };
            let check_temp = builder.new_local(format!("_try_ok{}", builder.local_counter), Type::Bool, false);
            let val_temp = builder.new_local(format!("_try_val{}", builder.local_counter), Type::Inferred, false);
            let ok_block = builder.new_block();
            let fail_block = builder.new_block();
            if let Some(func) = &mut builder.current_function {
                // Current block: check is_ok/has_value, branch
                if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                    bb.statements.push(MirStatement::Assign {
                        place: MirPlace::Local(check_temp),
                        rvalue: MirRvalue::Use(MirOperand::Copy(
                            MirPlace::Field(Box::new(MirPlace::Local(src_id)), "is_ok".to_string())
                        )),
                    });
                    bb.terminator = MirTerminator::If {
                        condition: MirOperand::Copy(MirPlace::Local(check_temp)),
                        then_block: ok_block,
                        else_block: fail_block,
                    };
                }
                // fail_block: propagate error upward
                if let Some(bb) = func.basic_blocks.get_mut(fail_block) {
                    bb.terminator = MirTerminator::Return(Some(inner_op));
                }
                // ok_block: extract .value
                if let Some(bb) = func.basic_blocks.get_mut(ok_block) {
                    bb.statements.push(MirStatement::Assign {
                        place: MirPlace::Local(val_temp),
                        rvalue: MirRvalue::Use(MirOperand::Copy(
                            MirPlace::Field(Box::new(MirPlace::Local(src_id)), "value".to_string())
                        )),
                    });
                }
            }
            *current_block = ok_block;
            Ok(MirOperand::Copy(MirPlace::Local(val_temp)))
        }


        Expr::NamedArg { value, .. } => {
            // Strip the name and lower the value — named args are
            // purely syntactic sugar at the call site.
            lower_expr_to_operand(builder, value, current_block)
        }
        Expr::Match { expr, arms, span } => {
            // Desugar match into an if/else chain:
            //   match x { 1 => a, 2 => b, _ => c }
            // becomes:
            //   if x == 1 { a } else if x == 2 { b } else { c }
            let dummy_span = *span;
            let match_expr = expr.as_ref().clone();

            // Build the if/else chain from last arm to first
            let mut else_expr: Option<Expr> = None;
            for arm in arms.iter().rev() {
                let cond = pattern_to_condition(&arm.pattern, &match_expr, dummy_span);
                let body = arm.body.clone();
                let then_block = crate::ast::Block {
                    stmts: vec![],
                    expr: Some(Box::new(body)),
                    span: dummy_span,
                };
                let else_block = else_expr.map(|e| crate::ast::Block {
                    stmts: vec![],
                    expr: Some(Box::new(e)),
                    span: dummy_span,
                });

                if let Some(cond_expr) = cond {
                    else_expr = Some(Expr::If {
                        condition: Box::new(cond_expr),
                        then_block,
                        else_block,
                        span: dummy_span,
                    });
                } else {
                    // Wildcard or default — always matches
                    else_expr = Some(then_block.expr.map(|e| *e).unwrap_or(
                        Expr::Literal { value: crate::ast::Literal::Null, span: dummy_span }
                    ));
                }
            }

            let desugared = else_expr.unwrap_or(
                Expr::Literal { value: crate::ast::Literal::Null, span: dummy_span }
            );
            lower_expr_to_operand(builder, &desugared, current_block)
        }
        Expr::StringInterp { parts, span } => {
            use crate::ast::StringInterpPart;
            // Lower f"hello {name}" as a chain of string concat calls.
            // Each part is either a literal string or an expression converted
            // to string via __q_to_string, then concatenated with __q_strcat.
            let dummy_span = *span;
            let mut acc_expr: Option<Expr> = None;
            for part in parts {
                let part_expr: Expr = match part {
                    StringInterpPart::Literal(s) => Expr::Literal {
                        value: crate::ast::Literal::String(s.clone()),
                        span: dummy_span,
                    },
                    StringInterpPart::Expr(inner_e) => {
                        // Determine the right converter based on Quantum expression type
                        let conv_name = match inner_e.as_ref() {
                            Expr::Literal { value: crate::ast::Literal::String(_), .. } => "__q_to_string_s",
                            Expr::Literal { value: crate::ast::Literal::Float(_), .. }  => "__q_to_string_f",
                            Expr::Literal { value: crate::ast::Literal::Bool(_), .. }   => "__q_to_string_b",
                            Expr::Literal { value: crate::ast::Literal::Int(_), .. }    => "__q_to_string",
                            Expr::Ident { name: var_name, .. } => {
                                // Look up the variable's declared type
                                match builder.lookup_local_type(var_name) {
                                    Some(Type::String) => "__q_to_string_p",
                                    Some(Type::Float)  => "__q_to_string_f",
                                    Some(Type::Bool)   => "__q_to_string_b",
                                    Some(Type::Int) | Some(Type::I64) => "__q_to_string",
                                    // Unknown/Inferred: assume string pointer (from method results)
                                    _ => "__q_to_string_p",
                                }
                            }
                            // Method call result: check if it's a number or string method
                            Expr::MethodCall { method, .. } => {
                                match method.as_str() {
                                    "abs"|"sqrt"|"pow"|"floor"|"ceil"|"round"|"min"|"max"|"clamp" => "__q_to_string_f",
                                    _ => "__q_to_string_p", // string method
                                }
                            }
                            // Regular call: check if string-returning builtin
                            Expr::Call { func, .. } => {
                                if let Expr::Ident { name, .. } = func.as_ref() {
                                    if name.starts_with("q_str_") {
                                        "__q_to_string_p"
                                    } else { "__q_to_string" }
                                } else { "__q_to_string" }
                            }
                            _ => "__q_to_string",
                        };
                        Expr::Call {
                            func: Box::new(Expr::Ident { name: conv_name.to_string(), span: dummy_span }),
                            args: vec![*inner_e.clone()],
                            span: dummy_span,
                        }
                    }
                };
                acc_expr = Some(match acc_expr {
                    None => part_expr,
                    Some(prev) => Expr::Call {
                        func: Box::new(Expr::Ident { name: "__q_strcat".to_string(), span: dummy_span }),
                        args: vec![prev, part_expr],
                        span: dummy_span,
                    },
                });
            }
            let final_expr = acc_expr.unwrap_or(Expr::Literal {
                value: crate::ast::Literal::String(String::new()),
                span: dummy_span,
            });
            lower_expr_to_operand(builder, &final_expr, current_block)
        }

        Expr::Index { expr: base_expr, index, .. } => {
            // Reading `arr[i]` — lower the base to a place and the index
            // to an operand (mutable context here, so complex index
            // expressions like `arr[i+1]` work fine, unlike the more
            // limited place-lowering path used for assignment targets).
            let base_place = lower_expr_to_place(builder, base_expr, current_block)?;
            let index_operand = lower_expr_to_operand(builder, index, current_block)?;
            let place = MirPlace::Index(Box::new(base_place.clone()), Box::new(index_operand));

            // Infer the destination temp's type from the array's declared
            // element type (e.g. Array(String, 3) -> the read gets String,
            // not Inferred/int64_t) — without this, reading a string out of
            // an array and printing it picks the wrong print_* overload and
            // shows a garbage pointer value instead of the actual text.
            let elem_ty = if let MirPlace::Local(base_id) = &base_place {
                builder.current_function.as_ref()
                    .and_then(|f| f.locals.iter().chain(f.params.iter()).find(|l| l.id == *base_id))
                    .and_then(|l| match &l.ty {
                        Type::Array(inner, _) => Some((**inner).clone()),
                        _ => None,
                    })
            } else {
                None
            };

            let temp = builder.new_local(
                format!("_tmp{}", builder.local_counter),
                elem_ty.unwrap_or(Type::Inferred),
                false
            );
            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                    bb.statements.push(MirStatement::Assign {
                        place: MirPlace::Local(temp),
                        rvalue: MirRvalue::Use(MirOperand::Copy(place)),
                    });
                }
            }
            Ok(MirOperand::Copy(MirPlace::Local(temp)))
        }

        Expr::Array { elements, .. } => {
            // Fixed-size array literal — lower to MirRvalue::Aggregate,
            // which codegen renders as a C brace-initializer `{ ... }`.
            // NOTE: this only works correctly when used directly as a
            // `let` initializer (C arrays can only be brace-initialized at
            // declaration time, never reassigned wholesale later) — array
            // support beyond simple literal declarations is still a work
            // in progress.
            let mut ops = Vec::with_capacity(elements.len());
            for el in elements {
                ops.push(lower_expr_to_operand(builder, el, current_block)?);
            }
            let temp = builder.new_local(
                format!("_tmp{}", builder.local_counter),
                Type::Inferred,
                false
            );
            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                    bb.statements.push(MirStatement::Assign {
                        place: MirPlace::Local(temp),
                        rvalue: MirRvalue::Aggregate(ops),
                    });
                }
            }
            Ok(MirOperand::Copy(MirPlace::Local(temp)))
        }

        _ => {
            // For any other expression kind not explicitly handled above
            // (or by lower_expr_to_rvalue's own arms), build the rvalue
            // directly here rather than delegating to lower_expr_to_rvalue —
            // that function's own catch-all calls back into this function,
            // and routing an unhandled expression through both catch-alls
            // creates infinite mutual recursion (silently overflowing the
            // stack with NO error message, which is how the original array
            // indexing crash manifested). Binary/Unary are the only two
            // expression kinds lower_expr_to_rvalue adds special handling
            // for beyond a plain Use(operand), so inline that fallback
            // logic here instead of bouncing back through it.
            //
            // For Binary specifically, detect float-producing operands so
            // the destination temp gets Float instead of defaulting to
            // Inferred/int64_t (this branch is normally unreachable for
            // plain Binary expressions, since the earlier explicit
            // Expr::Binary arm intercepts those first — this exists as a
            // defensive fallback in case Binary ever reaches here via some
            // other path).
            let dest_ty = if let Expr::Binary { op, left, right, .. } = expr {
                if binop_produces_bool(*op) {
                    Type::Bool
                } else if expr_is_float_typed(builder, left) || expr_is_float_typed(builder, right) {
                    Type::Float
                } else {
                    Type::Inferred
                }
            } else if let Expr::Unary { op: UnOp::Not, .. } = expr {
                Type::Bool
            } else {
                Type::Inferred
            };

            let temp = builder.new_local(
                format!("_tmp{}", builder.local_counter),
                dest_ty,
                false
            );

            let rvalue = match expr {
                Expr::Binary { op, left, right, .. } => {
                    let left_op = lower_expr_to_operand(builder, left, current_block)?;
                    let right_op = lower_expr_to_operand(builder, right, current_block)?;
                    MirRvalue::BinaryOp(*op, left_op, right_op)
                }
                Expr::Unary { op, expr: inner, .. } => {
                    let operand = lower_expr_to_operand(builder, inner, current_block)?;
                    MirRvalue::UnaryOp(*op, operand)
                }
                other => {
                    return Err(format!(
                        "Unsupported expression in this context: {:?}",
                        std::mem::discriminant(other)
                    ));
                }
            };

            if let Some(func) = &mut builder.current_function {
                if let Some(bb) = func.basic_blocks.get_mut(*current_block) {
                    bb.statements.push(MirStatement::Assign {
                        place: MirPlace::Local(temp),
                        rvalue,
                    });
                }
            }

            Ok(MirOperand::Copy(MirPlace::Local(temp)))
        }
    }
}

fn lower_expr_to_rvalue(
    builder: &mut MirBuilder,
    expr: &Expr,
    current_block: &mut usize,
) -> Result<MirRvalue, String> {
    match expr {
        Expr::Binary { op, left, right, .. } => {
            let left_op = lower_expr_to_operand(builder, left, current_block)?;
            let right_op = lower_expr_to_operand(builder, right, current_block)?;
            Ok(MirRvalue::BinaryOp(*op, left_op, right_op))
        }

        Expr::Unary { op, expr, .. } => {
            let operand = lower_expr_to_operand(builder, expr, current_block)?;
            Ok(MirRvalue::UnaryOp(*op, operand))
        }

        Expr::Array { elements, .. } => {
            // Array literal directly in `let` initializer position lowers
            // to Aggregate, which codegen renders as a C brace-initializer
            // (`int arr[5] = {1,2,3,4,5};`) — only valid at declaration
            // time, which is exactly where Stmt::Let uses this function.
            let mut ops = Vec::with_capacity(elements.len());
            for el in elements {
                ops.push(lower_expr_to_operand(builder, el, current_block)?);
            }
            Ok(MirRvalue::Aggregate(ops))
        }

        Expr::StructInit { name, fields, .. } => {
            let mut field_ops = Vec::with_capacity(fields.len());
            for (field_name, field_expr) in fields {
                let op = lower_expr_to_operand(builder, field_expr, current_block)?;
                field_ops.push((field_name.clone(), op));
            }
            Ok(MirRvalue::StructInit(name.clone(), field_ops))
        }

        // Every other expression kind below is handled by an EXPLICIT
        // (non-catch-all) arm inside lower_expr_to_operand, so bouncing
        // through it here terminates after exactly one hop — this is safe,
        // unlike the old generic `_ => lower_expr_to_operand(...)` fallback
        // which used to catch genuinely-unhandled variants too and bounce
        // them into lower_expr_to_operand's *own* catch-all, which bounced
        // straight back here, forever (the original array-indexing stack
        // overflow). If you add a new Expr variant, give it an explicit
        // arm in lower_expr_to_operand's match — never rely on either
        // function's catch-all to handle a new variant correctly.
        Expr::Literal { .. }
        | Expr::Ident { .. }
        | Expr::Call { .. }
        | Expr::MethodCall { .. }
        | Expr::If { .. }
        | Expr::Ternary { .. }
        | Expr::Cast { .. }
        | Expr::Index { .. }
        | Expr::FieldAccess { .. }
        | Expr::StructInit { .. }
        | Expr::Some { .. }
        | Expr::None { .. }
        | Expr::Ok { .. }
        | Expr::Err { .. }
        | Expr::Try { .. }
        | Expr::NamedArg { .. }
        | Expr::StringInterp { .. }
        | Expr::Match { .. } => {
            let operand = lower_expr_to_operand(builder, expr, current_block)?;
            Ok(MirRvalue::Use(operand))
        }

        other => Err(format!(
            "lower_expr_to_rvalue: unsupported expression {:?} — this is an internal compiler error, please report it",
            std::mem::discriminant(other)
        )),
    }
}

fn lower_expr_to_place(builder: &mut MirBuilder, expr: &Expr, current_block: &mut usize) -> Result<MirPlace, String> {
    match expr {
        Expr::Ident { name, .. } => {
            if let Some(local_id) = builder.lookup_local(name) {
                Ok(MirPlace::Local(local_id))
            } else {
                Err(format!("Undefined variable: {}", name))
            }
        }

        Expr::FieldAccess { expr, field, .. } => {
            let base = lower_expr_to_place(builder, expr, current_block)?;
            Ok(MirPlace::Field(Box::new(base), field.clone()))
        }

        Expr::Index { expr, index, .. } => {
            let base = lower_expr_to_place(builder, expr, current_block)?;
            // Now that this function takes a mutable builder and the
            // current block, arbitrary index expressions (e.g. `arr[j+1]`,
            // `arr[f(x)]`) lower the same way reads do — through
            // lower_expr_to_operand — instead of being restricted to bare
            // literals/identifiers. This was a real limitation blocking
            // common patterns like swap-based sorts (`arr[j+1] = temp`).
            let index_operand = lower_expr_to_operand(builder, index, current_block)?;
            Ok(MirPlace::Index(Box::new(base), Box::new(index_operand)))
        }

        _ => Err(format!("Expression cannot be used as place"))
    }
}

fn literal_to_constant(lit: &Literal) -> MirConstant {
    match lit {
        Literal::Int(n) => MirConstant::Int(*n),
        Literal::Float(f) => MirConstant::Float(*f),
        Literal::Bool(b) => MirConstant::Bool(*b),
        Literal::String(s) => MirConstant::String(s.clone()),
        Literal::Null => MirConstant::Null,
        Literal::Char(_) => MirConstant::Int(0), // TODO: proper char handling
    }
}

fn expr_to_constant(expr: &Expr) -> Option<MirConstant> {
    match expr {
        Expr::Literal { value, .. } => Some(literal_to_constant(value)),
        _ => None
    }
}

pub fn optimize(mir: MirProgram, opt_level: u8) -> MirProgram {
    if opt_level == 0 {
        return mir;
    }

    // TODO: Implement optimizations
    // - Dead code elimination
    // - Constant folding
    // - Inline small functions
    // - Common subexpression elimination

    mir
}


/// Convert a match pattern to an equality condition expression.
/// Returns None for wildcards (always match).
fn pattern_to_condition(
    pattern: &crate::ast::Pattern,
    match_expr: &Expr,
    span: crate::lexer::Span,
) -> Option<Expr> {
    use crate::ast::{Pattern, Literal, BinOp};
    match pattern {
        Pattern::Wildcard => None,
        Pattern::Literal(lit) => {
            Some(Expr::Binary {
                op: BinOp::Eq,
                left: Box::new(match_expr.clone()),
                right: Box::new(Expr::Literal { value: lit.clone(), span }),
                span,
            })
        }
        Pattern::Ident(name) => {
            // Could be a variable binding or a constant — treat as equality
            Some(Expr::Binary {
                op: BinOp::Eq,
                left: Box::new(match_expr.clone()),
                right: Box::new(Expr::Ident { name: name.clone(), span }),
                span,
            })
        }
        Pattern::Or(alts) => {
            // 1 | 2 | 3 => (x == 1) || (x == 2) || (x == 3)
            let mut cond: Option<Expr> = None;
            for alt in alts {
                let alt_cond = pattern_to_condition(alt, match_expr, span);
                cond = Some(match (cond, alt_cond) {
                    (None, Some(c)) => c,
                    (Some(prev), Some(c)) => Expr::Binary {
                        op: BinOp::Or,
                        left: Box::new(prev),
                        right: Box::new(c),
                        span,
                    },
                    (Some(prev), None) => prev, // wildcard in OR → always true
                    (None, None) => return None,
                });
            }
            cond
        }
        Pattern::Tuple(_) | Pattern::Struct { .. } => None, // complex — default match
    }
}


/// Emit a simple builtin method call: receiver becomes first arg
fn emit_method_call(
    builder: &mut MirBuilder,
    current_block: &mut usize,
    receiver: &Expr,
    extra_args: &[Expr],
    func_name: &str,
    ret_ty: Type,
) -> Result<MirOperand, String> {
    let recv_op = lower_expr_to_operand(builder, receiver, current_block)?;
    let mut all_args = vec![recv_op];
    for arg in extra_args {
        all_args.push(lower_expr_to_operand(builder, arg, current_block)?);
    }
    let temp_id = builder.local_counter;
    let temp = builder.new_local(format!("_m{}", temp_id), ret_ty, false);
    let next_block = builder.new_block();
    let cb = *current_block;
    if let Some(func_data) = &mut builder.current_function {
        if let Some(bb) = func_data.basic_blocks.get_mut(cb) {
            bb.terminator = MirTerminator::Call {
                func: func_name.to_string(),
                args: all_args,
                destination: Some(MirPlace::Local(temp)),
                target: next_block,
            };
        }
    }
    *current_block = next_block;
    Ok(MirOperand::Copy(MirPlace::Local(temp)))
}
