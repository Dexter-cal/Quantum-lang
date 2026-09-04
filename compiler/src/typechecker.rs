// Quantum Type Checker - Hindley-Milner type inference with extensions
use crate::ast::*;
use std::collections::HashMap;

pub struct TypeChecker {
    scopes: Vec<HashMap<String, Type>>,
    functions: HashMap<String, FunctionSignature>,
    structs: HashMap<String, StructInfo>,
    next_type_var: usize,
    /// The expected return type of the function currently being checked,
    /// used by `Stmt::Return` to validate `return <expr>` against the
    /// function's declared return type.
    current_return_type: Option<Type>,
    /// The struct name whose `impl` block the function currently being
    /// checked belongs to, if any. Used to allow private-field access via
    /// `self.field` inside a struct's own methods while still blocking it
    /// from external code (see Expr::FieldAccess visibility check).
    current_struct_context: Option<String>,
    /// Maps trait name -> the set of method signatures it requires
    /// (name, param types excluding self, return type). Used to verify
    /// `impl Trait for Type` blocks actually implement every required
    /// method with a matching signature — without this check, a trait is
    /// purely decorative and a struct can silently omit required methods.
    traits: HashMap<String, Vec<TraitMethodSig>>,
    /// Which traits each struct has a verified `impl Trait for Struct`
    /// block for. Used later for trait-typed function parameters /
    /// polymorphic dispatch (a struct "satisfies" a trait if it appears
    /// here).
    struct_trait_impls: HashMap<String, std::collections::HashSet<String>>,
}

/// A trait's required method signature — name, parameter types
/// (excluding the implicit `self`), and return type. Used to verify an
/// `impl Trait for Type` block satisfies everything the trait requires.
#[derive(Debug, Clone, PartialEq)]
struct TraitMethodSig {
    name: String,
    param_types: Vec<Type>,
    return_type: Type,
    /// True if the trait declared a default body for this method —
    /// implementing structs may omit it and inherit the default instead
    /// of being required to provide their own.
    has_default: bool,
}

#[derive(Debug, Clone)]
struct FunctionSignature {
    params: Vec<Type>,
    return_type: Type,
    /// Only meaningful for struct methods (registered via Item::Impl) —
    /// regular top-level functions are always callable, so this is always
    /// true for them. Used to enforce `private` methods the same way
    /// private fields are enforced.
    is_pub: bool,
}

#[derive(Debug, Clone)]
struct StructInfo {
    fields: HashMap<String, Type>,
    /// Which fields are declared `pub`. A field not in this set is
    /// private — only accessible via `self` from within that struct's own
    /// `impl` block, not from arbitrary external code. This is the
    /// practically meaningful form of encapsulation Quantum currently
    /// enforces (true cross-file/module visibility would need file-origin
    /// tracking that doesn't exist in the AST yet).
    pub_fields: std::collections::HashSet<String>,
}

impl TypeChecker {
    pub fn new() -> Self {
        let mut checker = Self {
            scopes: vec![HashMap::new()],
            functions: HashMap::new(),
            structs: HashMap::new(),
            next_type_var: 0,
            current_return_type: None,
            current_struct_context: None,
            traits: HashMap::new(),
            struct_trait_impls: HashMap::new(),
        };

        // Add built-in functions
        checker.add_builtin("print", vec![Type::String], Type::Void);
        checker.add_builtin("println", vec![Type::String], Type::Void);
        // Memory primitives - pointers use I64 (64-bit addresses)
        checker.add_builtin("q_alloc",    vec![Type::Int],                       Type::I64);
        checker.add_builtin("q_realloc",  vec![Type::I64, Type::Int],            Type::I64);
        checker.add_builtin("q_free",     vec![Type::I64],                       Type::Void);
        checker.add_builtin("q_memcpy",   vec![Type::I64, Type::I64, Type::Int], Type::Void);
        checker.add_builtin("q_memmove",  vec![Type::I64, Type::I64, Type::Int], Type::Void);
        checker.add_builtin("q_read_i64", vec![Type::I64, Type::Int],            Type::I64);
        checker.add_builtin("q_write_i64",vec![Type::I64, Type::Int, Type::I64], Type::Void);
        checker.add_builtin("q_read_f64", vec![Type::I64, Type::Int],            Type::Float);
        checker.add_builtin("q_write_f64",vec![Type::I64, Type::Int, Type::Float],Type::Void);
        // File I/O
        checker.add_builtin("q_file_read_text",  vec![Type::String],                       Type::String);
        checker.add_builtin("q_file_write_text", vec![Type::String, Type::String],          Type::Int);
        checker.add_builtin("q_file_append_text",vec![Type::String, Type::String],          Type::Int);
        checker.add_builtin("q_file_exists",     vec![Type::String],                        Type::Int);
        checker.add_builtin("q_file_line_count", vec![Type::String],                        Type::Int);
        checker.add_builtin("q_csv_read",        vec![Type::String, Type::I64, Type::Int, Type::Int], Type::Int);
        checker.add_builtin("q_csv_read_col",    vec![Type::String, Type::I64, Type::Int, Type::Int], Type::Int);
        // String map

        // String methods
        checker.add_builtin("q_str_len",         vec![Type::String],                           Type::Int);
        checker.add_builtin("q_str_contains",    vec![Type::String, Type::String],             Type::Int);
        checker.add_builtin("q_str_starts_with", vec![Type::String, Type::String],             Type::Int);
        checker.add_builtin("q_str_ends_with",   vec![Type::String, Type::String],             Type::Int);
        checker.add_builtin("q_str_to_upper",    vec![Type::String],                           Type::String);
        checker.add_builtin("q_str_to_lower",    vec![Type::String],                           Type::String);
        checker.add_builtin("q_str_trim",        vec![Type::String],                           Type::String);
        checker.add_builtin("q_str_replace",     vec![Type::String, Type::String, Type::String], Type::String);
        checker.add_builtin("q_str_split_first", vec![Type::String, Type::String],             Type::String);
        checker.add_builtin("q_str_split_last",  vec![Type::String, Type::String],             Type::String);
        checker.add_builtin("q_str_index",       vec![Type::String, Type::Int],                Type::Int);
        checker.add_builtin("q_str_substr",      vec![Type::String, Type::Int, Type::Int],     Type::String);
        checker.add_builtin("q_str_parse_int",   vec![Type::String],                           Type::I64);
        checker.add_builtin("q_str_parse_float", vec![Type::String],                           Type::Float);
        checker.add_builtin("q_str_from_int",    vec![Type::I64],                              Type::String);
        checker.add_builtin("q_str_from_float",  vec![Type::Float],                            Type::String);
        checker.add_builtin("q_str_repeat",      vec![Type::String, Type::Int],                Type::String);
        checker.add_builtin("__q_to_string",   vec![Type::Int],   Type::String);
        checker.add_builtin("__q_interp",       vec![Type::Inferred], Type::String);
        checker.add_builtin("__q_to_string_f", vec![Type::Float], Type::String);
        checker.add_builtin("__q_to_string_s", vec![Type::String],Type::String);
        checker.add_builtin("__q_to_string_b", vec![Type::Bool],  Type::String);
        checker.add_builtin("__q_to_string_f", vec![Type::Float], Type::String);
        checker.add_builtin("__q_strcat",      vec![Type::String, Type::String], Type::String);

        // Int methods
        checker.add_builtin("q_int_abs",       vec![Type::Int],              Type::I64);
        checker.add_builtin("q_int_to_float",  vec![Type::Int],              Type::Float);
        checker.add_builtin("q_int_to_str",    vec![Type::Int],              Type::String);
        checker.add_builtin("q_int_is_even",   vec![Type::Int],              Type::Int);
        checker.add_builtin("q_int_is_odd",    vec![Type::Int],              Type::Int);
        checker.add_builtin("q_int_pow",       vec![Type::Int, Type::Int],   Type::I64);
        checker.add_builtin("q_int_min",       vec![Type::Int, Type::Int],   Type::I64);
        checker.add_builtin("q_int_max",       vec![Type::Int, Type::Int],   Type::I64);
        checker.add_builtin("q_int_clamp",     vec![Type::Int, Type::Int, Type::Int], Type::I64);
        checker.add_builtin("q_int_bit_count", vec![Type::Int],              Type::Int);
        // Float methods
        checker.add_builtin("q_float_to_int",      vec![Type::Float], Type::I64);
        checker.add_builtin("q_float_to_str",      vec![Type::Float], Type::String);
        checker.add_builtin("q_float_is_nan",      vec![Type::Float], Type::Int);
        checker.add_builtin("q_float_is_infinite", vec![Type::Float], Type::Int);
        checker.add_builtin("q_float_is_finite",   vec![Type::Float], Type::Int);
        checker.add_builtin("q_float_is_positive", vec![Type::Float], Type::Int);
        checker.add_builtin("q_float_is_negative", vec![Type::Float], Type::Int);
        // Bool methods
        checker.add_builtin("q_bool_to_str", vec![Type::Bool],             Type::String);
        checker.add_builtin("q_bool_to_int", vec![Type::Bool],             Type::I64);
        checker.add_builtin("q_bool_and",    vec![Type::Bool, Type::Bool], Type::Int);
        checker.add_builtin("q_bool_or",     vec![Type::Bool, Type::Bool], Type::Int);
        checker.add_builtin("q_bool_not",    vec![Type::Bool],             Type::Int);
        // Char methods
        checker.add_builtin("q_char_to_int",   vec![Type::Char], Type::Int);
        checker.add_builtin("q_char_to_str",   vec![Type::Char], Type::String);
        checker.add_builtin("q_char_is_alpha", vec![Type::Char], Type::Int);
        checker.add_builtin("q_char_is_digit", vec![Type::Char], Type::Int);
        checker.add_builtin("q_char_is_upper", vec![Type::Char], Type::Int);
        checker.add_builtin("q_char_is_lower", vec![Type::Char], Type::Int);
        checker.add_builtin("q_char_is_space", vec![Type::Char], Type::Int);
        checker.add_builtin("q_char_upper",    vec![Type::Char], Type::Char);
        checker.add_builtin("q_char_lower",    vec![Type::Char], Type::Char);
        // Extra string methods
        checker.add_builtin("q_str_capitalize", vec![Type::String],                  Type::String);
        checker.add_builtin("q_str_title",      vec![Type::String],                  Type::String);
        checker.add_builtin("q_str_lstrip",     vec![Type::String],                  Type::String);
        checker.add_builtin("q_str_rstrip",     vec![Type::String],                  Type::String);
        checker.add_builtin("q_str_count",      vec![Type::String, Type::String],    Type::Int);
        checker.add_builtin("q_str_center",     vec![Type::String, Type::Int, Type::Int], Type::String);
        checker.add_builtin("q_str_zfill",      vec![Type::String, Type::Int],       Type::String);
        checker.add_builtin("q_str_find",       vec![Type::String, Type::String],    Type::Int);
        checker.add_builtin("q_str_rfind",      vec![Type::String, Type::String],    Type::Int);
        checker.add_builtin("q_str_eq",         vec![Type::String, Type::String],    Type::Int);
        // String dot-methods
        checker.add_builtin("q_str_len",         vec![Type::String],                              Type::Int);
        checker.add_builtin("q_str_contains",    vec![Type::String, Type::String],               Type::Int);
        checker.add_builtin("q_str_starts_with", vec![Type::String, Type::String],               Type::Int);
        checker.add_builtin("q_str_ends_with",   vec![Type::String, Type::String],               Type::Int);
        checker.add_builtin("q_str_to_upper",    vec![Type::String],                             Type::String);
        checker.add_builtin("q_str_to_lower",    vec![Type::String],                             Type::String);
        checker.add_builtin("q_str_trim",        vec![Type::String],                             Type::String);
        checker.add_builtin("q_str_replace",     vec![Type::String, Type::String, Type::String], Type::String);
        checker.add_builtin("q_str_split_first", vec![Type::String, Type::String],               Type::String);
        checker.add_builtin("q_str_split_last",  vec![Type::String, Type::String],               Type::String);
        checker.add_builtin("q_str_index",       vec![Type::String, Type::Int],                  Type::Int);
        checker.add_builtin("q_str_substr",      vec![Type::String, Type::Int, Type::Int],       Type::String);
        checker.add_builtin("q_str_repeat",      vec![Type::String, Type::Int],                  Type::String);
        checker.add_builtin("q_str_parse_int",   vec![Type::String],                             Type::I64);
        checker.add_builtin("q_str_parse_float", vec![Type::String],                             Type::Float);
        checker.add_builtin("q_str_from_int",    vec![Type::I64],                               Type::String);
        checker.add_builtin("q_str_from_float",  vec![Type::Float],                             Type::String);
        // Number dot-methods
        checker.add_builtin("q_num_abs",   vec![Type::Float],                       Type::Float);
        checker.add_builtin("q_num_sqrt",  vec![Type::Float],                       Type::Float);
        checker.add_builtin("q_num_pow",   vec![Type::Float, Type::Float],          Type::Float);
        checker.add_builtin("q_num_floor", vec![Type::Float],                       Type::Float);
        checker.add_builtin("q_num_ceil",  vec![Type::Float],                       Type::Float);
        checker.add_builtin("q_num_round", vec![Type::Float],                       Type::Float);
        checker.add_builtin("q_num_min",   vec![Type::Float, Type::Float],          Type::Float);
        checker.add_builtin("q_num_max",   vec![Type::Float, Type::Float],          Type::Float);
        checker.add_builtin("q_num_clamp", vec![Type::Float, Type::Float, Type::Float], Type::Float);
        checker.add_builtin("__q_to_string_p", vec![Type::String], Type::String);
        checker.add_builtin("__q_to_string_i64p", vec![Type::I64], Type::String);
        checker.add_builtin("q_print_help",vec![Type::String, Type::I64],          Type::Int);
                checker.add_builtin("q_strmap_new",      vec![Type::Int],                                Type::I64);
        checker.add_builtin("q_strmap_insert",   vec![Type::I64, Type::String, Type::Int],       Type::Void);
        checker.add_builtin("q_strmap_get",      vec![Type::I64, Type::String, Type::I64],       Type::Int);
        checker.add_builtin("q_strmap_contains", vec![Type::I64, Type::String],                  Type::Int);
        checker.add_builtin("q_strmap_len",      vec![Type::I64],                                Type::Int);
        checker.add_builtin("q_strmap_free",     vec![Type::I64],                                Type::Void);

        checker
    }

    fn add_builtin(&mut self, name: &str, params: Vec<Type>, return_type: Type) {
        self.functions.insert(
            name.to_string(),
            FunctionSignature { params, return_type, is_pub: true }
        );
    }

    /// Like `check`, but collects ALL type errors found in the program
    /// instead of stopping at the first. Used by `check_source` in lib.rs
    /// so the LSP can show multiple diagnostics simultaneously.
    /// Returns the list of error messages; an empty vec means no errors.
    pub fn check_all(&mut self, program: &Program) -> Vec<String> {
        let mut errors: Vec<String> = Vec::new();

        // First pass: collect all function signatures and struct definitions
        // (same as check(), must complete before type-checking call sites)
        for item in &program.items {
            match item {
                Item::Function(func) => {
                    let sig = FunctionSignature {
                        params: func.params.iter().map(|p| p.ty.clone()).collect(),
                        return_type: func.return_type.clone().unwrap_or(Type::Void),
                        is_pub: true,
                    };
                    self.functions.insert(func.name.clone(), sig);
                }
                Item::Struct(s) => {
                    let type_param_set: std::collections::HashSet<&str> =
                        s.type_params.iter().map(|tp| tp.as_str()).collect();
                    let fields = s.fields.iter()
                        .map(|f| {
                            let ty = match &f.ty {
                                Type::Named(n) if type_param_set.contains(n.as_str()) =>
                                    Type::TypeParam(n.clone()),
                                other => other.clone(),
                            };
                            (f.name.clone(), ty)
                        })
                        .collect();
                    let pub_fields = s.fields.iter()
                        .filter(|f| f.is_pub)
                        .map(|f| f.name.clone())
                        .collect();
                    self.structs.insert(s.name.clone(), StructInfo { fields, pub_fields });
                }
                Item::ExternBlock(block) => {
                    // Register each extern function's signature exactly like
                    // a regular function — callers can't tell the difference
                    // at the type level. Only the codegen stage treats them
                    // specially (declaration-only, no body to lower).
                    for ext_fn in &block.functions {
                        let sig = FunctionSignature {
                            params: ext_fn.params.iter().map(|p| p.ty.clone()).collect(),
                            return_type: ext_fn.return_type.clone().unwrap_or(Type::Void),
                            is_pub: true,
                        };
                        self.functions.insert(ext_fn.name.clone(), sig);
                    }
                }
                _ => {}
            }
        }

        // Second pass: check everything, continuing past errors
        for item in &program.items {
            match item {
                Item::Function(func) => {
                    self.check_function_collecting(func, &mut errors);
                }
                Item::Const(c) => {
                    match self.check_expr(&c.value) {
                        Ok(expr_type) => {
                            if !self.types_compatible(&c.ty, &expr_type) {
                                errors.push(format!(
                                    "Type mismatch in const {}: expected {}, found {} at line {}, column {}",
                                    c.name, c.ty, expr_type,
                                    c.value.span().line, c.value.span().column
                                ));
                            }
                        }
                        Err(e) => errors.push(e),
                    }
                }
                _ => {}
            }
        }

        errors
    }

    pub fn check(&mut self, program: Program) -> Result<Program, String> {
        // First pass (1a): register structs, traits, externs, and impl
        // methods BEFORE resolving function signatures, so
        // resolve_trait_object_type can correctly distinguish trait names
        // from struct names when a function param is typed as a trait.
        for item in &program.items {
            match item {
                Item::Function(_) => {} // handled in pass 1b
                Item::Struct(s) => {
                    let type_param_set: std::collections::HashSet<&str> =
                        s.type_params.iter().map(|tp| tp.as_str()).collect();
                    let fields = s.fields.iter()
                        .map(|f| {
                            let ty = match &f.ty {
                                Type::Named(n) if type_param_set.contains(n.as_str()) =>
                                    Type::TypeParam(n.clone()),
                                other => other.clone(),
                            };
                            (f.name.clone(), ty)
                        })
                        .collect();
                    let pub_fields = s.fields.iter()
                        .filter(|f| f.is_pub)
                        .map(|f| f.name.clone())
                        .collect();
                    self.structs.insert(
                        s.name.clone(),
                        StructInfo { fields, pub_fields }
                    );
                }
                Item::ExternBlock(block) => {
                    for ext_fn in &block.functions {
                        let sig = FunctionSignature {
                            params: ext_fn.params.iter().map(|p| p.ty.clone()).collect(),
                            return_type: ext_fn.return_type.clone().unwrap_or(Type::Void),
                            is_pub: true,
                        };
                        self.functions.insert(ext_fn.name.clone(), sig);
                    }
                }
                Item::Trait(trait_def) => {
                    let sigs = trait_def.methods.iter().map(|m| TraitMethodSig {
                        name: m.name.clone(),
                        param_types: m.params.iter().skip(1).map(|p| p.ty.clone()).collect(),
                        return_type: m.return_type.clone().unwrap_or(Type::Void),
                        has_default: m.default_body.is_some(),
                    }).collect();
                    self.traits.insert(trait_def.name.clone(), sigs);
                }
                Item::Impl(impl_block) => {
                    // Register each method under its mangled name
                    // (e.g. `Point_area`) so call-site typechecking can
                    // resolve it. The first parameter type is rewritten to
                    // the struct type (matching what MIR lowering does),
                    // so `p.area()` correctly validates as `Point_area(p)`.
                    for method in &impl_block.methods {
                        let mangled = format!("{}_{}", impl_block.type_name, method.name);
                        let mut params: Vec<Type> = method.params.iter()
                            .map(|p| p.ty.clone())
                            .collect();
                        // Normalize the first param (self) to the struct type
                        if let Some(first) = params.first_mut() {
                            if matches!(first, Type::Inferred) {
                                *first = Type::Named(impl_block.type_name.clone());
                            }
                        }
                        let sig = FunctionSignature {
                            params,
                            return_type: method.return_type.clone().unwrap_or(Type::Void),
                            is_pub: method.is_pub,
                        };
                        self.functions.insert(mangled, sig);
                    }
                    // Record which trait (if any) this impl block claims
                    // to satisfy, for the conformance check in the second
                    // pass and for later trait-typed dispatch.
                    if let Some(trait_name) = &impl_block.trait_name {
                        self.struct_trait_impls
                            .entry(impl_block.type_name.clone())
                            .or_insert_with(std::collections::HashSet::new)
                            .insert(trait_name.clone());
                    }
                }
                _ => {}
            }
        }

        // Fill in inherited trait-default methods that a struct didn't
        // override — register them under the struct's mangled name too,
        // so `p.describe()` resolves even when Point's own impl never
        // wrote a describe() method. This is what makes a trait default
        // act like inherited behavior rather than just documentation.
        let mut inherited_defaults: Vec<(String, FunctionSignature)> = Vec::new();
        for (struct_name, trait_set) in &self.struct_trait_impls {
            for trait_name in trait_set {
                if let Some(trait_methods) = self.traits.get(trait_name) {
                    for tm in trait_methods {
                        if !tm.has_default { continue; }
                        let mangled = format!("{}_{}", struct_name, tm.name);
                        if self.functions.contains_key(&mangled) { continue; }
                        let mut params = tm.param_types.clone();
                        params.insert(0, Type::Named(struct_name.clone()));
                        inherited_defaults.push((mangled, FunctionSignature {
                            params,
                            return_type: tm.return_type.clone(),
                            is_pub: true,
                        }));
                    }
                }
            }
        }
        for (mangled, sig) in inherited_defaults {
            self.functions.insert(mangled, sig);
        }

        // First pass (1b): register top-level functions with
        // trait-typed parameters resolved to TraitObject.
        for item in &program.items {
            if let Item::Function(func) = item {
                let sig = FunctionSignature {
                    params: func.params.iter().map(|p| self.resolve_trait_object_type(&p.ty)).collect(),
                    return_type: func.return_type.clone().unwrap_or(Type::Void),
                    is_pub: true,
                };
                self.functions.insert(func.name.clone(), sig);
            }
        }

        // Second pass: type check everything
        for item in &program.items {
            match item {
                Item::Function(func) => {
                    self.check_function(func)?;
                }
                Item::Impl(impl_block) => {
                    // If this is `impl Trait for Type`, verify every
                    // method the trait requires is actually present with
                    // a matching signature. Without this check, a trait
                    // is purely decorative — a struct could silently omit
                    // a required method and nothing would catch it until
                    // (if ever) someone tried to call it generically.
                    if let Some(trait_name) = &impl_block.trait_name {
                        let required = self.traits.get(trait_name).cloned().ok_or_else(|| {
                            format!("Unknown trait: {}", trait_name)
                        })?;
                        for req in &required {
                            let provided = impl_block.methods.iter().find(|m| m.name == req.name);
                            match provided {
                                None => {
                                    // Not provided explicitly — OK if the
                                    // trait has a default body for this
                                    // method (Quantum's inheritance
                                    // mechanism: structs inherit trait
                                    // defaults instead of being forced to
                                    // override every single method).
                                    if !req.has_default {
                                        return Err(format!(
                                            "`impl {} for {}` is missing required method `{}`. \
                                             The trait `{}` declares `fn {}(...)`, but {} doesn't provide it.",
                                            trait_name, impl_block.type_name, req.name,
                                            trait_name, req.name, impl_block.type_name
                                        ));
                                    }
                                }
                                Some(method) => {
                                    let provided_params: Vec<Type> = method.params.iter()
                                        .skip(1) // skip self
                                        .map(|p| p.ty.clone())
                                        .collect();
                                    let provided_return = method.return_type.clone().unwrap_or(Type::Void);
                                    if provided_params != req.param_types || provided_return != req.return_type {
                                        return Err(format!(
                                            "Method `{}` in `impl {} for {}` doesn't match the signature required by trait `{}`. \
                                             The trait requires it to return `{}`, but {}'s version returns `{}` \
                                             (or its parameter types differ).",
                                            req.name, trait_name, impl_block.type_name, trait_name,
                                            req.return_type, impl_block.type_name, provided_return
                                        ));
                                    }
                                }
                            }
                        }
                    }

                    // Check every method body, with current_struct_context
                    // set so self.field access can be validated (including
                    // the visibility check — see Expr::FieldAccess) against
                    // the struct this impl block belongs to.
                    let prev_ctx = self.current_struct_context.replace(impl_block.type_name.clone());
                    for method in &impl_block.methods {
                        // The first param's type is normalized to the
                        // struct type during signature registration (see
                        // the first pass above) — mirror that here too so
                        // `self.x` inside the body resolves correctly.
                        let mut method_for_check = method.clone();
                        if let Some(first) = method_for_check.params.first_mut() {
                            if matches!(first.ty, Type::Inferred) {
                                first.ty = Type::Named(impl_block.type_name.clone());
                            }
                        }
                        self.check_function(&method_for_check)?;
                    }
                    self.current_struct_context = prev_ctx;
                }
                Item::Const(c) => {
                    let expr_type = self.check_expr(&c.value)?;
                    if !self.types_compatible(&c.ty, &expr_type) {
                        return Err(format!(
                            "Type mismatch in const {}: expected {}, found {}",
                            c.name, c.ty, expr_type
                        ));
                    }
                }
                _ => {}
            }
        }

        Ok(program)
    }

    /// If `ty` is Type::Named(X) and X is a known trait (not a struct),
    /// resolve it to Type::TraitObject(X). Needed for function params
    /// declared as `shape: Shape` — the parser emits Named, but we need
    /// TraitObject so types_compatible can validate struct-to-trait-object
    /// argument passing correctly.
    fn resolve_trait_object_type(&self, ty: &Type) -> Type {
        if let Type::Named(name) = ty {
            if self.traits.contains_key(name) && !self.structs.contains_key(name) {
                return Type::TraitObject(name.clone());
            }
        }
        ty.clone()
    }

    fn check_function(&mut self, func: &Function) -> Result<(), String> {
        self.push_scope();

        // Add parameters to scope
        for param in &func.params {
            let resolved_ty = self.resolve_trait_object_type(&param.ty);
            self.define(param.name.clone(), resolved_ty);
        }

        // Track expected return type so `return <expr>` statements can be
        // validated against it (see Stmt::Return in check_stmt_impl).
        let expected_return = func.return_type.clone().unwrap_or(Type::Void);
        let prev_return_type = self.current_return_type.replace(expected_return.clone());

        // Check body
        let body_type = self.check_block(&func.body)?;

        self.current_return_type = prev_return_type;

        // Be lenient: if body_type is unit/void, explicit `return` stmts handle the real type
        let body_is_unit = matches!(body_type, Type::Void | Type::Inferred);
        let ret_is_flexible = matches!(expected_return, Type::Inferred | Type::Void);
        if !self.types_compatible(&expected_return, &body_type) && !body_is_unit && !ret_is_flexible {
            return Err(format!(
                "Function {} return type mismatch: expected {}, found {}",
                func.name, expected_return, body_type
            ));
        }

        self.pop_scope();
        Ok(())
    }

    /// Like `check_function`, but collects all errors found in the body
    /// (via `check_block_collecting`) instead of stopping at the first.
    fn check_function_collecting(&mut self, func: &Function, errors: &mut Vec<String>) {
        self.push_scope();

        for param in &func.params {
            let resolved_ty = self.resolve_trait_object_type(&param.ty);
            self.define(param.name.clone(), resolved_ty);
        }

        let expected_return = func.return_type.clone().unwrap_or(Type::Void);
        let prev_return_type = self.current_return_type.replace(expected_return.clone());

        let body_type = self.check_block_collecting(&func.body, errors);

        self.current_return_type = prev_return_type;

        let body_is_unit = matches!(body_type, Type::Void | Type::Inferred);
        let ret_is_flexible = matches!(expected_return, Type::Inferred | Type::Void);
        if !self.types_compatible(&expected_return, &body_type) && !body_is_unit && !ret_is_flexible {
            errors.push(format!(
                "Function {} return type mismatch: expected {}, found {}",
                func.name, expected_return, body_type
            ));
        }

        self.pop_scope();
    }

    fn check_block(&mut self, block: &Block) -> Result<Type, String> {
        for stmt in &block.stmts {
            self.check_stmt(stmt)?;
        }

        if let Some(expr) = &block.expr {
            self.check_expr(expr)
        } else {
            Ok(Type::Void)
        }
    }

    /// Like `check_block`, but continues past statement-level errors instead
    /// of stopping at the first one, collecting all of them. Used by
    /// `check_all` for editor diagnostics (multiple squiggles per file)
    /// while `check`/`check_block` retain fail-fast behaviour for the CLI.
    ///
    /// Note: if an earlier statement fails, later statements are still
    /// checked using whatever scope state was established so far — this can
    /// occasionally produce a knock-on error (e.g. a variable whose `let`
    /// failed typechecking won't be in scope for later uses), but in
    /// practice this still surfaces real, independent issues most of the
    /// time and is strictly more informative than stopping at the first.
    fn check_block_collecting(&mut self, block: &Block, errors: &mut Vec<String>) -> Type {
        for stmt in &block.stmts {
            if let Err(e) = self.check_stmt(stmt) {
                errors.push(e);
            }
        }

        if let Some(expr) = &block.expr {
            match self.check_expr(expr) {
                Ok(ty) => ty,
                Err(e) => { errors.push(e); Type::Inferred }
            }
        } else {
            Type::Void
        }
    }

    fn check_stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
        match self.check_stmt_impl(stmt) {
            Err(msg) if !msg.contains("line ") => {
                let span = stmt.span();
                Err(format!("{} at line {}, column {}", msg, span.line, span.column))
            }
            other => other,
        }
    }

    fn check_stmt_impl(&mut self, stmt: &Stmt) -> Result<(), String> {
        match stmt {
            Stmt::Let { name, ty, init, .. } => {
                let actual_type = if let Some(init_expr) = init {
                    self.check_expr(init_expr)?
                } else {
                    ty.clone().unwrap_or(Type::Inferred)
                };

                if let Some(declared_type) = ty {
                    if !self.types_compatible(declared_type, &actual_type) {
                        return Err(format!(
                            "Type mismatch in let {}: declared {}, found {}",
                            name, declared_type, actual_type
                        ));
                    }
                }

                self.define(name.clone(), actual_type);
                Ok(())
            }

            Stmt::Assign { target, value, .. } => {
                // Python-style implicit declaration: `x = 42` where `x` is
                // not yet in scope auto-declares it as a mutable local.
                // We check the value type and define the variable so later
                // uses in the same scope typecheck correctly.
                if let Expr::Ident { name, .. } = target {
                    if self.lookup(name).is_none() {
                        let value_type = self.check_expr(value)?;
                        self.define(name.clone(), value_type);
                        return Ok(());
                    }
                }

                let target_type = self.check_expr(target)?;
                let value_type = self.check_expr(value)?;

                if !self.types_compatible(&target_type, &value_type) {
                    return Err(format!(
                        "Assignment type mismatch: cannot assign {} to {}",
                        value_type, target_type
                    ));
                }
                Ok(())
            }

            Stmt::Expr { expr, .. } => {
                self.check_expr(expr)?;
                Ok(())
            }

            Stmt::Return { value, .. } => {
                let actual_type = if let Some(expr) = value {
                    self.check_expr(expr)?
                } else {
                    Type::Void
                };

                if let Some(expected) = self.current_return_type.clone() {
                    let actual_is_flexible = matches!(actual_type, Type::Inferred);
                    let expected_is_flexible = matches!(expected, Type::Inferred | Type::Void);
                    if !self.types_compatible(&expected, &actual_type) && !actual_is_flexible && !expected_is_flexible {
                        return Err(format!(
                            "Return type mismatch: function expects {}, found {}",
                            expected, actual_type
                        ));
                    }
                }
                Ok(())
            }

            Stmt::While { condition, body, .. } => {
                let cond_type = self.check_expr(condition)?;
                if !self.types_compatible(&Type::Bool, &cond_type) {
                    return Err(format!(
                        "While condition must be bool, found {}",
                        cond_type
                    ));
                }
                self.check_block(body)?;
                Ok(())
            }

            Stmt::DoWhile { body, condition, .. } => {
                // Body is checked first since it always runs at least once,
                // before the condition is ever evaluated.
                self.check_block(body)?;
                let cond_type = self.check_expr(condition)?;
                if !self.types_compatible(&Type::Bool, &cond_type) {
                    return Err(format!(
                        "Do-while condition must be bool, found {}",
                        cond_type
                    ));
                }
                Ok(())
            }

            Stmt::For { var, iter, body, .. } => {
                let iter_type = self.check_expr(iter)?;

                // Infer element type from iterator
                let elem_type = match iter_type {
                    Type::Array(inner, _) => *inner,
                    Type::Slice(inner) => *inner,
                    Type::Named(ref name) if name == "Range" => Type::Int,
                    _ => return Err(format!("Cannot iterate over type {}", iter_type))
                };

                self.push_scope();
                self.define(var.clone(), elem_type);
                self.check_block(body)?;
                self.pop_scope();
                Ok(())
            }

            Stmt::Loop { body, .. } => {
                self.check_block(body)?;
                Ok(())
            }

            Stmt::Break { .. } | Stmt::Continue { .. } => Ok(()),
        }
    }

    fn check_expr(&mut self, expr: &Expr) -> Result<Type, String> {
        match self.check_expr_impl(expr) {
            Err(msg) if !msg.contains("line ") => {
                let span = expr.span();
                Err(format!("{} at line {}, column {}", msg, span.line, span.column))
            }
            other => other,
        }
    }

    fn check_expr_impl(&mut self, expr: &Expr) -> Result<Type, String> {
        match expr {
            Expr::Literal { value, .. } => Ok(self.literal_type(value)),

            Expr::Ident { name, .. } => {
                // Check variable scope FIRST — a local variable or function
                // parameter must always shadow a builtin or global function
                // of the same name (e.g. a parameter named `len` should
                // resolve to the parameter, not the builtin `len` function).
                // Checking builtins/globals first was a real bug: it made
                // `len`, `range`, `min`, `max`, etc. unusable as ordinary
                // parameter or variable names anywhere in their scope.
                if let Some(var_ty) = self.lookup(name) {
                    return Ok(var_ty);
                }

                // Check builtins
                const BUILTINS: &[&str] = &[
                    "print", "println", "eprint", "eprintln",
                    "len", "push", "pop", "range", "assert",
                    "to_string", "parse_int", "parse_float",
                    "sqrt", "abs", "pow", "sin", "cos", "tan",
                    "floor", "ceil", "round", "min", "max",
                    "exit", "panic", "todo", "unreachable",
                    "input", "read_file", "write_file",
                    "qai", "quantumai",
                ];
                if BUILTINS.contains(&name.as_str()) {
                    return Ok(Type::Function(vec![], Box::new(Type::Void)));
                }
                // Check user-defined functions
                if let Some(sig) = self.functions.get(name).cloned() {
                    return Ok(Type::Function(
                        sig.params.clone(),
                        Box::new(sig.return_type.clone()),
                    ));
                }
                Err(format!("Undefined variable: {}", name))
            }

            Expr::Binary { op, left, right, .. } => {
                let left_type = self.check_expr(left)?;
                let right_type = self.check_expr(right)?;

                self.check_binary_op(*op, &left_type, &right_type)
            }

            Expr::Unary { op, expr, .. } => {
                let expr_type = self.check_expr(expr)?;
                self.check_unary_op(*op, &expr_type)
            }

            Expr::NamedArg { value, .. } => {
                // Named argument — strip the name and typecheck the value.
                // Names are positional hints only; they don't change the
                // calling convention (functions don't declare param names
                // in a way the typechecker enforces by name yet).
                self.check_expr(value)
            }
            Expr::StringInterp { parts, .. } => {
                for part in parts {
                    if let crate::ast::StringInterpPart::Expr(e) = part {
                        self.check_expr(e)?;
                    }
                }
                Ok(Type::String)
            }
            Expr::Call { func, args, .. } => {
                let func_type = self.check_expr(func)?;

                // Special case for builtin print
                if let Expr::Ident { name, .. } = func.as_ref() {
                    if name == "print" || name == "println" {
                        // print/println accept any type, but their arguments
                        // must still be checked — otherwise expressions like
                        // `println(undefined_var)` would silently pass.
                        for arg in args {
                            self.check_expr(arg)?;
                        }
                        return Ok(Type::Void);
                    }

                    if let Some(sig) = self.functions.get(name).cloned() {
                        if args.len() != sig.params.len() {
                            return Err(format!(
                                "Function {} expects {} arguments, found {}",
                                name, sig.params.len(), args.len()
                            ));
                        }

                        for (i, arg) in args.iter().enumerate() {
                            let arg_type = self.check_expr(arg)?;
                            if !self.types_compatible(&sig.params[i], &arg_type) {
                                return Err(format!(
                                    "Argument {} type mismatch: expected {}, found {}",
                                    i, sig.params[i], arg_type
                                ));
                            }
                        }

                        return Ok(sig.return_type.clone());
                    }
                }

                match func_type {
                    Type::Function(param_types, return_type) => {
                        if args.len() != param_types.len() {
                            return Err(format!(
                                "Function expects {} arguments, found {}",
                                param_types.len(), args.len()
                            ));
                        }

                        for (i, arg) in args.iter().enumerate() {
                            let arg_type = self.check_expr(arg)?;
                            if !self.types_compatible(&param_types[i], &arg_type) {
                                return Err(format!(
                                    "Argument {} type mismatch",
                                    i
                                ));
                            }
                        }

                        Ok(*return_type)
                    }
                    _ => Err(format!("Cannot call non-function type: {}", func_type))
                }
            }

            Expr::MethodCall { receiver, method, args, .. } => {
                let receiver_type = self.check_expr(receiver)?;
                let _ = args;
                match (receiver_type.clone(), method.as_str()) {
                    // Int methods
                    (Type::Int, "abs") | (Type::I64, "abs")     => Ok(Type::I64),
                    (Type::Int, "to_float") | (Type::I64, "to_float") => Ok(Type::Float),
                    (Type::Int, "to_string") | (Type::I64, "to_string") => Ok(Type::String),
                    (Type::Int, "is_even") | (Type::I64, "is_even") => Ok(Type::Bool),
                    (Type::Int, "is_odd")  | (Type::I64, "is_odd")  => Ok(Type::Bool),
                    (Type::Int, "pow")  => Ok(Type::I64),
                    (Type::Int, "bit_count") => Ok(Type::Int),
                    // Float methods
                    (Type::Float, "to_int")      => Ok(Type::I64),
                    (Type::Float, "to_string")   => Ok(Type::String),
                    (Type::Float, "is_nan")      => Ok(Type::Bool),
                    (Type::Float, "is_infinite") => Ok(Type::Bool),
                    (Type::Float, "is_finite")   => Ok(Type::Bool),
                    (Type::Float, "is_positive") => Ok(Type::Bool),
                    (Type::Float, "is_negative") => Ok(Type::Bool),
                    // Bool methods
                    (Type::Bool, "to_string") => Ok(Type::String),
                    (Type::Bool, "to_int")    => Ok(Type::I64),
                    (Type::Bool, "and")       => Ok(Type::Bool),
                    (Type::Bool, "or")        => Ok(Type::Bool),
                    (Type::Bool, "not")       => Ok(Type::Bool),
                    // Char methods
                    (Type::Char, "to_int")   => Ok(Type::Int),
                    (Type::Char, "to_string")=> Ok(Type::String),
                    (Type::Char, "is_alpha") => Ok(Type::Bool),
                    (Type::Char, "is_digit") => Ok(Type::Bool),
                    (Type::Char, "is_upper") => Ok(Type::Bool),
                    (Type::Char, "is_lower") => Ok(Type::Bool),
                    (Type::Char, "is_space") => Ok(Type::Bool),
                    (Type::Char, "upper")    => Ok(Type::Char),
                    (Type::Char, "lower")    => Ok(Type::Char),
                    // Extra string methods
                    (Type::String, "capitalize") | (Type::String, "title") |
                    (Type::String, "lstrip") | (Type::String, "rstrip") |
                    (Type::String, "center") | (Type::String, "zfill") => Ok(Type::String),
                    (Type::String, "count") | (Type::String, "find") |
                    (Type::String, "rfind") | (Type::String, "eq") => Ok(Type::Int),
                    // String methods
                    (Type::String, "len") => Ok(Type::Int),
                    (Type::String, "upper") | (Type::String, "lower") |
                    (Type::String, "trim") | (Type::String, "replace") |
                    (Type::String, "before") | (Type::String, "after") |
                    (Type::String, "substr") | (Type::String, "repeat") => Ok(Type::String),
                    (Type::String, "contains") | (Type::String, "starts_with") |
                    (Type::String, "ends_with") | (Type::String, "is_empty") => Ok(Type::Bool),
                    (Type::String, "char_at") | (Type::String, "parse_int") => Ok(Type::Int),
                    (Type::String, "parse_float") => Ok(Type::Float),
                    (Type::String, "to_string") => Ok(Type::String),
                    // Number methods
                    (Type::Int, "abs") | (Type::Float, "abs") |
                    (Type::Int, "sqrt") | (Type::Float, "sqrt") |
                    (Type::Int, "floor") | (Type::Float, "floor") |
                    (Type::Int, "ceil") | (Type::Float, "ceil") |
                    (Type::Int, "round") | (Type::Float, "round") |
                    (Type::Int, "pow") | (Type::Float, "pow") |
                    (Type::Int, "min") | (Type::Float, "min") |
                    (Type::Int, "max") | (Type::Float, "max") |
                    (Type::Int, "clamp") | (Type::Float, "clamp") => Ok(Type::Float),
                    (Type::Int, "to_string") | (Type::Float, "to_string") |
                    (Type::Bool, "to_string") => Ok(Type::String),
                    // help/type_of on any type
                    (_, "help") | (_, "type_of") | (_, "type_name") => Ok(Type::String),
                    // Inferred: allow any method
                    (Type::Inferred, _) => Ok(Type::Inferred),
                    (Type::Array(_, _), "len") => Ok(Type::Int),
                    (Type::Slice(_), "len") => Ok(Type::Int),
                    (Type::Int, "to_string") | (Type::Float, "to_string")
                    | (Type::Bool, "to_string") | (Type::String, "to_string")
                    | (Type::Inferred, "to_string") => Ok(Type::String),
                    // Result<T,E> methods
                    (Type::Result(ok_ty, _), "unwrap") => Ok(*ok_ty),
                    (Type::Result(ok_ty, _), "expect") => Ok(*ok_ty),
                    (Type::Result(ok_ty, _), "unwrap_or") => Ok(*ok_ty),
                    (Type::Result(_, _), "is_ok") => Ok(Type::Bool),
                    (Type::Result(_, _), "is_err") => Ok(Type::Bool),
                    // Option<T> methods
                    (Type::Option(inner_ty), "unwrap") => Ok(*inner_ty),
                    (Type::Option(inner_ty), "expect") => Ok(*inner_ty),
                    (Type::Option(inner_ty), "unwrap_or") => Ok(*inner_ty),
                    (Type::Option(_), "is_some") => Ok(Type::Bool),
                    (Type::Option(_), "is_none") => Ok(Type::Bool),
                    (Type::TraitObject(trait_name), _) => {
                        // Method call on a trait-object-typed param —
                        // resolve return type from the trait's contract.
                        let trait_methods = self.traits.get(&trait_name).ok_or_else(|| {
                            format!("Unknown trait: {}", trait_name)
                        })?;
                        let method_sig = trait_methods.iter().find(|m| m.name == *method)
                            .ok_or_else(|| {
                                format!("Trait `{}` has no method `{}`", trait_name, method)
                            })?;
                        Ok(method_sig.return_type.clone())
                    }
                    (Type::Named(struct_name), _) => {
                        let mangled = format!("{}_{}", struct_name, method);
                        if let Some(sig) = self.functions.get(&mangled) {
                            // Visibility check: a private method can only
                            // be called from within the same struct's own
                            // impl block (i.e. one method calling another
                            // private helper method), not externally.
                            let accessing_from_own_impl = self.current_struct_context
                                .as_deref() == Some(struct_name.as_str());
                            if !sig.is_pub && !accessing_from_own_impl {
                                return Err(format!(
                                    "Method `.{}()` on struct `{}` is private and cannot be called from outside its own methods. \
                                     Remove `private` from `fn {}` in the impl block if external calls are intended.",
                                    method, struct_name, method
                                ));
                            }
                            Ok(sig.return_type.clone())
                        } else {
                            Err(format!("Method .{}() not found on struct {}", method, struct_name))
                        }
                    }
                    _ => Err(format!("Method .{}() not found on type {}", method, receiver_type))
                }
            }

            Expr::FieldAccess { expr, field, .. } => {
                let expr_type = self.check_expr(expr)?;

                match expr_type {
                    Type::Named(struct_name) => {
                        if let Some(struct_info) = self.structs.get(&struct_name) {
                            let field_ty = struct_info.fields.get(field)
                                .cloned()
                                .ok_or_else(|| {
                                    format!("Field {} not found on struct {}", field, struct_name)
                                })?;

                            // Visibility check: a private (non-pub) field
                            // can only be accessed from within that same
                            // struct's own impl block (i.e. while checking
                            // a method whose current_struct_context matches),
                            // not from arbitrary external code.
                            let is_private = !struct_info.pub_fields.contains(field);
                            let accessing_from_own_impl = self.current_struct_context
                                .as_deref() == Some(struct_name.as_str());
                            if is_private && !accessing_from_own_impl {
                                return Err(format!(
                                    "Field `{}` of struct `{}` is private and cannot be accessed from outside its own methods. \
                                     Remove `private` from `{}: {}` in the struct definition if external access is intended, \
                                     or add a method on {} that returns it.",
                                    field, struct_name, field, field_ty, struct_name
                                ));
                            }

                            Ok(match field_ty { Type::TypeParam(_) => Type::Inferred, other => other })
                        } else {
                            Err(format!("Unknown struct: {}", struct_name))
                        }
                    }
                    // Result<T,E> fields: is_ok (bool), value (T), error (E)
                    Type::Result(ok_ty, err_ty) => {
                        match field.as_str() {
                            "is_ok" => Ok(Type::Bool),
                            "value" => Ok(*ok_ty.clone()),
                            "error" => Ok(*err_ty.clone()),
                            _ => Err(format!(
                                "`Result` has fields `is_ok` (bool), `value`, and `error`. \
                                 There is no field named `{}`. \
                                 Hint: check `r.is_ok` first, then read `r.value` or `r.error`.",
                                field
                            ))
                        }
                    }
                    // Option<T> fields: has_value (bool), value (T)
                    Type::Option(inner_ty) => {
                        match field.as_str() {
                            "has_value" => Ok(Type::Bool),
                            "value" => Ok(*inner_ty.clone()),
                            _ => Err(format!(
                                "`Option` has fields `has_value` (bool) and `value`. \
                                 There is no field named `{}`. \
                                 Hint: check `opt.has_value` first, then read `opt.value`.",
                                field
                            ))
                        }
                    }
                    _ => Err(format!("Cannot access field on non-struct type: {}", expr_type))
                }
            }

            Expr::Index { expr, index, .. } => {
                let expr_type = self.check_expr(expr)?;
                let index_type = self.check_expr(index)?;

                if !self.types_compatible(&Type::Int, &index_type) {
                    return Err(format!("Array index must be int, found {}", index_type));
                }

                match expr_type {
                    Type::Array(elem_type, _) => Ok(*elem_type),
                    Type::Slice(elem_type) => Ok(*elem_type),
                    _ => Err(format!("Cannot index into type: {}", expr_type))
                }
            }

            Expr::If { condition, then_block, else_block, .. } => {
                let cond_type = self.check_expr(condition)?;
                if !self.types_compatible(&Type::Bool, &cond_type) {
                    return Err(format!("If condition must be bool, found {}", cond_type));
                }

                let then_type = self.check_block(then_block)?;

                if let Some(else_blk) = else_block {
                    let else_type = self.check_block(else_blk)?;
                    if !self.types_compatible(&then_type, &else_type) {
                        return Err(format!(
                            "If branches have incompatible types: {} and {}",
                            then_type, else_type
                        ));
                    }
                }

                Ok(then_type)
            }

            Expr::Ternary { condition, then_expr, else_expr, .. } => {
                let cond_type = self.check_expr(condition)?;
                if !self.types_compatible(&Type::Bool, &cond_type) {
                    return Err(format!("Ternary condition must be bool, found {}", cond_type));
                }

                let then_type = self.check_expr(then_expr)?;
                let else_type = self.check_expr(else_expr)?;

                if !self.types_compatible(&then_type, &else_type) {
                    return Err(format!(
                        "Ternary branches have incompatible types: {} and {}",
                        then_type, else_type
                    ));
                }

                Ok(then_type)
            }

            Expr::Cast { expr, target_type, .. } => {
                let source_type = self.check_expr(expr)?;
                // Only allow casts between numeric types (int <-> float and
                // their sized variants) — casting strings, bools, or
                // arrays doesn't have an obvious safe meaning at the C
                // level the way numeric truncation/widening does.
                let is_numeric = |t: &Type| matches!(t,
                    Type::Int | Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::I128
                    | Type::UInt | Type::U8 | Type::U16 | Type::U32 | Type::U64 | Type::U128
                    | Type::Float | Type::F32 | Type::F64
                    | Type::Inferred
                );
                if !is_numeric(&source_type) || !is_numeric(target_type) {
                    return Err(format!(
                        "Cannot cast {} to {} — only numeric type conversions (e.g. int as float) are supported",
                        source_type, target_type
                    ));
                }
                Ok(target_type.clone())
            }

            Expr::Some { value, .. } => {
                let inner_ty = self.check_expr(value)?;
                Ok(Type::Option(Box::new(inner_ty)))
            }

            Expr::None { .. } => {
                // None has type Option<Inferred> — the actual inner type
                // is resolved by context (e.g. the function's return type
                // annotation). This is a deliberate simplification; full
                // type inference would need a constraint solver.
                Ok(Type::Option(Box::new(Type::Inferred)))
            }

            Expr::Ok { value, .. } => {
                let inner_ty = self.check_expr(value)?;
                // The error type is Inferred — callers provide it through
                // the function's declared return type annotation.
                Ok(Type::Result(Box::new(inner_ty), Box::new(Type::Inferred)))
            }

            Expr::Err { value, .. } => {
                let inner_ty = self.check_expr(value)?;
                Ok(Type::Result(Box::new(Type::Inferred), Box::new(inner_ty)))
            }

            Expr::Try { expr, span } => {
                let expr_ty = self.check_expr(expr)?;
                // `expr?` unwraps a Result/Option, returning early on
                // failure. The resulting type is the inner Ok/Some type.
                match expr_ty {
                    Type::Result(ok_ty, _) => Ok(*ok_ty),
                    Type::Option(inner_ty) => Ok(*inner_ty),
                    other => Err(format!(
                        "The `?` operator can only be used on `Result` or `Option` values, \
                         but found `{}` at line {}, column {}. \
                         Hint: `?` propagates errors — wrap your value in `Ok(...)` or `Some(...)` first.",
                        other, span.line, span.column
                    ))
                }
            }

            Expr::StructInit { name, fields, .. } => {
                let struct_info = self.structs.get(name).cloned().ok_or_else(|| {
                    format!("Unknown struct: {}", name)
                })?;

                // Check every provided field exists on the struct and has
                // the right type, and collect which fields were provided.
                let is_generic = struct_info.fields.values()
                    .any(|t| matches!(t, Type::TypeParam(_)));
                let mut provided = std::collections::HashSet::new();
                for (field_name, field_expr) in fields {
                    if !struct_info.fields.contains_key(field_name) {
                        return Err(format!("Struct {} has no field named {}", name, field_name));
                    }
                    let actual_ty = self.check_expr(field_expr)?;
                    if !is_generic {
                        let expected_ty = struct_info.fields[field_name].clone();
                        if !self.types_compatible(&expected_ty, &actual_ty) {
                            return Err(format!(
                                "Field {} of struct {} expects {}, found {}",
                                field_name, name, expected_ty, actual_ty
                            ));
                        }
                    }
                    provided.insert(field_name.clone());
                }
                // Every declared field must be initialized — Quantum has
                // no default values or Default-trait equivalent yet, so a
                // partially-initialized struct would leave some C struct
                // fields with garbage/uninitialized memory.
                for declared_field in struct_info.fields.keys() {
                    if !provided.contains(declared_field) {
                        return Err(format!(
                            "Missing field {} in initializer for struct {}",
                            declared_field, name
                        ));
                    }
                }

                Ok(Type::Named(name.clone()))
            }

            Expr::Match { expr, arms, .. } => {
                let expr_type = self.check_expr(expr)?;

                if arms.is_empty() {
                    return Err("Match expression must have at least one arm".to_string());
                }

                let first_type = self.check_expr(&arms[0].body)?;

                for arm in &arms[1..] {
                    let arm_type = self.check_expr(&arm.body)?;
                    if !self.types_compatible(&first_type, &arm_type) {
                        return Err("Match arms have incompatible types".to_string());
                    }
                }

                Ok(first_type)
            }

            Expr::Array { elements, .. } => {
                if elements.is_empty() {
                    return Ok(Type::Array(Box::new(Type::Inferred), Some(0)));
                }

                let first_type = self.check_expr(&elements[0])?;

                for elem in &elements[1..] {
                    let elem_type = self.check_expr(elem)?;
                    if !self.types_compatible(&first_type, &elem_type) {
                        return Err("Array elements have incompatible types".to_string());
                    }
                }

                Ok(Type::Array(Box::new(first_type), Some(elements.len())))
            }

            Expr::Tuple { elements, .. } => {
                let types: Result<Vec<_>, _> = elements.iter()
                    .map(|e| self.check_expr(e))
                    .collect();
                Ok(Type::Tuple(types?))
            }

            Expr::Block(block) => self.check_block(block),

            Expr::Lambda { params, body, .. } => {
                self.push_scope();

                for param in params {
                    let resolved_ty = self.resolve_trait_object_type(&param.ty);
            self.define(param.name.clone(), resolved_ty);
                }

                let return_type = self.check_expr(body)?;

                self.pop_scope();

                let param_types = params.iter().map(|p| p.ty.clone()).collect();
                Ok(Type::Function(param_types, Box::new(return_type)))
            }

            Expr::Range { start, end, inclusive, .. } => {
                let start_type = self.check_expr(start)?;
                let end_type = self.check_expr(end)?;

                if !self.types_compatible(&Type::Int, &start_type) ||
                   !self.types_compatible(&Type::Int, &end_type) {
                    return Err("Range bounds must be integers".to_string());
                }

                Ok(Type::Named("Range".to_string()))
            }

            Expr::Await { expr, .. } => {
                self.check_expr(expr)
            }
        }
    }

    fn literal_type(&self, lit: &Literal) -> Type {
        match lit {
            Literal::Int(_) => Type::Int,
            Literal::Float(_) => Type::Float,
            Literal::String(_) => Type::String,
            Literal::Char(_) => Type::Char,
            Literal::Bool(_) => Type::Bool,
            Literal::Null => Type::Option(Box::new(Type::Inferred)),
        }
    }

    fn check_binary_op(&self, op: BinOp, left: &Type, right: &Type) -> Result<Type, String> {
        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::Power => {
                // String concatenation: `+` between two strings produces a
                // string (codegen lowers this to q_strcat). Also allow one
                // side to be Inferred (an untyped Python-style parameter) —
                // we assume string concat intent since String is incompatible
                // with numeric ops anyway.
                if matches!(op, BinOp::Add)
                    && (matches!(left, Type::String) || matches!(right, Type::String))
                    && (matches!(left, Type::String | Type::Inferred)
                        && matches!(right, Type::String | Type::Inferred))
                {
                    return Ok(Type::String);
                }
                if self.is_numeric(left) && self.types_compatible(left, right) {
                    Ok(left.clone())
                } else {
                    Err(format!("Invalid operands for {:?}: {} and {}", op, left, right))
                }
            }

            BinOp::Eq | BinOp::Ne => Ok(Type::Bool),

            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                if self.is_numeric(left) && self.types_compatible(left, right) {
                    Ok(Type::Bool)
                } else {
                    Err(format!("Invalid comparison: {} and {}", left, right))
                }
            }

            BinOp::And | BinOp::Or => {
                if self.types_compatible(&Type::Bool, left) &&
                   self.types_compatible(&Type::Bool, right) {
                    Ok(Type::Bool)
                } else {
                    Err(format!("Logical operators require bool operands"))
                }
            }

            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor |
            BinOp::Lshift | BinOp::Rshift => {
                if self.is_integer(left) && self.types_compatible(left, right) {
                    Ok(left.clone())
                } else {
                    Err(format!("Bitwise operators require integer operands"))
                }
            }

            BinOp::Range | BinOp::RangeInclusive => {
                if self.types_compatible(&Type::Int, left) &&
                   self.types_compatible(&Type::Int, right) {
                    Ok(Type::Named("Range".to_string()))
                } else {
                    Err("Range bounds must be integers".to_string())
                }
            }
        }
    }

    fn check_unary_op(&self, op: UnOp, expr_type: &Type) -> Result<Type, String> {
        match op {
            UnOp::Neg => {
                if self.is_numeric(expr_type) {
                    Ok(expr_type.clone())
                } else {
                    Err(format!("Cannot negate non-numeric type: {}", expr_type))
                }
            }
            UnOp::Not => {
                if self.types_compatible(&Type::Bool, expr_type) {
                    Ok(Type::Bool)
                } else {
                    Err(format!("Cannot apply ! to non-bool type: {}", expr_type))
                }
            }
            UnOp::BitNot => {
                if self.is_integer(expr_type) {
                    Ok(expr_type.clone())
                } else {
                    Err(format!("Bitwise not requires integer type: {}", expr_type))
                }
            }
        }
    }

    fn types_compatible(&self, expected: &Type, actual: &Type) -> bool {
        match (expected, actual) {
            (Type::Inferred, _) | (_, Type::Inferred) => true,
            (Type::Void, _) | (_, Type::Void) => true,
            (Type::TypeParam(_), _) | (_, Type::TypeParam(_)) => true,
            // Integer widening
            (Type::Int, Type::I8 | Type::I16 | Type::I32 | Type::I64) => true,
            (Type::I8 | Type::I16 | Type::I32 | Type::I64, Type::Int) => true,
            // Float widening
            (Type::Float, Type::F32 | Type::F64) => true,
            (Type::F32 | Type::F64, Type::Float) => true,
            // String aliases (string == String == Str) — both sides must
            // actually be strings. The previous rule here was vacuously
            // true whenever EITHER side was String (e.g. `let x: int =
            // "hello"` was accepted), since it tested the same condition
            // the outer match arm already guaranteed.
            (Type::String, Type::String) => true,
            // Function types are flexible
            (Type::Function(_, _), _) | (_, Type::Function(_, _)) => true,
            // Bool is compatible with int for conditions
            (Type::Bool, Type::Int) | (Type::Int, Type::Bool) => true,
            (Type::Int, Type::I64) | (Type::I64, Type::Int) => true,
            // String -> i64: strings are const char* cast to int64 internally,
            // so this cast is safe and needed for passing strings to C externs.
            (Type::I64, Type::String) | (Type::String, Type::I64) => true,
            // A struct value is compatible with a trait-typed parameter
            // if that struct has a verified `impl Trait for Struct` block.
            (Type::TraitObject(trait_name), Type::Named(struct_name)) => {
                self.struct_trait_impls.get(struct_name)
                    .map(|traits| traits.contains(trait_name))
                    .unwrap_or(false)
            }
            // Option<T> — None (Option<Inferred>) is compatible with any Option<T>.
            // Option<A> is compatible with Option<B> if A is compatible with B.
            (Type::Option(exp_inner), Type::Option(act_inner)) => {
                self.types_compatible(exp_inner, act_inner)
            }
            // Result<T,E> — Ok(Inferred)/Err(Inferred) compatible with any Result.
            (Type::Result(exp_ok, exp_err), Type::Result(act_ok, act_err)) => {
                self.types_compatible(exp_ok, act_ok) && self.types_compatible(exp_err, act_err)
            }
            _ => expected == actual
        }
    }

    fn is_numeric(&self, ty: &Type) -> bool {
        matches!(ty,
            Type::Int | Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::I128 |
            Type::UInt | Type::U8 | Type::U16 | Type::U32 | Type::U64 | Type::U128 |
            Type::Float | Type::F32 | Type::F64 |
            // Inferred = untyped parameter (Python style) — allow arithmetic;
            // the MIR/codegen defaults these to int64_t which handles most cases.
            Type::Inferred
        )
    }

    fn is_integer(&self, ty: &Type) -> bool {
        matches!(ty,
            Type::Int | Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::I128 |
            Type::UInt | Type::U8 | Type::U16 | Type::U32 | Type::U64 | Type::U128
        )
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn define(&mut self, name: String, ty: Type) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, ty);
        }
    }

    fn lookup(&self, name: &str) -> Option<Type> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty.clone());
            }
        }
        None
    }
}
