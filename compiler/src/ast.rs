// Quantum AST - Abstract Syntax Tree definitions
use crate::lexer::Span;
use std::fmt;

pub type NodeId = usize;

#[derive(Debug, Clone)]
pub struct Program {
    pub items: Vec<Item>,
    /// True when `#[safe_mode]` appears anywhere in the source — wraps
    /// the compiled main() in a crash handler at the C level.
    pub safe_mode: bool,
}

#[derive(Debug, Clone)]
pub enum Item {
    Function(Function),
    Struct(Struct),
    Enum(Enum),
    Trait(Trait),
    Impl(Impl),
    Import(Import),
    Const(Const),
    Model(Model),
    ExternBlock(ExternBlock),
}

/// An `extern "C" { ... }` block — declares the signatures of functions
/// implemented in an external C library, without providing a body. The
/// compiler emits a matching C forward declaration (no definition) and
/// trusts the linker to resolve the symbol from a linked library (see
/// `#link "name"` directives or `--link <name>` CLI flag).
#[derive(Debug, Clone)]
pub struct ExternBlock {
    /// ABI string — currently only "C" is meaningful, but the syntax
    /// allows for future ABIs.
    pub abi: String,
    pub functions: Vec<ExternFunction>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ExternFunction {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<Type>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub id: NodeId,
    pub name: String,
    /// Generic type parameters, e.g. `<T>` or `<T, U>` in
    /// `fn identity<T>(x: T) -> T`. Empty for non-generic functions.
    /// Quantum compiles generics via monomorphization (like C++/Rust) —
    /// a separate concrete function is generated per type this function
    /// is actually called with, rather than any runtime generic mechanism.
    pub type_params: Vec<String>,
    pub params: Vec<Param>,
    pub return_type: Option<Type>,
    pub body: Block,
    pub is_pub: bool,
    pub is_async: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: Type,
    pub mutable: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub expr: Option<Box<Expr>>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let {
        name: String,
        ty: Option<Type>,
        init: Option<Expr>,
        mutable: bool,
        span: Span,
    },
    Assign {
        target: Expr,
        value: Expr,
        span: Span,
    },
    Expr {
        expr: Expr,
        span: Span,
    },
    Return {
        value: Option<Expr>,
        span: Span,
    },
    Break {
        span: Span,
    },
    Continue {
        span: Span,
    },
    While {
        condition: Expr,
        body: Block,
        span: Span,
    },
    For {
        var: String,
        iter: Expr,
        body: Block,
        span: Span,
    },
    Loop {
        body: Block,
        span: Span,
    },
    /// `do { ... } while cond` — like `while`, but the body always runs
    /// at least once before the condition is ever checked.
    DoWhile {
        body: Block,
        condition: Expr,
        span: Span,
    },
}

#[derive(Debug, Clone)]
pub enum Expr {
    Literal {
        value: Literal,
        span: Span,
    },
    StringInterp {
        parts: Vec<StringInterpPart>,
        span: Span,
    },
    Ident {
        name: String,
        span: Span,
    },
    Binary {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
        span: Span,
    },
    Unary {
        op: UnOp,
        expr: Box<Expr>,
        span: Span,
    },
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
        span: Span,
    },
    MethodCall {
        receiver: Box<Expr>,
        method: String,
        args: Vec<Expr>,
        span: Span,
    },
    FieldAccess {
        expr: Box<Expr>,
        field: String,
        span: Span,
    },
    Index {
        expr: Box<Expr>,
        index: Box<Expr>,
        span: Span,
    },
    If {
        condition: Box<Expr>,
        then_block: Block,
        else_block: Option<Block>,
        span: Span,
    },
    /// `condition ? then_expr : else_expr` — a compact expression-level
    /// conditional, distinct from `If` (which holds full Blocks and is
    /// statement-shaped). Always has both branches, since a ternary
    /// without an else clause wouldn't make sense as an expression value.
    Ternary {
        condition: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
        span: Span,
    },
    /// `expr as Type` — an explicit numeric cast (e.g. `x as float`,
    /// `y as int`). Quantum has no implicit int/float promotion (mixing
    /// them in an expression is a type error), so this is the only way to
    /// convert between numeric types.
    /// `Point { x: 10, y: 20 }` — struct literal construction.
    StructInit {
        name: String,
        fields: Vec<(String, Expr)>,
        span: Span,
    },
    /// A keyword argument in a function call: `fit(x, y, epochs: 100)`.
    /// The name must match one of the callee's declared parameter names.
    /// Reordered to match positional order during the keyword-arg
    /// resolution pass (after parsing, before typechecking).
    NamedArg { name: String, value: Box<Expr>, span: Span },
    Some { value: Box<Expr>, span: Span },
    /// `None` — an empty Option
    None { span: Span },
    /// `Ok(value)` — a successful Result
    Ok { value: Box<Expr>, span: Span },
    /// `Err(error)` — a failed Result
    Err { value: Box<Expr>, span: Span },
    /// `expr?` — propagate error: if expr is Err(e) or None, return early
    Try { expr: Box<Expr>, span: Span },
    Cast {
        expr: Box<Expr>,
        target_type: Type,
        span: Span,
    },
    Match {
        expr: Box<Expr>,
        arms: Vec<MatchArm>,
        span: Span,
    },
    Array {
        elements: Vec<Expr>,
        span: Span,
    },
    Tuple {
        elements: Vec<Expr>,
        span: Span,
    },
    Block(Block),
    Lambda {
        params: Vec<Param>,
        body: Box<Expr>,
        span: Span,
    },
    Range {
        start: Box<Expr>,
        end: Box<Expr>,
        inclusive: bool,
        span: Span,
    },
    Await {
        expr: Box<Expr>,
        span: Span,
    },
}

impl Stmt {
    pub fn span(&self) -> Span {
        match self {
            Stmt::Let { span, .. } => *span,
            Stmt::Assign { span, .. } => *span,
            Stmt::Expr { span, .. } => *span,
            Stmt::Return { span, .. } => *span,
            Stmt::Break { span } => *span,
            Stmt::Continue { span } => *span,
            Stmt::While { span, .. } => *span,
            Stmt::For { span, .. } => *span,
            Stmt::Loop { span, .. } => *span,
            Stmt::DoWhile { span, .. } => *span,
        }
    }
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Literal { span, .. } => *span,
            Expr::StringInterp { span, .. } => *span,
            Expr::Ident { span, .. } => *span,
            Expr::Binary { span, .. } => *span,
            Expr::Unary { span, .. } => *span,
            Expr::Call { span, .. } => *span,
            Expr::MethodCall { span, .. } => *span,
            Expr::FieldAccess { span, .. } => *span,
            Expr::Index { span, .. } => *span,
            Expr::If { span, .. } => *span,
            Expr::Ternary { span, .. } => *span,
            Expr::Cast { span, .. } => *span,
            Expr::StructInit { span, .. } => *span,
            Expr::Some { span, .. } => *span,
            Expr::NamedArg { span, .. } => *span,
            Expr::None { span } => *span,
            Expr::Ok { span, .. } => *span,
            Expr::Err { span, .. } => *span,
            Expr::Try { span, .. } => *span,
            Expr::Match { span, .. } => *span,
            Expr::Array { span, .. } => *span,
            Expr::Tuple { span, .. } => *span,
            Expr::Block(block) => block.span,
            Expr::Lambda { span, .. } => *span,
            Expr::Range { span, .. } => *span,
            Expr::Await { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: Expr,
}

#[derive(Debug, Clone)]
pub enum Pattern {
    Wildcard,
    Literal(Literal),
    Ident(String),
    Tuple(Vec<Pattern>),
    Struct { name: String, fields: Vec<String> },
    Or(Vec<Pattern>),  // 1 | 2 | 3
}

#[derive(Debug, Clone)]
pub enum StringInterpPart {
    Literal(String),
    Expr(Box<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    Float(f64),
    String(String),
    Char(char),
    Bool(bool),
    Null,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Power,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Lshift,
    Rshift,
    Range,
    RangeInclusive,
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::Mod => write!(f, "%"),
            BinOp::Power => write!(f, "**"),
            BinOp::Eq => write!(f, "=="),
            BinOp::Ne => write!(f, "!="),
            BinOp::Lt => write!(f, "<"),
            BinOp::Le => write!(f, "<="),
            BinOp::Gt => write!(f, ">"),
            BinOp::Ge => write!(f, ">="),
            BinOp::And => write!(f, "&&"),
            BinOp::Or => write!(f, "||"),
            BinOp::BitAnd => write!(f, "&"),
            BinOp::BitOr => write!(f, "|"),
            BinOp::BitXor => write!(f, "^"),
            BinOp::Lshift => write!(f, "<<"),
            BinOp::Rshift => write!(f, ">>"),
            BinOp::Range => write!(f, ".."),
            BinOp::RangeInclusive => write!(f, "..="),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnOp {
    Neg,
    Not,
    BitNot,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Int,
    I8, I16, I32, I64, I128,
    UInt,
    U8, U16, U32, U64, U128,
    Float,
    F32, F64,
    Bool,
    Char,
    String,
    Void,
    Array(Box<Type>, Option<usize>),
    Slice(Box<Type>),
    Tuple(Vec<Type>),
    Function(Vec<Type>, Box<Type>),
    Named(String),
    /// A trait used as a type — e.g. `fn describe(shape: Shape)` where
    /// `Shape` is a trait. Lowered to a vtable struct in C codegen.
    TraitObject(String),
    Generic(String, Vec<Type>),
    /// A reference to a generic type parameter inside a generic
    /// function's signature/body, e.g. `T` in `fn identity<T>(x: T) -> T`.
    /// Resolved to a concrete type at each monomorphized call site.
    TypeParam(String),
    Reference(Box<Type>, bool), // type, mutable
    Option(Box<Type>),
    Result(Box<Type>, Box<Type>),
    Inferred,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Type::Int => write!(f, "int"),
            Type::I8 => write!(f, "i8"),
            Type::I16 => write!(f, "i16"),
            Type::I32 => write!(f, "i32"),
            Type::I64 => write!(f, "i64"),
            Type::I128 => write!(f, "i128"),
            Type::UInt => write!(f, "uint"),
            Type::U8 => write!(f, "u8"),
            Type::U16 => write!(f, "u16"),
            Type::U32 => write!(f, "u32"),
            Type::U64 => write!(f, "u64"),
            Type::U128 => write!(f, "u128"),
            Type::Float => write!(f, "float"),
            Type::F32 => write!(f, "f32"),
            Type::F64 => write!(f, "f64"),
            Type::Bool => write!(f, "bool"),
            Type::Char => write!(f, "char"),
            Type::String => write!(f, "String"),
            Type::Void => write!(f, "()"),
            Type::Array(t, Some(n)) => write!(f, "[{}; {}]", t, n),
            Type::Array(t, None) => write!(f, "[{}]", t),
            Type::Slice(t) => write!(f, "[{}]", t),
            Type::Tuple(types) => {
                write!(f, "(")?;
                for (i, t) in types.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", t)?;
                }
                write!(f, ")")
            },
            Type::Function(params, ret) => {
                write!(f, "fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", ret)
            },
            Type::Named(name) => write!(f, "{}", name),
            Type::TraitObject(name) => write!(f, "dyn {}", name),
            Type::Generic(name, args) => {
                write!(f, "{}<", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", arg)?;
                }
                write!(f, ">")
            },
            Type::TypeParam(name) => write!(f, "{}", name),
            Type::Reference(t, false) => write!(f, "&{}", t),
            Type::Reference(t, true) => write!(f, "&mut {}", t),
            Type::Option(t) => write!(f, "Option<{}>", t),
            Type::Result(ok, err) => write!(f, "Result<{}, {}>", ok, err),
            Type::Inferred => write!(f, "_"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Struct {
    pub id: NodeId,
    pub name: String,
    pub type_params: Vec<String>,
    pub fields: Vec<StructField>,
    pub is_pub: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct StructField {
    pub name: String,
    pub ty: Type,
    pub is_pub: bool,
}

#[derive(Debug, Clone)]
pub struct Enum {
    pub id: NodeId,
    pub name: String,
    pub variants: Vec<EnumVariant>,
    pub is_pub: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct EnumVariant {
    pub name: String,
    pub fields: Option<Vec<Type>>,
}

#[derive(Debug, Clone)]
pub struct Trait {
    pub id: NodeId,
    pub name: String,
    pub methods: Vec<TraitMethod>,
    pub is_pub: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct TraitMethod {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<Type>,
    /// If present, this is a *default* implementation — any struct that
    /// implements this trait gets this method for free unless its own
    /// `impl Trait for Struct` block overrides it with its own version.
    /// This is Quantum's answer to inheritance: instead of classical
    /// class hierarchies, behavior is shared via trait defaults.
    pub default_body: Option<Block>,
}

#[derive(Debug, Clone)]
pub struct Impl {
    pub id: NodeId,
    pub trait_name: Option<String>,
    pub type_name: String,
    pub methods: Vec<Function>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Import {
    pub path: Vec<String>,
    pub items: Option<Vec<String>>,
    pub alias: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Const {
    pub name: String,
    pub ty: Type,
    pub value: Expr,
    pub is_pub: bool,
    pub span: Span,
}

// AI-specific AST nodes
#[derive(Debug, Clone)]
pub struct Model {
    pub id: NodeId,
    pub name: String,
    pub layers: Vec<Layer>,
    pub forward: Function,
    pub is_pub: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub name: String,
    pub layer_type: Expr,
}
