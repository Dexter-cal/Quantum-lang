// Quantum Parser - Recursive descent parser with operator precedence
use crate::ast::*;
use crate::lexer::{Token, TokenKind, Span};
use std::collections::HashMap;

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
    next_id: NodeId,
    /// When true, suppresses struct-literal parsing (`Ident { field: ... }`)
    /// at the current position. Needed because `if cond { ... }` and
    /// `while cond { ... }` are ambiguous with `if Point { x: 1 } { ... }`
    /// otherwise — the `{` after a bare identifier could be either a
    /// struct literal or the start of the if/while body. Set to true while
    /// parsing an `if`/`while`/`for`-in-range condition, restored
    /// afterward.
    no_struct_literal: bool,
    /// Names currently in scope as generic type parameters (e.g. {"T"}
    /// while parsing the signature/body of `fn identity<T>(x: T) -> T`).
    /// parse_type checks this to emit Type::TypeParam instead of
    /// Type::Named for a bare identifier matching one of these names.
    active_type_params: Vec<String>,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            current: 0,
            next_id: 0,
            no_struct_literal: false,
            active_type_params: Vec::new(),
        }
    }

    fn next_id(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn parse(&mut self) -> Result<Program, String> {
        let mut items = Vec::new();

        while !self.is_at_end() {
            items.push(self.parse_item()?);
        }

        Ok(Program { items, safe_mode: false })
    }

    /// Like `parse`, but collects multiple parse errors rather than stopping
    /// at the first. Returns a (possibly partial) Program plus all errors.
    pub fn parse_recovering(&mut self, errors: &mut Vec<String>) -> Program {
        let mut items = Vec::new();
        while !self.is_at_end() {
            match self.parse_item() {
                Ok(item) => items.push(item),
                Err(e) => {
                    errors.push(e);
                    // Skip to next top-level item boundary
                    loop {
                        if self.is_at_end() { break; }
                        match self.peek().kind {
                            TokenKind::Fn | TokenKind::Struct
                            | TokenKind::Enum | TokenKind::Pub => break,
                            _ => { self.advance(); }
                        }
                    }
                }
            }
        }
        Program { items, safe_mode: false }
    }

    fn parse_item(&mut self) -> Result<Item, String> {
        let is_pub = self.match_token(&[TokenKind::Pub]);

        match &self.peek().kind {
            TokenKind::Fn => Ok(Item::Function(self.parse_function(is_pub)?)),
            TokenKind::Struct => Ok(Item::Struct(self.parse_struct(is_pub)?)),
            TokenKind::Enum => Ok(Item::Enum(self.parse_enum(is_pub)?)),
            TokenKind::Trait => Ok(Item::Trait(self.parse_trait(is_pub)?)),
            TokenKind::Impl => Ok(Item::Impl(self.parse_impl()?)),
            TokenKind::Import | TokenKind::From => Ok(Item::Import(self.parse_import()?)),
            TokenKind::Const => Ok(Item::Const(self.parse_const(is_pub)?)),
            TokenKind::Extern => Ok(Item::ExternBlock(self.parse_extern_block()?)),
            _ => Err(format!("Expected item, found {:?}", self.peek().kind))
        }
    }

    /// Parse `extern "C" { fn name(params) -> RetType ... }`.
    /// Each function inside has no body — just a signature — since it's
    /// implemented externally (in a linked C library).
    fn parse_extern_block(&mut self) -> Result<ExternBlock, String> {
        let start = self.advance(); // `extern`

        // ABI string, e.g. "C". Defaults to "C" if omitted.
        let abi = if let TokenKind::String(s) = &self.peek().kind {
            let s = s.clone();
            self.advance();
            s
        } else {
            "C".to_string()
        };

        self.expect(&TokenKind::LBrace)?;
        let mut functions = Vec::new();

        while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
            self.expect(&TokenKind::Fn)?;
            let fname = self.expect_ident()?;
            let fn_start = self.previous().span;

            self.expect(&TokenKind::LParen)?;
            let params = self.parse_params()?;
            self.expect(&TokenKind::RParen)?;

            let return_type = if self.match_token(&[TokenKind::Arrow]) {
                Some(self.parse_type()?)
            } else {
                None
            };

            // No body — extern functions are declarations only. Allow an
            // optional trailing semicolon for readability/familiarity with
            // C/Rust extern syntax.
            self.match_token(&[TokenKind::Semicolon]);

            functions.push(ExternFunction {
                name: fname,
                params,
                return_type,
                span: fn_start,
            });
        }

        let end = self.expect(&TokenKind::RBrace)?;

        Ok(ExternBlock {
            abi,
            functions,
            span: Span {
                start: start.span.start,
                end: end.span.end,
                line: start.span.line,
                column: start.span.column,
            },
        })
    }

    fn parse_function(&mut self, is_pub: bool) -> Result<Function, String> {
        let start = self.advance(); // fn
        let name = self.expect_ident()?;

        // Optional generic type parameters: `fn identity<T>(x: T) -> T`.
        // While active, parse_type recognizes bare identifiers matching
        // these names as Type::TypeParam instead of Type::Named.
        let type_params = if self.check(&TokenKind::Lt) {
            self.advance();
            let mut params = Vec::new();
            while !self.check(&TokenKind::Gt) && !self.is_at_end() {
                params.push(self.expect_ident()?);
                if !self.match_token(&[TokenKind::Comma]) { break; }
            }
            self.expect(&TokenKind::Gt)?;
            params
        } else {
            Vec::new()
        };
        let prev_type_params = std::mem::replace(&mut self.active_type_params, type_params.clone());

        self.expect(&TokenKind::LParen)?;
        let params = self.parse_params()?;
        self.expect(&TokenKind::RParen)?;

        let return_type = if self.match_token(&[TokenKind::Arrow]) {
            Some(self.parse_type()?)
        } else {
            None
        };

        let body = self.parse_block()?;
        let end = body.span;
        self.active_type_params = prev_type_params;

        Ok(Function {
            id: self.next_id(),
            name,
            type_params,
            params,
            return_type,
            body,
            is_pub,
            is_async: false,
            span: Span { start: start.span.start, end: end.end, line: start.span.line, column: start.span.column },
        })
    }

    fn parse_params(&mut self) -> Result<Vec<Param>, String> {
        let mut params = Vec::new();

        while !self.check(&TokenKind::RParen) && !self.is_at_end() {
            let mutable = self.match_token(&[TokenKind::Mut]);
            // Accept `self` keyword as a parameter name — it's a reserved
            // token, not an identifier, so expect_ident() won't accept it.
            // Rename it to `self_` internally to avoid C keyword conflicts.
            let name = if self.check(&TokenKind::SelfKw) {
                self.advance();
                "self_".to_string()
            } else {
                self.expect_ident()?
            };

            // Type annotation is optional — `name: Type` or just `name`.
            // Without a type, the parameter gets Type::Inferred and the
            // typechecker / MIR will resolve it from usage context.
            let ty = if self.match_token(&[TokenKind::Colon]) {
                self.parse_type()?
            } else {
                crate::ast::Type::Inferred
            };

            params.push(Param {
                name,
                ty,
                mutable,
                span: self.previous().span,
            });

            if !self.match_token(&[TokenKind::Comma]) {
                break;
            }
        }

        Ok(params)
    }

    fn parse_block(&mut self) -> Result<Block, String> {
        let start = self.expect(&TokenKind::LBrace)?;
        let mut stmts = Vec::new();
        let mut expr = None;

        while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
            if self.check(&TokenKind::RBrace) {
                break;
            }

            // Check if this is the final expression (no semicolon)
            let stmt_start = self.current;
            let stmt_or_expr = self.parse_stmt()?;

            match stmt_or_expr {
                Stmt::Expr { expr: e, .. } if !self.match_token(&[TokenKind::Semicolon]) => {
                    // Don't treat print/println calls as trailing expressions —
                    // they are always statements. Also skip if next token is another statement.
                    let is_print_call = matches!(&e,
                        Expr::Call { func, .. } if matches!(func.as_ref(),
                            Expr::Ident { name, .. } if name == "print" || name == "println"
                                || name == "eprint" || name == "eprintln"
                        )
                    );
                    // Detect if the NEXT token starts a new statement, so the
                    // current expression should be treated as a statement
                    // (not the block's trailing expression). This covers:
                    //   - another function call: `foo()`
                    //   - an assignment (with or without `let`): `x = ...`
                    //     or `result = add(1,2)` (Python-style implicit let)
                    let next_is_call = match self.peek().kind.clone() {
                        TokenKind::Ident(_) => self.tokens.get(self.current + 1)
                            .map(|t| matches!(t.kind, TokenKind::LParen | TokenKind::Dot))
                            .unwrap_or(false),
                        _ => false,
                    };
                    let next_is_assign = match self.peek().kind.clone() {
                        TokenKind::Ident(_) => self.tokens.get(self.current + 1)
                            .map(|t| matches!(t.kind, TokenKind::Eq | TokenKind::PlusEq
                                | TokenKind::MinusEq | TokenKind::StarEq | TokenKind::SlashEq))
                            .unwrap_or(false),
                        _ => false,
                    };
                    let next_is_stmt = next_is_call
                        || next_is_assign
                        || self.check(&TokenKind::Print)
                        || self.check(&TokenKind::Println)
                        || self.check(&TokenKind::Let)
                        || self.check(&TokenKind::Return)
                        || self.check(&TokenKind::If)
                        || self.check(&TokenKind::While)
                        || self.check(&TokenKind::For)
                        || self.check(&TokenKind::Loop);
                    // NOTE: deliberately do NOT include RBrace here. Reaching
                    // `}` next is exactly the signal that the current
                    // expression IS the block's trailing value (e.g. the
                    // `100` in `{ 100 }`), not a statement to be discarded.
                    // Treating RBrace as a "next is a statement" signal was a
                    // bug that broke every value-producing block (if-as-
                    // expression, ternary, function bodies with non-void
                    // trailing expressions) by always pushing the trailing
                    // expression into `stmts` instead of `block.expr`.
                    if is_print_call || next_is_stmt {
                        let span = e.span();
                        stmts.push(Stmt::Expr { expr: e, span });
                    } else {
                        expr = Some(Box::new(e));
                        break;
                    }
                }
                s => {
                    stmts.push(s);
                    self.match_token(&[TokenKind::Semicolon]);
                }
            }
        }

        let end = self.expect(&TokenKind::RBrace)?;

        Ok(Block {
            stmts,
            expr,
            span: Span { start: start.span.start, end: end.span.end, line: start.span.line, column: start.span.column },
        })
    }

    /// Like `parse_block`, but attempts error recovery at the statement level:
    /// if `parse_stmt` fails, the error is saved, tokens are skipped until
    /// a statement-start boundary (next `let`/`return`/`if`/etc. or `}`),
    /// and parsing continues from there. This lets the LSP collect multiple
    /// parse errors per file instead of stopping at the first.
    ///
    /// Returns the (possibly partial) Block and all errors encountered.
    pub fn parse_block_recovering(
        &mut self,
        errors: &mut Vec<String>,
    ) -> Result<Block, String> {
        let start = self.expect(&TokenKind::LBrace)?;
        let mut stmts = Vec::new();
        let mut expr = None;

        while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
            match self.parse_stmt() {
                Ok(stmt_or_expr) => {
                    match stmt_or_expr {
                        Stmt::Expr { expr: e, .. }
                            if !self.match_token(&[TokenKind::Semicolon]) =>
                        {
                            let is_print_call = matches!(&e,
                                Expr::Call { func, .. } if matches!(func.as_ref(),
                                    Expr::Ident { name, .. }
                                        if name == "print" || name == "println"
                                            || name == "eprint" || name == "eprintln"
                                )
                            );
                            let next_is_call = match self.peek().kind.clone() {
                                TokenKind::Ident(_) => self.tokens.get(self.current + 1)
                                    .map(|t| matches!(t.kind, TokenKind::LParen | TokenKind::Dot))
                                    .unwrap_or(false),
                                _ => false,
                            };
                            let next_is_assign = match self.peek().kind.clone() {
                                TokenKind::Ident(_) => self.tokens.get(self.current + 1)
                                    .map(|t| matches!(t.kind, TokenKind::Eq | TokenKind::PlusEq
                                        | TokenKind::MinusEq | TokenKind::StarEq | TokenKind::SlashEq))
                                    .unwrap_or(false),
                                _ => false,
                            };
                            let next_is_stmt = next_is_call
                                || next_is_assign
                                || self.check(&TokenKind::Print)
                                || self.check(&TokenKind::Println)
                                || self.check(&TokenKind::Let)
                                || self.check(&TokenKind::Return)
                                || self.check(&TokenKind::If)
                                || self.check(&TokenKind::While)
                                || self.check(&TokenKind::For)
                                || self.check(&TokenKind::Loop);
                            // (RBrace deliberately excluded — see parse_block's
                            // comment on the same heuristic for why.)
                            if is_print_call || next_is_stmt {
                                let span = e.span();
                                stmts.push(Stmt::Expr { expr: e, span });
                            } else {
                                expr = Some(Box::new(e));
                                break;
                            }
                        }
                        s => {
                            stmts.push(s);
                            self.match_token(&[TokenKind::Semicolon]);
                        }
                    }
                }
                Err(e) => {
                    errors.push(e);
                    // Synchronize: skip tokens until we reach something that
                    // looks like a fresh statement start or the end of the
                    // block. This prevents a cascading error storm from a
                    // single malformed statement.
                    self.synchronize_to_stmt_boundary();
                }
            }
        }

        let end = self.expect(&TokenKind::RBrace)?;
        Ok(Block {
            stmts,
            expr,
            span: Span {
                start: start.span.start,
                end: end.span.end,
                line: start.span.line,
                column: start.span.column,
            },
        })
    }

    /// Skip tokens until we reach a token that is likely to be the start of
    /// the next statement (or the `}` that closes the current block). Used
    /// by `parse_block_recovering` to re-synchronize after a parse error.
    fn synchronize_to_stmt_boundary(&mut self) {
        loop {
            if self.is_at_end() { break; }
            match self.peek().kind {
                // These tokens reliably start a new statement
                TokenKind::Let
                | TokenKind::Return
                | TokenKind::If
                | TokenKind::While
                | TokenKind::For
                | TokenKind::Loop
                | TokenKind::Break
                | TokenKind::Continue
                | TokenKind::Fn
                | TokenKind::Struct
                | TokenKind::Enum
                // `}` closes the current block — stop here so the caller
                // can parse it as the block's closing delimiter
                | TokenKind::RBrace => break,
                _ => { self.advance(); }
            }
        }
    }

    fn parse_stmt(&mut self) -> Result<Stmt, String> {
        match &self.peek().kind {
            TokenKind::Let => self.parse_let(),
            TokenKind::Return => self.parse_return(),
            TokenKind::Break => {
                let span = self.advance().span;
                Ok(Stmt::Break { span })
            }
            TokenKind::Continue => {
                let span = self.advance().span;
                Ok(Stmt::Continue { span })
            }
            TokenKind::While => self.parse_while(),
            TokenKind::For => self.parse_for(),
            TokenKind::Loop => self.parse_loop(),
            TokenKind::Do => self.parse_do_while(),
            TokenKind::Print | TokenKind::Println => {
                let expr = self.parse_print()?;
                let span = expr.span();
                Ok(Stmt::Expr { expr, span })
            }
            _ => {
                let expr = self.parse_expr()?;
                let span = expr.span();

                // Check for assignment
                if self.match_token(&[TokenKind::Eq]) {
                    let value = self.parse_expr()?;
                    Ok(Stmt::Assign { target: expr, value, span })
                } else if let Some(op) = self.compound_assign_op() {
                    // Desugar `x += 5` into `x = x + 5` at the AST level —
                    // this means compound assignment gets MIR/codegen
                    // support for free, since it reuses the already-
                    // working Stmt::Assign + Expr::Binary lowering paths
                    // rather than needing any new machinery of its own.
                    self.advance(); // consume the compound-assign token
                    let rhs = self.parse_expr()?;
                    let binary_span = Span {
                        start: expr.span().start,
                        end: rhs.span().end,
                        line: expr.span().line,
                        column: expr.span().column,
                    };
                    let value = Expr::Binary {
                        op,
                        left: Box::new(expr.clone()),
                        right: Box::new(rhs),
                        span: binary_span,
                    };
                    Ok(Stmt::Assign { target: expr, value, span })
                } else {
                    Ok(Stmt::Expr { expr, span })
                }
            }
        }
    }

    /// Returns the corresponding BinOp if the current token is a compound
    /// assignment operator (`+=`, `-=`, `*=`, `/=`), without consuming it.
    fn compound_assign_op(&self) -> Option<BinOp> {
        match &self.peek().kind {
            TokenKind::PlusEq => Some(BinOp::Add),
            TokenKind::MinusEq => Some(BinOp::Sub),
            TokenKind::StarEq => Some(BinOp::Mul),
            TokenKind::SlashEq => Some(BinOp::Div),
            _ => None,
        }
    }

    fn parse_let(&mut self) -> Result<Stmt, String> {
        let start = self.advance(); // let
        let mutable = self.match_token(&[TokenKind::Mut]);
        let name = self.expect_ident()?;

        let ty = if self.match_token(&[TokenKind::Colon]) {
            Some(self.parse_type()?)
        } else {
            None
        };

        let init = if self.match_token(&[TokenKind::Eq]) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        Ok(Stmt::Let {
            name,
            ty,
            init,
            mutable,
            span: start.span,
        })
    }

    fn parse_return(&mut self) -> Result<Stmt, String> {
        let start = self.advance(); // return

        let value = if !self.check(&TokenKind::Semicolon) && !self.check(&TokenKind::RBrace) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        Ok(Stmt::Return { value, span: start.span })
    }

    fn parse_while(&mut self) -> Result<Stmt, String> {
        let start = self.advance(); // while
        let prev_flag = self.no_struct_literal;
        self.no_struct_literal = true;
        let condition = self.parse_expr()?;
        self.no_struct_literal = prev_flag;
        let body = self.parse_block()?;

        Ok(Stmt::While {
            condition,
            body,
            span: start.span,
        })
    }

    fn parse_for(&mut self) -> Result<Stmt, String> {
        let start = self.advance(); // for
        let var = self.expect_ident()?;
        self.expect(&TokenKind::In)?;
        let prev_flag = self.no_struct_literal;
        self.no_struct_literal = true;
        let iter = self.parse_expr()?;
        self.no_struct_literal = prev_flag;
        let body = self.parse_block()?;

        Ok(Stmt::For {
            var,
            iter,
            body,
            span: start.span,
        })
    }

    fn parse_loop(&mut self) -> Result<Stmt, String> {
        let start = self.advance(); // loop
        let body = self.parse_block()?;

        Ok(Stmt::Loop {
            body,
            span: start.span,
        })
    }

    fn parse_do_while(&mut self) -> Result<Stmt, String> {
        let start = self.advance(); // do
        let body = self.parse_block()?;
        self.expect(&TokenKind::While)?;
        // Same struct-literal ambiguity as if/while/for conditions —
        // see the no_struct_literal field doc comment.
        let prev_flag = self.no_struct_literal;
        self.no_struct_literal = true;
        let condition = self.parse_expr()?;
        self.no_struct_literal = prev_flag;

        Ok(Stmt::DoWhile {
            body,
            condition,
            span: start.span,
        })
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        let expr = self.parse_binary_expr(0)?;

        // Ternary: `condition ? then_expr : else_expr`. Checked after a
        // full binary expression so `a > b ? x : y` parses the comparison
        // as the condition, not `a > (b ? x : y)`.
        if self.check(&TokenKind::Question) {
            self.advance(); // `?`
            let then_expr = self.parse_expr()?;
            self.expect(&TokenKind::Colon)?;
            let else_expr = self.parse_expr()?;
            let span = Span {
                start: expr.span().start,
                end: else_expr.span().end,
                line: expr.span().line,
                column: expr.span().column,
            };
            return Ok(Expr::Ternary {
                condition: Box::new(expr),
                then_expr: Box::new(then_expr),
                else_expr: Box::new(else_expr),
                span,
            });
        }

        Ok(expr)
    }

    fn parse_binary_expr(&mut self, min_precedence: u8) -> Result<Expr, String> {
        let mut left = self.parse_unary_expr()?;

        while let Some(precedence) = self.get_precedence(&self.peek().kind) {
            if precedence < min_precedence {
                break;
            }

            let op_token = self.advance();
            let op = self.token_to_binop(&op_token.kind)?;
            let right = self.parse_binary_expr(precedence + 1)?;

            let span = Span {
                start: left.span().start,
                end: right.span().end,
                line: left.span().line,
                column: left.span().column,
            };

            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
                span,
            };
        }

        Ok(left)
    }

    fn parse_unary_expr(&mut self) -> Result<Expr, String> {
        match &self.peek().kind {
            TokenKind::Minus | TokenKind::Not | TokenKind::Tilde => {
                let op_token = self.advance();
                let op = match op_token.kind {
                    TokenKind::Minus => UnOp::Neg,
                    TokenKind::Not => UnOp::Not,
                    TokenKind::Tilde => UnOp::BitNot,
                    _ => unreachable!(),
                };

                let expr = self.parse_unary_expr()?;
                let span = op_token.span;

                Ok(Expr::Unary {
                    op,
                    expr: Box::new(expr),
                    span,
                })
            }
            _ => self.parse_postfix_expr()
        }
    }

    fn parse_postfix_expr(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_primary_expr()?;

        loop {
            match &self.peek().kind {
                TokenKind::LParen => {
                    self.advance();
                    let args = self.parse_call_args()?;
                    self.expect(&TokenKind::RParen)?;

                    let span = expr.span();
                    expr = Expr::Call {
                        func: Box::new(expr),
                        args,
                        span,
                    };
                }
                TokenKind::Dot => {
                    self.advance();
                    let field = self.expect_ident()?;

                    // Check if it's a method call
                    if self.check(&TokenKind::LParen) {
                        self.advance();
                        let args = self.parse_call_args()?;
                        self.expect(&TokenKind::RParen)?;

                        let span = expr.span();
                        expr = Expr::MethodCall {
                            receiver: Box::new(expr),
                            method: field,
                            args,
                            span,
                        };
                    } else {
                        let span = expr.span();
                        expr = Expr::FieldAccess {
                            expr: Box::new(expr),
                            field,
                            span,
                        };
                    }
                }
                TokenKind::LBracket => {
                    self.advance();
                    let index = self.parse_expr()?;
                    self.expect(&TokenKind::RBracket)?;

                    let span = expr.span();
                    expr = Expr::Index {
                        expr: Box::new(expr),
                        index: Box::new(index),
                        span,
                    };
                }
                TokenKind::As => {
                    self.advance();
                    let target_type = self.parse_type()?;
                    let span = expr.span();
                    expr = Expr::Cast {
                        expr: Box::new(expr),
                        target_type,
                        span,
                    };
                }
                TokenKind::Question => {
                    // `expr?` — propagate errors/None early. If the
                    // expression is Err(e) or None, return that from
                    // the current function; otherwise unwrap the value.
                    self.advance();
                    let span = expr.span();
                    expr = Expr::Try { expr: Box::new(expr), span };
                }
                _ => break
            }
        }

        Ok(expr)
    }

    fn parse_primary_expr(&mut self) -> Result<Expr, String> {
        let token = self.peek().clone();

        match &token.kind {
            TokenKind::Int(n) => {
                self.advance();
                Ok(Expr::Literal {
                    value: Literal::Int(*n),
                    span: token.span,
                })
            }
            TokenKind::Float(f) => {
                self.advance();
                Ok(Expr::Literal {
                    value: Literal::Float(*f),
                    span: token.span,
                })
            }
            TokenKind::String(s) => {
                self.advance();
                Ok(Expr::Literal {
                    value: Literal::String(s.clone()),
                    span: token.span,
                })
            }
            TokenKind::FString(raw) => {
                let raw = raw.clone(); let span = token.span; self.advance();
                let mut parts = Vec::new();
                let mut chars = raw.chars().peekable();
                let mut lit = String::new();
                while let Some(ch) = chars.next() {
                    if ch == '{' && chars.peek() != Some(&'{') {
                        if !lit.is_empty() { parts.push(crate::ast::StringInterpPart::Literal(lit.clone())); lit.clear(); }
                        let mut expr_src = String::new(); let mut depth=1;
                        for ec in chars.by_ref() {
                            if ec=='{' { depth+=1; expr_src.push(ec); }
                            else if ec=='}' { depth-=1; if depth==0 { break; } expr_src.push(ec); }
                            else { expr_src.push(ec); }
                        }
                        let expr_src = expr_src.trim().to_string();
                        if expr_src.is_empty() { parts.push(crate::ast::StringInterpPart::Literal("{}".into())); }
                        else {
                            match crate::lexer::Lexer::new(&expr_src).tokenize() {
                                Ok(toks) => { let mut sp = Parser::new(toks);
                                    match sp.parse_expr() {
                                        Ok(e) => parts.push(crate::ast::StringInterpPart::Expr(Box::new(e))),
                                        Err(_) => parts.push(crate::ast::StringInterpPart::Literal(format!("{{{}}}", expr_src))),
                                    }
                                }
                                Err(_) => parts.push(crate::ast::StringInterpPart::Literal(format!("{{{}}}", expr_src))),
                            }
                        }
                    } else { if ch=='{' { chars.next(); lit.push('{'); } else if ch=='}' && chars.peek()==Some(&'}') { chars.next(); lit.push('}'); } else { lit.push(ch); } }
                }
                if !lit.is_empty() { parts.push(crate::ast::StringInterpPart::Literal(lit)); }
                Ok(Expr::StringInterp { parts, span })
            }
            TokenKind::Char(c) => {
                self.advance();
                Ok(Expr::Literal {
                    value: Literal::Char(*c),
                    span: token.span,
                })
            }
            TokenKind::SelfKw => {
                // `self` inside a method body — emit as an identifier
                // named `self_` (the name used by the MIR desugaring) so
                // `self.x` parses as `Ident("self_").FieldAccess("x")`.
                self.advance();
                Ok(Expr::Ident {
                    name: "self_".to_string(),
                    span: token.span,
                })
            }
            TokenKind::SomeKw => {
                self.advance();
                self.expect(&TokenKind::LParen)?;
                let value = self.parse_expr()?;
                self.expect(&TokenKind::RParen)?;
                Ok(Expr::Some { value: Box::new(value), span: token.span })
            }
            TokenKind::NoneKw => {
                self.advance();
                Ok(Expr::None { span: token.span })
            }
            TokenKind::OkKw => {
                self.advance();
                self.expect(&TokenKind::LParen)?;
                let value = self.parse_expr()?;
                self.expect(&TokenKind::RParen)?;
                Ok(Expr::Ok { value: Box::new(value), span: token.span })
            }
            TokenKind::ErrKw => {
                self.advance();
                self.expect(&TokenKind::LParen)?;
                let value = self.parse_expr()?;
                self.expect(&TokenKind::RParen)?;
                Ok(Expr::Err { value: Box::new(value), span: token.span })
            }
            TokenKind::True => {
                self.advance();
                Ok(Expr::Literal {
                    value: Literal::Bool(true),
                    span: token.span,
                })
            }
            TokenKind::False => {
                self.advance();
                Ok(Expr::Literal {
                    value: Literal::Bool(false),
                    span: token.span,
                })
            }
            TokenKind::Null => {
                self.advance();
                Ok(Expr::Literal {
                    value: Literal::Null,
                    span: token.span,
                })
            }
            TokenKind::Ident(name) => {
                self.advance();
                // Struct literal: `Point { x: 10, y: 20 }`. Only attempted
                // when not in a no-struct-literal context (if/while/for
                // conditions — see the field doc comment on
                // no_struct_literal) and when the `{` is actually followed
                // by a field-init pattern (`ident :`), so a plain block
                // expression or a name immediately followed by an
                // unrelated `{` doesn't get misparsed.
                if !self.no_struct_literal && self.check(&TokenKind::LBrace) {
                    let looks_like_struct_init = matches!(
                        (self.tokens.get(self.current + 1).map(|t| &t.kind),
                         self.tokens.get(self.current + 2).map(|t| &t.kind)),
                        (Some(TokenKind::Ident(_)), Some(TokenKind::Colon))
                    ) || matches!(
                        self.tokens.get(self.current + 1).map(|t| &t.kind),
                        Some(TokenKind::RBrace)
                    );
                    if looks_like_struct_init {
                        return self.parse_struct_init(name.clone(), token.span);
                    }
                }
                Ok(Expr::Ident {
                    name: name.clone(),
                    span: token.span,
                })
            }
            TokenKind::If => self.parse_if_expr(),
            TokenKind::Match => self.parse_match_expr(),
            TokenKind::LParen => self.parse_paren_or_tuple(),
            TokenKind::LBracket => self.parse_array(),
            TokenKind::LBrace => Ok(Expr::Block(self.parse_block()?)),
            TokenKind::Print | TokenKind::Println => self.parse_print(),
            _ => Err(format!("Unexpected token: {:?} at line {}, column {}", token.kind, token.span.line, token.span.column))
        }
    }

    /// Parse the `{ field: expr, ... }` portion of a struct literal, given
    /// the struct name and the span of its name token (already consumed).
    fn parse_struct_init(&mut self, name: String, start_span: Span) -> Result<Expr, String> {
        self.expect(&TokenKind::LBrace)?;
        let mut fields = Vec::new();

        while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
            let field_name = self.expect_ident()?;
            self.expect(&TokenKind::Colon)?;
            let value = self.parse_expr()?;
            fields.push((field_name, value));
            self.match_token(&[TokenKind::Comma]);
        }

        let end = self.expect(&TokenKind::RBrace)?;

        Ok(Expr::StructInit {
            name,
            fields,
            span: Span {
                start: start_span.start,
                end: end.span.end,
                line: start_span.line,
                column: start_span.column,
            },
        })
    }

    fn parse_if_expr(&mut self) -> Result<Expr, String> {
        let start = self.advance(); // if
        let prev_flag = self.no_struct_literal;
        self.no_struct_literal = true;
        let condition = Box::new(self.parse_expr()?);
        self.no_struct_literal = prev_flag;
        let then_block = self.parse_block()?;

        let else_block = if self.match_token(&[TokenKind::Else]) {
            Some(self.parse_block()?)
        } else {
            None
        };

        Ok(Expr::If {
            condition,
            then_block,
            else_block,
            span: start.span,
        })
    }

    fn parse_match_expr(&mut self) -> Result<Expr, String> {
        let start = self.advance(); // match
        let expr = Box::new(self.parse_expr()?);

        self.expect(&TokenKind::LBrace)?;
        let mut arms = Vec::new();

        while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
            let pattern = self.parse_pattern()?;
            let guard = if self.match_token(&[TokenKind::If]) {
                Some(self.parse_expr()?)
            } else {
                None
            };

            self.expect(&TokenKind::FatArrow)?;
            let body = self.parse_expr()?;

            arms.push(MatchArm { pattern, guard, body });

            self.match_token(&[TokenKind::Comma]);
        }

        self.expect(&TokenKind::RBrace)?;

        Ok(Expr::Match {
            expr,
            arms,
            span: start.span,
        })
    }

    fn parse_pattern(&mut self) -> Result<Pattern, String> {
        // Parse a single pattern, then check for | (OR pattern)
        let first = self.parse_single_pattern()?;
        if self.check(&TokenKind::Pipe) {
            let mut alts = vec![first];
            while self.match_token(&[TokenKind::Pipe]) {
                alts.push(self.parse_single_pattern()?);
            }
            // Represent OR patterns as nested Or pattern or just use first
            // For now, emit as a chain — codegen handles Or via multiple arms
            Ok(Pattern::Or(alts))
        } else {
            Ok(first)
        }
    }

    fn parse_single_pattern(&mut self) -> Result<Pattern, String> {
        match &self.peek().kind.clone() {
            TokenKind::Ident(name) if name == "_" => {
                self.advance();
                Ok(Pattern::Wildcard)
            }
            TokenKind::Ident(name) if name == "true" => {
                self.advance();
                Ok(Pattern::Literal(Literal::Bool(true)))
            }
            TokenKind::Ident(name) if name == "false" => {
                self.advance();
                Ok(Pattern::Literal(Literal::Bool(false)))
            }
            TokenKind::True => {
                self.advance();
                Ok(Pattern::Literal(Literal::Bool(true)))
            }
            TokenKind::False => {
                self.advance();
                Ok(Pattern::Literal(Literal::Bool(false)))
            }
            TokenKind::Ident(name) => {
                let name = name.clone();
                self.advance();
                // Check for struct pattern: Name { field, ... }
                if self.check(&TokenKind::LBrace) {
                    self.advance();
                    let mut fields = Vec::new();
                    while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
                        if let TokenKind::Ident(f) = &self.peek().kind.clone() {
                            let f = f.clone();
                            self.advance();
                            fields.push(f);
                            self.match_token(&[TokenKind::Comma]);
                        } else { break; }
                    }
                    self.expect(&TokenKind::RBrace)?;
                    Ok(Pattern::Struct { name, fields })
                } else {
                    Ok(Pattern::Ident(name))
                }
            }
            TokenKind::Int(n) => {
                let n = *n;
                self.advance();
                Ok(Pattern::Literal(Literal::Int(n)))
            }
            TokenKind::Float(f) => {
                let f = *f;
                self.advance();
                Ok(Pattern::Literal(Literal::Float(f)))
            }
            TokenKind::String(s) => {
                let s = s.clone();
                self.advance();
                Ok(Pattern::Literal(Literal::String(s)))
            }
            TokenKind::Char(c) => {
                let c = *c;
                self.advance();
                Ok(Pattern::Literal(Literal::Char(c)))
            }
            // Negative numbers: -42
            TokenKind::Minus => {
                self.advance();
                match &self.peek().kind.clone() {
                    TokenKind::Int(n) => {
                        let n = -n;
                        self.advance();
                        Ok(Pattern::Literal(Literal::Int(n)))
                    }
                    TokenKind::Float(f) => {
                        let f = -f;
                        self.advance();
                        Ok(Pattern::Literal(Literal::Float(f)))
                    }
                    _ => Err("Expected number after - in pattern".to_string())
                }
            }
            // Range pattern: 1..=5
            _ => Err(format!("Expected pattern, found {} — valid patterns: integer, float, string, bool, identifier, _",
                self.peek().kind.to_string()))
        }
    }

    fn parse_paren_or_tuple(&mut self) -> Result<Expr, String> {
        let start = self.advance(); // (

        if self.check(&TokenKind::RParen) {
            self.advance();
            return Ok(Expr::Tuple {
                elements: vec![],
                span: start.span,
            });
        }

        let first = self.parse_expr()?;

        if self.match_token(&[TokenKind::Comma]) {
            let mut elements = vec![first];

            while !self.check(&TokenKind::RParen) {
                elements.push(self.parse_expr()?);
                if !self.match_token(&[TokenKind::Comma]) {
                    break;
                }
            }

            self.expect(&TokenKind::RParen)?;
            Ok(Expr::Tuple { elements, span: start.span })
        } else {
            self.expect(&TokenKind::RParen)?;
            Ok(first)
        }
    }

    fn parse_array(&mut self) -> Result<Expr, String> {
        let start = self.advance(); // [
        let mut elements = Vec::new();

        while !self.check(&TokenKind::RBracket) && !self.is_at_end() {
            elements.push(self.parse_expr()?);
            if !self.match_token(&[TokenKind::Comma]) {
                break;
            }
        }

        self.expect(&TokenKind::RBracket)?;

        Ok(Expr::Array { elements, span: start.span })
    }

    fn parse_print(&mut self) -> Result<Expr, String> {
        let start = self.peek().clone();
        // Determine which keyword: "print" or "println"
        let fn_name = match &start.kind {
            TokenKind::Println => "println",
            TokenKind::Print => "print",
            _ => "println",
        };
        self.advance(); // consume print/println token
        self.expect(&TokenKind::LParen)?;
        let args = self.parse_call_args()?;
        self.expect(&TokenKind::RParen)?;

        Ok(Expr::Call {
            func: Box::new(Expr::Ident {
                name: fn_name.to_string(),
                span: start.span,
            }),
            args,
            span: start.span,
        })
    }

    fn parse_call_args(&mut self) -> Result<Vec<Expr>, String> {
        let mut args = Vec::new();

        while !self.check(&TokenKind::RParen) && !self.is_at_end() {
            // Detect `name: expr` keyword argument syntax.
            // Must be Ident followed immediately by Colon (not `::`)
            // to avoid ambiguity with type annotations and module paths.
            let is_kwarg = if let TokenKind::Ident(_) = &self.peek().kind {
                matches!(self.peek_ahead(1).map(|t| &t.kind), Some(TokenKind::Colon))
            } else {
                false
            };

            if is_kwarg {
                let span = self.peek().span;
                let name = if let TokenKind::Ident(n) = &self.peek().kind { n.clone() } else { unreachable!() };
                self.advance(); // consume ident
                self.advance(); // consume colon
                let value = self.parse_expr()?;
                args.push(Expr::NamedArg { name, value: Box::new(value), span });
            } else {
                args.push(self.parse_expr()?);
            }

            if !self.match_token(&[TokenKind::Comma]) {
                break;
            }
        }

        Ok(args)
    }

    fn parse_type(&mut self) -> Result<Type, String> {
        match &self.peek().kind {
            TokenKind::Ident(name) => {
                let name = name.clone();
                self.advance();

                match name.as_str() {
                    "int" => Ok(Type::Int),
                    "i8" => Ok(Type::I8),
                    "i16" => Ok(Type::I16),
                    "i32" => Ok(Type::I32),
                    "i64" => Ok(Type::I64),
                    "i128" => Ok(Type::I128),
                    "uint" => Ok(Type::UInt),
                    "u8" => Ok(Type::U8),
                    "u16" => Ok(Type::U16),
                    "u32" => Ok(Type::U32),
                    "u64" => Ok(Type::U64),
                    "u128" => Ok(Type::U128),
                    "float" => Ok(Type::Float),
                    "f32" => Ok(Type::F32),
                    "f64" => Ok(Type::F64),
                    "bool" => Ok(Type::Bool),
                    "char" => Ok(Type::Char),
                    "string" | "String" | "str" => Ok(Type::String),
                    "void" => Ok(Type::Void),
                    "Option" => {
                        // Option<T> — optional value, either Some(T) or None
                        self.expect(&TokenKind::Lt)?;
                        let inner = self.parse_type()?;
                        self.expect(&TokenKind::Gt)?;
                        Ok(Type::Option(Box::new(inner)))
                    }
                    "Result" => {
                        // Result<T, E> — success (Ok(T)) or failure (Err(E))
                        self.expect(&TokenKind::Lt)?;
                        let ok_ty = self.parse_type()?;
                        self.expect(&TokenKind::Comma)?;
                        let err_ty = self.parse_type()?;
                        self.expect(&TokenKind::Gt)?;
                        Ok(Type::Result(Box::new(ok_ty), Box::new(err_ty)))
                    }
                    _ => {
                        if self.active_type_params.iter().any(|p| p == &name) {
                            Ok(Type::TypeParam(name))
                        } else {
                            Ok(Type::Named(name))
                        }
                    }
                }
            }
            TokenKind::LBracket => {
                self.advance();
                let inner = Box::new(self.parse_type()?);
                // `[T; N]` — fixed-size array type (used for function
                // parameters/return types, e.g. `fn f(arr: [int; 5])`).
                // Without the `; N` part, `[T]` is a Slice (unsized).
                if self.match_token(&[TokenKind::Semicolon]) {
                    let size_tok = self.peek().kind.clone();
                    let size = if let TokenKind::Int(n) = size_tok {
                        self.advance();
                        n as usize
                    } else {
                        return Err(format!("Expected array size (integer literal), found {:?}", self.peek().kind));
                    };
                    self.expect(&TokenKind::RBracket)?;
                    Ok(Type::Array(inner, Some(size)))
                } else {
                    self.expect(&TokenKind::RBracket)?;
                    Ok(Type::Slice(inner))
                }
            }
            _ => Err(format!("Expected type, found {:?}", self.peek().kind))
        }
    }

    fn parse_struct(&mut self, is_pub: bool) -> Result<Struct, String> {
        let start = self.advance(); // struct
        let name = self.expect_ident()?;
        let mut type_params = Vec::new();
        if self.check(&TokenKind::Lt) {
            self.advance();
            while !self.check(&TokenKind::Gt) && !self.is_at_end() {
                let tp = self.expect_ident()?;
                type_params.push(tp);
                if !self.match_token(&[TokenKind::Comma]) { break; }
            }
            self.expect(&TokenKind::Gt)?;
        }
        self.expect(&TokenKind::LBrace)?;
        let mut fields = Vec::new();

        while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
            // Fields are public by default — `pub` is accepted but
            // redundant (kept for readability/explicitness). Use
            // `private` to opt a field OUT of external access; it's then
            // only readable/writable via `self` from within that struct's
            // own impl block. This default (public-by-default,
            // opt-in-private) was chosen specifically so existing
            // Quantum code doesn't silently break — Rust-style
            // private-by-default would have made every previously-working
            // struct field inaccessible without warning.
            let field_private = self.match_token(&[TokenKind::Private]);
            self.match_token(&[TokenKind::Pub]); // accepted, no-op (already the default)
            let field_name = self.expect_ident()?;
            self.expect(&TokenKind::Colon)?;
            let field_type = self.parse_type()?;

            fields.push(StructField {
                name: field_name,
                ty: field_type,
                is_pub: !field_private,
            });

            self.match_token(&[TokenKind::Comma]);
        }

        self.expect(&TokenKind::RBrace)?;

        Ok(Struct {
            id: self.next_id(),
            name,
            type_params,
            fields,
            is_pub,
            span: start.span,
        })
    }

    fn parse_enum(&mut self, is_pub: bool) -> Result<Enum, String> {
        let start = self.advance(); // enum
        let name = self.expect_ident()?;

        self.expect(&TokenKind::LBrace)?;
        let mut variants = Vec::new();

        while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
            let variant_name = self.expect_ident()?;
            variants.push(EnumVariant {
                name: variant_name,
                fields: None,
            });

            self.match_token(&[TokenKind::Comma]);
        }

        self.expect(&TokenKind::RBrace)?;

        Ok(Enum {
            id: self.next_id(),
            name,
            variants,
            is_pub,
            span: start.span,
        })
    }

    fn parse_trait(&mut self, is_pub: bool) -> Result<Trait, String> {
        let start = self.advance(); // trait
        let name = self.expect_ident()?;

        self.expect(&TokenKind::LBrace)?;
        let mut methods = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
            self.expect(&TokenKind::Fn)?;
            let method_name = self.expect_ident()?;
            self.expect(&TokenKind::LParen)?;
            let params = self.parse_params()?;
            self.expect(&TokenKind::RParen)?;
            let return_type = if self.match_token(&[TokenKind::Arrow]) {
                Some(self.parse_type()?)
            } else {
                None
            };
            // A trait method can either be a bare signature (no body —
            // every implementing struct must provide its own), or have a
            // default body in braces (implementing structs inherit this
            // implementation for free unless they override it).
            let default_body = if self.check(&TokenKind::LBrace) {
                Some(self.parse_block()?)
            } else {
                // Signature-only: accept and discard an optional trailing `;`.
                self.match_token(&[TokenKind::Semicolon]);
                None
            };

            methods.push(TraitMethod {
                name: method_name,
                params,
                return_type,
                default_body,
            });
        }
        self.expect(&TokenKind::RBrace)?;

        Ok(Trait {
            id: self.next_id(),
            name,
            methods,
            is_pub,
            span: start.span,
        })
    }

    fn parse_impl(&mut self) -> Result<Impl, String> {
        let start = self.advance(); // impl
        let first_name = self.expect_ident()?;

        // `impl Trait for Type { ... }` vs plain `impl Type { ... }` —
        // disambiguated by checking for the `for` keyword right after the
        // first identifier.
        let (trait_name, type_name) = if self.match_token(&[TokenKind::For]) {
            let type_name = self.expect_ident()?;
            (Some(first_name), type_name)
        } else {
            (None, first_name)
        };

        self.expect(&TokenKind::LBrace)?;
        let mut methods = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
            // Methods are public by default, same convention as struct
            // fields — `private` opts a method out of external calls
            // (e.g. p.internal_helper() from outside), while still being
            // callable from other methods within the same impl block.
            let method_private = self.match_token(&[TokenKind::Private]);
            self.match_token(&[TokenKind::Pub]); // accepted, no-op
            // parse_function starts with `self.advance() // fn`, so we
            // must NOT pre-consume `fn` here — just check it exists for
            // a clearer error message, then let parse_function handle it.
            if !self.check(&TokenKind::Fn) {
                return Err(format!("Expected 'fn' in impl block, found {:?}", self.peek().kind));
            }
            methods.push(self.parse_function(!method_private)?);
        }
        self.expect(&TokenKind::RBrace)?;

        Ok(Impl {
            id: self.next_id(),
            trait_name,
            type_name,
            methods,
            span: start.span,
        })
    }

    fn parse_import(&mut self) -> Result<Import, String> {
        let start = self.advance(); // import/from
        let mut path = vec![self.expect_ident()?];

        while self.match_token(&[TokenKind::ColonColon, TokenKind::Dot]) {
            path.push(self.expect_ident()?);
        }

        // Optional: `as alias`
        let alias = if self.match_token(&[TokenKind::As]) {
            Some(self.expect_ident()?)
        } else {
            None
        };

        // Optional: `{ item1, item2 }`
        let items = if self.check(&TokenKind::LBrace) {
            self.advance();
            let mut names = vec![self.expect_ident()?];
            while self.match_token(&[TokenKind::Comma]) {
                if self.check(&TokenKind::RBrace) { break; }
                names.push(self.expect_ident()?);
            }
            let _ = self.expect(&TokenKind::RBrace);
            Some(names)
        } else {
            None
        };

        Ok(Import {
            path,
            items,
            alias,
            span: start.span,
        })
    }

    fn parse_const(&mut self, is_pub: bool) -> Result<Const, String> {
        self.advance(); // const
        let name = self.expect_ident()?;
        self.expect(&TokenKind::Colon)?;
        let ty = self.parse_type()?;
        self.expect(&TokenKind::Eq)?;
        let value = self.parse_expr()?;

        Ok(Const {
            name,
            ty,
            value,
            is_pub,
            span: self.previous().span,
        })
    }

    fn parse_model(&mut self, is_pub: bool) -> Result<Model, String> {
        let start = self.advance(); // model
        let name = self.expect_ident()?;

        self.expect(&TokenKind::LBrace)?;
        let layers = Vec::new(); // TODO: Parse layers

        // Find forward function
        let forward_stub = Function {
            id: self.next_id(),
            name: "forward".to_string(),
            type_params: vec![],
            params: vec![],
            return_type: None,
            body: Block {
                stmts: vec![],
                expr: None,
                span: start.span,
            },
            is_pub: false,
            is_async: false,
            span: start.span,
        };

        self.expect(&TokenKind::RBrace)?;

        Ok(Model {
            id: self.next_id(),
            name,
            layers,
            forward: forward_stub,
            is_pub,
            span: start.span,
        })
    }

    // Helper functions
    fn peek(&self) -> &Token {
        if self.current < self.tokens.len() {
            &self.tokens[self.current]
        } else {
            // Return the last token (which should be EOF) as a sentinel
            self.tokens.last().expect("token stream is empty")
        }
    }

    fn peek_ahead(&self, offset: usize) -> Option<&Token> {
        self.tokens.get(self.current + offset)
    }

    fn previous(&self) -> &Token {
        &self.tokens[self.current - 1]
    }

    fn advance(&mut self) -> Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous().clone()
    }

    fn check(&self, kind: &TokenKind) -> bool {
        if self.is_at_end() {
            return false;
        }
        std::mem::discriminant(&self.peek().kind) == std::mem::discriminant(kind)
    }

    fn match_token(&mut self, kinds: &[TokenKind]) -> bool {
        for kind in kinds {
            if self.check(kind) {
                self.advance();
                return true;
            }
        }
        false
    }

    fn expect(&mut self, kind: &TokenKind) -> Result<Token, String> {
        if self.check(kind) {
            Ok(self.advance())
        } else {
            let tok = self.peek();
            Err(format!("Expected {:?}, found {:?} at line {}, column {}", kind, tok.kind, tok.span.line, tok.span.column))
        }
    }

    fn expect_ident(&mut self) -> Result<String, String> {
        match &self.peek().kind {
            TokenKind::Ident(name) => {
                let name = name.clone();
                self.advance();
                Ok(name)
            }
            _ => Err(format!("Expected identifier, found {:?}", self.peek().kind))
        }
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.tokens.len()
    }

    fn get_precedence(&self, kind: &TokenKind) -> Option<u8> {
        match kind {
            TokenKind::Or => Some(1),
            TokenKind::And => Some(2),
            TokenKind::EqEq | TokenKind::Ne => Some(3),
            TokenKind::Lt | TokenKind::Le | TokenKind::Gt | TokenKind::Ge => Some(4),
            TokenKind::Pipe => Some(5),
            TokenKind::Caret => Some(6),
            TokenKind::Ampersand => Some(7),
            TokenKind::Lshift | TokenKind::Rshift => Some(8),
            TokenKind::Plus | TokenKind::Minus => Some(9),
            TokenKind::Star | TokenKind::Slash | TokenKind::Percent => Some(10),
            TokenKind::Power => Some(11),
            TokenKind::DotDot | TokenKind::DotDotEq => Some(2),
            _ => None
        }
    }

    fn token_to_binop(&self, kind: &TokenKind) -> Result<BinOp, String> {
        match kind {
            TokenKind::Plus => Ok(BinOp::Add),
            TokenKind::Minus => Ok(BinOp::Sub),
            TokenKind::Star => Ok(BinOp::Mul),
            TokenKind::Slash => Ok(BinOp::Div),
            TokenKind::Percent => Ok(BinOp::Mod),
            TokenKind::Power => Ok(BinOp::Power),
            TokenKind::EqEq => Ok(BinOp::Eq),
            TokenKind::Ne => Ok(BinOp::Ne),
            TokenKind::Lt => Ok(BinOp::Lt),
            TokenKind::Le => Ok(BinOp::Le),
            TokenKind::Gt => Ok(BinOp::Gt),
            TokenKind::Ge => Ok(BinOp::Ge),
            TokenKind::And => Ok(BinOp::And),
            TokenKind::Or => Ok(BinOp::Or),
            TokenKind::Ampersand => Ok(BinOp::BitAnd),
            TokenKind::Pipe => Ok(BinOp::BitOr),
            TokenKind::Caret => Ok(BinOp::BitXor),
            TokenKind::Lshift => Ok(BinOp::Lshift),
            TokenKind::Rshift => Ok(BinOp::Rshift),
            TokenKind::DotDot => Ok(BinOp::Range),
            TokenKind::DotDotEq => Ok(BinOp::RangeInclusive),
            _ => Err(format!("Not a binary operator: {:?}", kind))
        }
    }
}
