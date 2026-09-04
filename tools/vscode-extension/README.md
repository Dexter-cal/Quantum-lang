# Quantum Language Server (`quantum-lsp`)

A minimal Language Server Protocol (LSP) implementation for Quantum
(`.qtm`) files.

## What it does

- **Diagnostics**: runs lex → parse → typecheck (via
  `quantum_compiler::check_source`) on every open/changed/saved file and
  reports the first error found, with line/column position when the
  underlying error carries one.
- **Hover**: shows short documentation for Quantum keywords (`fn`, `let`,
  `for`, `match`, ...) and common QuantumAI API names (`Model`,
  `Sequential`, `.dense`, `BenchmarkReport`, ...) — see
  `src/hover_docs.rs`.

## What it does NOT do (yet)

- No autocomplete / code completion
- No go-to-definition / find references
- No rename / code actions / formatting
- Only the *first* diagnostic per file is reported (the compiler's
  front-end is not yet error-recovering, so it stops at the first error in
  each stage)
- Most typechecker errors don't carry line/column info yet, so those are
  reported at line 1 of the file rather than the exact location

These are reasonable next steps once this foundation is in place.

## Building

From the `quantum-lang` workspace root:

```bash
cargo build --release -p quantum-lsp
```

The binary is produced at `target/release/quantum-lsp`. Either add
`target/release` to your `PATH`, or point the VS Code extension at it
directly (see below).

## Testing without an editor

`quantum-lsp` speaks standard Content-Length-framed JSON-RPC over stdio —
the same transport every LSP client uses. You can drive it directly for
testing; see the workspace's test scripts for an example Python harness
that sends `initialize`, `textDocument/didOpen`, etc. and prints the
`publishDiagnostics` notifications and `hover` responses.

## VS Code extension

The extension in `tools/vscode-extension/` provides syntax highlighting
(already existed) plus, now, a language client that launches `quantum-lsp`
for diagnostics and hover.

### Setup

```bash
cd tools/vscode-extension
npm install
npm run compile
```

Then either:
- Press F5 in VS Code (with this folder open) to launch an Extension
  Development Host, or
- Run `npx vsce package` to produce a `.vsix` you can install with
  `code --install-extension quantum-lang-1.0.0.vsix`.

### Configuration

By default the extension looks for a `quantum-lsp` executable on your
`PATH`. If you haven't installed it system-wide, set the absolute path in
your VS Code settings:

```json
{
  "quantum.lspPath": "/path/to/quantum-lang/target/release/quantum-lsp"
}
```

## Architecture notes

`quantum-lsp` depends on `quantum-compiler` as a library (see
`compiler/src/lib.rs`, which exposes the lexer/parser/typechecker and a
`check_source()` helper returning a `Vec<Diagnostic>`). The `quantumc`
binary is unaffected — both the `[lib]` and `[[bin]]` targets are built
from the same crate.

No external LSP framework (e.g. `tower-lsp`) is used: its dependency tree
requires a Rust edition newer than what's available in this environment.
The protocol (Content-Length framing + JSON-RPC) is small enough that
`quantum-lsp/src/transport.rs` implements it directly in ~50 lines.
