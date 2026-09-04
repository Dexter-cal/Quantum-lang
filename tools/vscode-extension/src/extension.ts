// Quantum Language Support — VS Code extension entry point.
//
// Launches `quantum-lsp` (built from the quantum-lang workspace's
// `quantum-lsp` crate) as a language server and connects it via
// vscode-languageclient. The server communicates over stdio using
// standard Content-Length-framed JSON-RPC (LSP).

import * as path from 'path';
import { workspace, ExtensionContext } from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind,
} from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: ExtensionContext) {
    // Resolve the quantum-lsp binary. Users can override the path via the
    // "quantum.lspPath" setting; otherwise we assume it's on PATH (e.g.
    // after `cargo install --path quantum-lsp` or a release build copied
    // into PATH) or built at <workspace>/target/release/quantum-lsp.
    const config = workspace.getConfiguration('quantum');
    const configuredPath = config.get<string>('lspPath');

    const serverCommand = configuredPath && configuredPath.length > 0
        ? configuredPath
        : 'quantum-lsp';

    const serverOptions: ServerOptions = {
        run: { command: serverCommand, transport: TransportKind.stdio },
        debug: { command: serverCommand, transport: TransportKind.stdio },
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'quantum' }],
        synchronize: {
            fileEvents: workspace.createFileSystemWatcher('**/*.qtm'),
        },
    };

    client = new LanguageClient(
        'quantumLanguageServer',
        'Quantum Language Server',
        serverOptions,
        clientOptions,
    );

    // Starts the client and launches the server process.
    client.start();
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
