//! Content-Length-framed JSON-RPC transport, as used by LSP over stdio.
//!
//! Each message looks like:
//! ```text
//! Content-Length: 123\r\n
//! \r\n
//! {"jsonrpc":"2.0",...}
//! ```

use serde_json::Value;
use std::io::{self, BufRead, Write};

/// Read one JSON-RPC message from `reader`. Returns `Ok(None)` on clean EOF
/// (no more messages), `Ok(Some(value))` on a successfully parsed message,
/// or `Err` on a malformed header/body.
pub fn read_message<R: BufRead>(reader: &mut R) -> io::Result<Option<Value>> {
    let mut content_length: Option<usize> = None;

    // Read headers, terminated by an empty line.
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            // EOF before any header lines for this message — clean shutdown.
            return Ok(None);
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break; // End of headers
        }
        if let Some(value) = trimmed.strip_prefix("Content-Length:") {
            content_length = value.trim().parse::<usize>().ok();
        }
        // Other headers (e.g. Content-Type) are ignored.
    }

    let len = content_length.ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "missing Content-Length header")
    })?;

    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;

    let value: Value = serde_json::from_slice(&buf)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    Ok(Some(value))
}

/// Write one JSON-RPC message to `writer`, framed with a Content-Length
/// header as required by LSP.
pub fn write_message<W: Write>(writer: &mut W, value: &Value) -> io::Result<()> {
    let body = serde_json::to_vec(value)?;
    write!(writer, "Content-Length: {}\r\n\r\n", body.len())?;
    writer.write_all(&body)?;
    writer.flush()?;
    Ok(())
}
