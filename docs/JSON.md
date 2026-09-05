# Quantum Standard Library — JSON Parsing (`json.qtm`)

The `json` module provides lightweight JSON string extraction and integer parsing.

## Import
```quantum
import json
```

## Functions

### `json_get_string(json_str: string, key: string) -> string`
Extracts string value associated with `key` from a JSON string.
- **Parameters:**
  - `json_str` — JSON formatted string (e.g. `{"name": "Alice"}`).
  - `key` — Key to extract.
- **Returns:** `string` — Extracted value string or empty string.

### `json_get_int(json_str: string, key: string) -> int`
Extracts integer value associated with `key` from a JSON string.
- **Parameters:**
  - `json_str` — JSON formatted string.
  - `key` — Key to extract.
- **Returns:** `int` — Parsed integer value or 0.

## Example
```quantum
import json

fn main() {
    let payload = "{\"user\": \"Developer\", \"score\": 100}"
    let user = json.json_get_string(payload, "user")
    let score = json.json_get_int(payload, "score")
    println("User: " + user + ", Score: " + score.to_string())
}
```
