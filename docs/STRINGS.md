# Quantum Standard Library — String Utilities (`strings.qtm`)

The `strings` module provides string manipulation, padding, character classification, and repeating.

## Import
```quantum
import strings
```

## Functions

- `str_is_digit(char_code: int) -> bool` — True if ASCII digit ('0'-'9').
- `str_parse_int(s: string) -> int` — Parses string into integer.
- `str_repeat(s: string, count: int) -> string` — Repeats string `count` times.
- `str_pad_left(s: string, len: int, pad: string) -> string` — Left-pads string to target length.
- `str_pad_right(s: string, len: int, pad: string) -> string` — Right-pads string to target length.

## Example
```quantum
import strings

fn main() {
    println(strings.str_repeat("=", 20))
    let padded = strings.str_pad_left("7", 4, "0")
    println("Padded ID: " + padded)
}
```
