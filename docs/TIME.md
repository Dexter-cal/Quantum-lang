# Quantum Standard Library — Date & Time (`time.qtm`)

The `time` module provides functions for timestamps, sleep delays, clock ticks, leap year calculations, and formatting.

## Import
```quantum
import time
```

## Functions

### `now() -> int`
Returns the current UNIX timestamp in seconds.
- **Returns:** `int` — Seconds since Jan 1, 1970 UTC.

### `sleep_ms(ms: int)`
Pauses execution for `ms` milliseconds.
- **Parameters:** `ms` — Duration to sleep in milliseconds.

### `sleep_sec(sec: int)`
Pauses execution for `sec` seconds.
- **Parameters:** `sec` — Duration to sleep in seconds.

### `clock_ticks() -> int`
Returns raw CPU clock ticks spent by the process.

### `is_leap_year(year: int) -> bool`
Checks whether a given calendar year is a leap year.
- **Parameters:** `year` — Integer year (e.g. 2024).
- **Returns:** `bool` — `true` if leap year, `false` otherwise.

### `format_duration_sec(seconds: int) -> string`
Formats a number of seconds into human-readable duration `Xh Ym Zs`.
- **Parameters:** `seconds` — Integer total seconds.
- **Returns:** `string` — Formatted string (e.g. `"1h 30m 15s"`).

## Example
```quantum
import time

fn main() {
    let t0 = time.now()
    println("Sleeping for 100ms...")
    time.sleep_ms(100)
    let elapsed = time.now() - t0
    println("Duration: " + time.format_duration_sec(elapsed))
}
```
