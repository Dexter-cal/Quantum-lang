# Quantum Standard Library Reference (`qtm-std`)

Welcome to the **Quantum Standard Library Reference**. This guide documents all modules included with the Quantum Programming Language, complete with examples and beginner tutorials.

---

## Table of Contents
1. [Overview & Ecosystem Architecture](#overview)
2. [Operating System (`os.qtm`)](#osqtm)
3. [Date & Time (`time.qtm`)](#timeqtm)
4. [Terminal UI (`ui.qtm`)](#uiqtm)
5. [JSON Parsing (`json.qtm`)](#jsonqtm)
6. [Mathematics (`math.qtm`)](#mathqtm)
7. [String Utilities (`strings.qtm`)](#stringsqtm)
8. [Vector Operations (`vec.qtm`)](#vecqtm)
9. [HashMaps (`hashmap.qtm`)](#hashmapqtm)
10. [Collections & Arrays (`collections_arrays.qtm`)](#collections_arraysqtm)
11. [QuantumAI Framework (`ai.qtm`)](#aiqtm)

---

## Overview

Quantum standard library modules live in the `qtm-std/` folder. They can be imported in any Quantum script using the `import` keyword.

```quantum
import os
import time
import ui
import json
import math

fn main() {
    ui.print_banner("My Quantum App", "1.0")
    println("Current time: " + time.now().to_string())
}
```

The compiler automatically resolves standard library paths (`qtm-std/`), so you can write `import math` or `import std.math` from anywhere!

---

## `os.qtm` — Operating System Module

Provides system command execution, environment variable lookup, and process control.

### Functions
- `exec_cmd(command: string) -> int` — Executes a system shell command and returns the exit status code.
- `get_env(variable: string) -> string` — Looks up an environment variable by name.

### Example
```quantum
import os

fn main() {
    let user = os.get_env("USER")
    println("Current user: " + user)

    os.exec_cmd("ls -la")
}
```

---

## `time.qtm` — Date & Time Module

Provides system timestamping, millisecond/second sleeping, and duration formatting.

### Functions
- `now() -> int` — Returns current UNIX timestamp in seconds.
- `sleep_ms(ms: int)` — Pauses execution for `ms` milliseconds.
- `sleep_sec(sec: int)` — Pauses execution for `sec` seconds.
- `clock_ticks() -> int` — Returns raw CPU clock ticks.
- `is_leap_year(year: int) -> bool` — Returns true if `year` is a leap year.
- `format_duration_sec(seconds: int) -> string` — Formats seconds as `Xh Ym Zs`.

### Example
```quantum
import time

fn main() {
    let start = time.now()
    time.sleep_ms(500)
    let elapsed = time.now() - start
    println("Elapsed: " + time.format_duration_sec(elapsed))
}
```

---

## `ui.qtm` — Terminal UI Framework

Provides ANSI color formatting, styled text, boxes, headers, and progress bars.

### Functions
- `bold(text: string) -> string` — Formats text as bold ANSI.
- `red(text: string) -> string` / `green` / `yellow` / `blue` / `cyan` — Colors text.
- `draw_box(title: string, content: string)` — Draws a styled ASCII box with title.
- `print_banner(app_name: string, version: string)` — Prints an application header banner.
- `progress_bar(percent: int, width: int)` — Renders a progress bar `[====  ] 50%`.

### Example
```quantum
import ui

fn main() {
    ui.print_banner("Quantum App", "2.1")
    println(ui.cyan("Processing task..."))
    ui.progress_bar(75, 20)
    ui.draw_box("Status", ui.green("Operation complete!"))
}
```

---

## `json.qtm` — JSON Parsing Module

Provides lightweight JSON key-value extraction and parsing.

### Functions
- `json_get_string(json_str: string, key: string) -> string` — Extracts string value for key.
- `json_get_int(json_str: string, key: string) -> int` — Extracts integer value for key.

### Example
```quantum
import json

fn main() {
    let data = "{\"name\": \"Quantum\", \"version\": 2}"
    let name = json.json_get_string(data, "name")
    let ver  = json.json_get_int(data, "version")
    println("App: " + name + " v" + ver.to_string())
}
```

---

## `math.qtm` — Mathematics Module

Wraps standard C math routines via FFI.

### Functions
- `math_sqrt(x: float) -> float` — Square root.
- `math_pow(x: float, y: float) -> float` — Power $x^y$.
- `math_sin(x: float)`, `math_cos(x: float)`, `math_tan(x: float)` — Trigonometry.
- `math_floor(x: float)`, `math_ceil(x: float)`, `math_round(x: float)` — Rounding.
- `math_clamp(x: float, min: float, max: float) -> float` — Clamp range.
- `math_pi() -> float` — Returns $\pi \approx 3.14159265$.

### Example
```quantum
import math

fn main() {
    let r = 5.0
    let area = math.math_pi() * math.math_pow(r, 2.0)
    println("Circle area: " + area.to_string())
}
```

---

## `strings.qtm` — String Utilities

Character classification, parsing, padding, and repeating.

### Functions
- `str_is_digit(c: int) -> bool` — Tests if character code is a digit.
- `str_parse_int(s: string) -> int` — Parses string to integer.
- `str_repeat(s: string, count: int) -> string` — Repeats string `count` times.
- `str_pad_left(s: string, len: int, pad: string) -> string` — Left pad.
- `str_pad_right(s: string, len: int, pad: string) -> string` — Right pad.

### Example
```quantum
import strings

fn main() {
    let num = strings.str_parse_int("123")
    println("Parsed: " + num.to_string())
    println(strings.str_repeat("*", 10))
}
```

---

## `ai.qtm` — QuantumAI Framework

First-class AI & Deep Learning capabilities integrated into Quantum.

### Quick Start
```quantum
import ai as qai

fn main() {
    let model = qai.create("classifier")
    let ds = qai.make_classification(500, 8, 4)
    qai.model_train(model, ds, 50)
    qai.model_summary(model)
}
```
