# Quantum Standard Library — Operating System (`os.qtm`)

The `os` module provides functions for executing shell commands, reading environment variables, and managing processes.

## Import
```quantum
import os
```

## Functions

### `exec_cmd(command: string) -> int`
Executes a system command in the host operating system shell.
- **Parameters:** `command` — The command string to run (e.g. `"ls -la"` or `"echo Hello"`).
- **Returns:** `int` — Exit status code (0 = success).
- **Example:**
```quantum
import os

fn main() {
    let status = os.exec_cmd("echo Hello from OS module!")
    println("Command exit status: " + status.to_string())
}
```

### `get_env(variable: string) -> string`
Retrieves the value of an environment variable.
- **Parameters:** `variable` — Environment variable name (e.g. `"USER"`, `"PATH"`).
- **Returns:** `string` — Value of environment variable or empty string if not found.
- **Example:**
```quantum
import os

fn main() {
    let user = os.get_env("USER")
    println("Current logged in user: " + user)
}
```

### `exit_process(code: int)`
Terminates the program execution immediately with the given exit code.
- **Parameters:** `code` — Exit code (0 = normal termination).
