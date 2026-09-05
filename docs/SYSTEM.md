# Quantum Standard Library — System (`system.qtm`)

The `system` module provides access to operating system info, environment variables, command arguments, processes, timers, memory, and application lifecycle.

## Import
```quantum
import system
```

## Modules & Functions

### `system.info`
- `info_os_name() -> string` — Returns operating system name (e.g. `"Linux"`).
- `info_architecture() -> string` — Returns CPU architecture (e.g. `"x64"`).
- `info_username() -> string` — Returns current logged-in username.
- `info_home_dir() -> string` — Returns path to home directory.
- `info_temp_dir() -> string` — Returns path to temporary directory.

### `system.env`
- `env_get(name: string) -> string` — Returns value of environment variable or `""`.
- `env_exists(name: string) -> bool` — `true` if environment variable is defined.

### `system.process`
- `process_spawn(command: string) -> int` — Spawns process command and returns exit code.

### `system.timer`
- `timer_now() -> int` — Returns current UNIX timestamp in seconds.
- `timer_sleep_ms(ms: int)` — Pauses execution for `ms` milliseconds.
- `timer_sleep_sec(sec: int)` — Pauses execution for `sec` seconds.

### `system.app`
- `app_exit(code: int)` — Terminates program with exit code.

## Example
```quantum
import system

fn main() {
    println("OS Name: " + system.info_os_name())
    println("User: " + system.info_username())
    println("Home: " + system.info_home_dir())

    if system.env_exists("PATH") {
        println("PATH environment variable is set.")
    }
}
```
