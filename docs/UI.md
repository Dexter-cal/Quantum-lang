# Quantum Standard Library — Terminal UI (`ui.qtm`)

The `ui` module provides ANSI colors, bold text, box rendering, banners, and progress bar components for CLI applications.

## Import
```quantum
import ui
```

## Functions

### `bold(text: string) -> string`
Wraps text in ANSI bold escape codes.

### `red(text: string) -> string` / `green` / `yellow` / `blue` / `cyan`
Wraps text in ANSI color codes.

### `draw_box(title: string, content: string)`
Renders a styled ASCII border box containing title and content.

### `print_banner(app_name: string, version: string)`
Prints an ASCII application header banner.

### `progress_bar(percent: int, width: int)`
Prints an ASCII progress bar `[====  ] 50%`.

## Example
```quantum
import ui

fn main() {
    ui.print_banner("Quantum App", "1.0")
    println(ui.green("Task running..."))
    ui.progress_bar(80, 20)
    ui.draw_box("Result", "Completed successfully!")
}
```
