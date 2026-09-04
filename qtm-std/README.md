# qtm-std — Quantum Standard Library

A small collection of `.qtm` modules providing math, string, and array
utilities, written in Quantum itself (dogfooding the language's
`extern "C"` FFI, multi-file `import` system, and numeric casts).

## Usage

Copy the module file(s) you need into your project directory (alongside
your `main.qtm`), then import by filename (without the `.qtm` extension):

```quantum
import math

fn main() {
    println(math_sqrt(64.0))
}
```

Quantum's `import` resolves `math` to `math.qtm` in the same directory and
merges its functions into your program at compile time. No package manager
or installation step required yet — just copy the file.

## Naming convention

Quantum doesn't have namespaced imports yet (no `math.sqrt(...)` syntax —
just `math_sqrt(...)`), so every function in every module is prefixed with
the module name to avoid collisions with your own functions or with other
imported modules. This mirrors how C and early Go-style code organized
itself before real module systems existed.

## Modules

### `math.qtm`

Wraps the full C standard math library (libm) via `extern "C"`: power,
roots, trigonometry, hyperbolic functions, logarithms, rounding, plus a
few pure-Quantum helpers that don't need libm at all. Always available
with zero extra linker flags — `-lm` is linked unconditionally by the
compiler.

```quantum
import math

fn main() {
    println(math_sqrt(16.0))           // 4
    println(math_pow(2.0, 10.0))       // 1024
    println(math_floor(9.7))           // 9
    println(math_clamp(15.0, 0.0, 10.0))  // 10
    println(math_pi())                 // 3.14159
}
```

### `strings.qtm`

Character classification (`ctype.h`), parsing (`atoi`/`atof`), and a few
pure-Quantum string helpers.

```quantum
import strings

fn main() {
    println(str_is_digit(53))               // true ('5')
    println(str_parse_int("42"))             // 42
    println(str_repeat("ab", 3))             // ababab
    println(str_pad_left("7", 3, "0"))       // 007
    println(str_pad_right("ab", 5, "*"))     // ab***
}
```

### `collections.qtm`

The original scalar-only module: range utilities and classic algorithms
that only need int/float/bool, no arrays.

```quantum
import collections

fn main() {
    println(range_sum(1, 10))    // 55
    println(is_prime(17))        // true
    println(gcd(48, 18))         // 6
    println(fibonacci(10))       // 55
}
```

### `collections_arrays.qtm`

A real array-based collections module: aggregation, searching, and
in-place mutation (sort, reverse, fill).

```quantum
import collections_arrays

fn main() {
    let nums = [5, 3, 8, 1, 9, 0, 0, 0, 0, 0]

    println(array_sum(nums, 5))       // 26
    println(array_max(nums, 5))       // 9
    println(array_average(nums, 5))   // 5.2
    println(array_contains(nums, 5, 8))  // true
    println(array_index_of(nums, 5, 8))  // 2

    array_sort(nums, 5)   // mutates in place — ascending
    println(nums[0])      // 1
    println(nums[4])      // 9
}
```

**Important constraint**: arrays cannot currently be *returned* from a
function (C doesn't allow returning array types by value, and Quantum's
codegen doesn't wrap them in a struct to work around this). Every function
in this module either returns a scalar or mutates an array passed in by
the caller — exactly like plain C array semantics, since Quantum arrays
decay to pointers when passed to a function:

```quantum
// This works — mutating an array passed in by the caller:
fn double_all(arr: [int; 5], len: int) {
    let mut i = 0
    while i < len {
        arr[i] = arr[i] * 2
        i = i + 1
    }
}

// This does NOT work — arrays can't be returned by value:
// fn make_array() -> [int; 5] { return [1, 2, 3, 4, 5] }
```

All array functions also take an explicit `len` parameter, since C array
parameters don't carry their length at runtime.

## Compound assignment (`+=`, `-=`, `*=`, `/=`)

```quantum
fn main() {
    let mut total = 0
    let mut i = 0
    while i < 5 {
        total += i
        i += 1
    }
    println(total)   // 10

    let arr = [1, 2, 3]
    arr[0] += 100
    println(arr[0])  // 101
}
```

These desugar to `x = x + value` etc. at parse time, so they work anywhere
a plain assignment does — including array elements and loop accumulators,
both used throughout `collections.qtm` and `collections_arrays.qtm`.

## The cast operator (`as`)

Quantum has no implicit int/float promotion — `int_value / float_value` is
a type error. Use `as` to convert explicitly:

```quantum
fn main() {
    let a = 7
    let b = 2
    let result = (a as float) / (b as float)
    println(result)   // 3.5 — without the casts, 7 / 2 would give 2 (int division)
}
```

This is what makes `array_average` in `collections_arrays.qtm` able to
return a true float average instead of a truncated integer one.

## Known limitations affecting these modules

- **No namespacing**: every function needs a unique prefixed name across
  your whole program, including all imported modules.
- **Arrays cannot be returned from functions** — see the
  `collections_arrays.qtm` section above for the workaround (mutate an
  array passed in by the caller instead).
- **`extern "C"` declarations for functions already declared by an
  included system header** (e.g. `strstr`, `getenv` — both already
  declared via the unconditionally-included `string.h`/`stdlib.h`) can hit
  a C type-conflict error (`char*` vs `const char*`) at compile time. This
  affects only functions that collide with something the generated C file
  already includes — most external library functions (libm, libcurl,
  etc.) aren't affected since their headers aren't auto-included.

## Compiler fixes made alongside this library

A few real compiler bugs were found and fixed while building these
modules: fixed-size array indexing (previously crashed with a stack
overflow), array function parameters (`[int; N]` syntax, didn't parse at
all), complex index expressions as assignment targets (`arr[j+1] = x`,
needed for in-place sorting), a parameter/builtin name-shadowing bug that
made names like `len`, `min`, and `max` unusable as ordinary parameter
names anywhere in their scope, the `as` cast operator (parsed but had zero
MIR lowering, plus several places where a cast or float-producing binary
expression's result got silently truncated back to an integer at the next
`let`/`return` boundary), and compound assignment operators (`+=`, `-=`,
`*=`, `/=` — the tokens existed in the lexer but the parser never used
them).

## Testing these modules yourself

Each module has a paired `test_*.qtm` file demonstrating every function.
Run with:

```bash
quantumc run test_math_module.qtm
quantumc run test_strings_module.qtm
quantumc run test_collections.qtm
quantumc run test_collections_arrays.qtm
```

## do-while

A `do { ... } while cond` loop — like `while`, but the body always runs at least once before the condition is ever checked.

```quantum
fn main() {
    let mut i = 10
    do {
        println(i)
        i = i + 1
    } while i < 5
}
// Output: 10 — the body runs once even though the condition is false
// from the very first check, which is the entire point of do-while
// versus a plain while loop.
```

`break` and `continue` work inside `do-while` exactly as in the other loop forms.

## Logical keywords (`and`/`or`/`not`) vs symbols (`&&`/`||`/`!`)

The symbol forms `&&`, `||`, `!` work by default, with no setup needed. The word forms `and`, `or`, `not` are **not** active by default — they're part of the optional Python language pack, not the base keyword set. To use them, set up a `quantum_lang.toml` in your project directory:

```bash
quantumc dict use python
```

```quantum
fn main() {
    let a = true
    let b = false
    println(a and b)   // false
    println(a or b)    // true
    println(not a)     // false
}
```

Without the dictionary pack loaded, `and`/`or`/`not` are treated as plain identifiers and won't parse as operators — `a and b` would error rather than silently doing the wrong thing.

## Loops: while, for, loop, do-while (with break/continue)

All three loop forms support `break` and `continue` correctly, including nested loops and `for`-loop `continue` (which correctly hits the increment step rather than skipping it).

```quantum
fn main() {
    let mut i = 0
    loop {
        i = i + 1
        if i >= 5 {
            break
        }
        if i == 2 {
            continue
        }
        println(i)
    }
    println("done")
}
// Output: 1, 3, 4, done
```

```quantum
fn main() {
    for i in 0..10 {
        if i == 3 {
            continue
        }
        if i == 6 {
            break
        }
        println(i)
    }
}
// Output: 0, 1, 2, 4, 5
```

Nested loops work correctly too — `break`/`continue` only affect the innermost enclosing loop:

```quantum
fn main() {
    for i in 0..3 {
        for j in 0..3 {
            if j == 1 {
                break   // only exits the inner loop
            }
            println(i * 10 + j)
        }
    }
}
// Output: 0, 10, 20
```

**Previously, `break`/`continue` were silent no-ops and `loop { ... }` bodies never executed at all** — this was fixed by adding a loop-context stack (tracking the innermost loop's continue/break targets) and fixing a deeper bug where a loop body's trailing `if`-statement (no else, no value — the common `if cond { break }` shape) was incorrectly treated as a function-level `return`, orphaning the loop-back edge entirely.

## Encapsulation (struct/method visibility)

Struct fields and methods are **public by default** — this matches how Quantum has worked since structs were introduced, so existing code doesn't break. Use `private` to opt a field or method out of external access:

```quantum
struct BankAccount {
    private balance: int,
}

impl BankAccount {
    fn get_balance(self) -> int {
        return self.balance       // OK: accessing private field from within the struct's own methods
    }
    private fn is_valid_amount(self, amount: int) -> bool {
        return amount > 0
    }
    fn deposit_check(self, amount: int) -> bool {
        return self.is_valid_amount(amount)   // OK: calling private method from another method
    }
}

fn main() {
    let acc = BankAccount { balance: 500 }
    println(acc.get_balance())          // 500 — public method
    println(acc.deposit_check(100))     // true — public method calling a private helper internally
    // println(acc.balance)              // ERROR: balance is private
    // println(acc.is_valid_amount(10))  // ERROR: is_valid_amount is private
}
```

A private field/method is accessible from any method within that struct's own `impl` block, but not from outside it. This is enforced at compile time.

**Note on scope**: this enforces visibility at the struct boundary, not a full module/file visibility system — Quantum doesn't yet track which file an item originated from across `import`s. For typical single-file or few-file Quantum programs, struct-level encapsulation is the practically meaningful boundary.

## Traits (interfaces) and polymorphism

Traits declare a contract — a set of method signatures a struct promises to implement. Quantum checks this contract at compile time AND dispatches correctly at runtime:

```quantum
trait Shape {
    fn area(self) -> float
    fn name(self) -> string
}

struct Circle { radius: float }
struct Square { side: float }

impl Shape for Circle {
    fn area(self) -> float { return self.radius * self.radius * 3.14159 }
    fn name(self) -> string { return "Circle" }
}

impl Shape for Square {
    fn area(self) -> float { return self.side * self.side }
    fn name(self) -> string { return "Square" }
}

fn describe(shape: Shape) {
    println(shape.name())   // dispatches to Circle_name or Square_name at runtime
    println(shape.area())   // dispatches to Circle_area or Square_area at runtime
}

fn main() {
    let c = Circle { radius: 3.0 }
    let s = Square { side: 4.0 }
    describe(c)   // Circle, 28.2743
    describe(s)   // Square, 16
}
```

**Compile-time checks**: if `Circle` is missing `perimeter`, or implements `area` with the wrong return type, the compiler catches it immediately with a specific error. If you reference an undeclared trait, that's caught too.

**Runtime dispatch**: `fn describe(shape: Shape)` uses a vtable — a small struct containing a `void*` data pointer and one function pointer per trait method. When you call `describe(c)` with a `Circle`, Quantum automatically wraps `c` in a `Shape_VTable` struct pointing at `Circle_area`, `Circle_name` etc. The `describe` function calls through those pointers, with no knowledge of the concrete type — that's real runtime polymorphism.

```quantum
// This is now real — one function, any struct that implements Shape:
fn measure_all(a: Shape, b: Shape) {
    println(a.area())
    println(b.area())
}
```

## Inheritance — via trait default methods

Quantum doesn't have classical class inheritance (`class Dog extends Animal`) — structs and traits don't map cleanly onto that model. Instead, **traits can provide default method implementations**, which any implementing struct inherits for free unless it overrides them. This is the same approach Rust, Swift, and Kotlin use instead of classical inheritance.

```quantum
trait Animal {
    fn name(self) -> string
    fn sound(self) -> string {
        return "..."          // default — most animals just go "..."
    }
    fn speak(self) -> string {
        return self.name()    // default can call other trait methods on self
    }
}

struct Dog { breed: string }
struct Cat { color: string }

impl Animal for Dog {
    fn name(self) -> string { return "Dog" }
    fn sound(self) -> string { return "Woof" }   // overrides the default
    // speak NOT written — inherits the default, which calls self.name()
}

impl Animal for Cat {
    fn name(self) -> string { return "Cat" }
    // sound NOT written — inherits "..."
    // speak NOT written — inherits the default too
}

fn main() {
    let d = Dog { breed: "Lab" }
    let c = Cat { color: "black" }
    println(d.sound())   // Woof  (Dog overrode it)
    println(d.speak())   // Dog   (inherited default, correctly calls Dog's own name())
    println(c.sound())   // ...   (Cat inherited the default)
    println(c.speak())   // Cat   (inherited default, correctly calls Cat's own name())
}
```

The compiler synthesizes a real, concrete function for every inherited default (e.g. `Cat_sound`) — it isn't simulated or interpreted at runtime, so it works identically to a hand-written method, including through polymorphic dispatch:

```quantum
fn announce(a: Animal) {
    println(a.speak())   // works even though speak() was never written for Dog
}
```

A trait method with no body (just a signature, like `fn name(self) -> string`) is still required — every implementing struct must provide it. Only methods with a `{ ... }` body are optional to override.

## Error handling: `Result<T, E>` and `Option<T>`

Quantum has no exceptions — failures are values you handle explicitly, similar to Rust. This makes every place a function can fail visible in its signature, and the compiler enforces that you deal with it.

```quantum
fn divide(a: int, b: int) -> Result<int, string> {
    if b == 0 {
        return Err("Cannot divide by zero")
    }
    return Ok(a / b)
}

fn main() {
    let r = divide(10, 2)
    if r.is_ok() {
        println(r.unwrap())       // 5
    } else {
        println(r.error)
    }

    // Or use unwrap_or for a safe fallback instead of branching:
    let safe = divide(5, 0).unwrap_or(-1)
    println(safe)                  // -1
}
```

`Option<T>` works the same way for "value or nothing" instead of "success or failure":

```quantum
fn find_first(arr: [int; 5], len: int, target: int) -> Option<int> {
    let mut i = 0
    while i < len {
        if arr[i] == target { return Some(i) }
        i += 1
    }
    return None
}

fn main() {
    let nums = [10, 20, 30, 40, 50]
    let found = find_first(nums, 5, 30)
    println(found.is_some())   // true
    println(found.unwrap())    // 2
}
```

**Available methods**: `.is_ok()` / `.is_err()` on `Result`, `.is_some()` / `.is_none()` on `Option`, and `.unwrap()`, `.expect()`, `.unwrap_or(default)` on both. You can also read the fields directly: `.is_ok`, `.has_value`, `.value`, `.error`.

### The `?` operator — propagating errors upward

```quantum
fn parse_positive(n: int) -> Result<int, string> {
    if n < 0 { return Err("Number must be positive") }
    return Ok(n)
}

fn double_positive(n: int) -> Result<int, string> {
    let x = parse_positive(n)?   // returns Err early if parse_positive failed
    return Ok(x * 2)
}

fn main() {
    println(double_positive(21).unwrap())     // 42
    println(double_positive(-5).unwrap_or(0)) // 0 — the Err propagated up
}
```

### Beginner-friendly crash messages

Calling `.unwrap()` on an `Err` or `None` doesn't just crash silently — it prints a clear explanation:

```
╔══════════════════════════════════════════╗
║         Quantum Runtime Error            ║
╚══════════════════════════════════════════╝

  ✗ called .unwrap() on an Err value — the operation failed. Check
    .is_ok first, or use a match expression to handle both Ok and
    Err cases.

  Your program stopped here because a value was used that didn't
  exist or wasn't valid.

  How to fix this:
    • Check the value before using it (use .is_ok or .has_value)
    • Use .unwrap_or(default) to provide a fallback
    • Use the ? operator to propagate errors up to the caller
```

### Auto-protection: `#[safe_mode]`

Add `#[safe_mode]` at the top of your file to install a crash handler that catches segfaults and aborts at runtime, explaining what likely went wrong instead of just stopping:

```quantum
#[safe_mode]

fn main() {
    let r = divide(10, 0)
    println(r.unwrap())   // would normally abort — now shows a clear explanation first
}
```

This is meant as a safety net for development and for beginners, not a substitute for actually handling errors with `Result`/`Option`/`?` — it explains crashes, it doesn't prevent the underlying bug.

**Implementation note**: `Result`/`Option` use a single universal C struct representation (`{ bool; int64_t; const char* }`) rather than per-type generated types, since Quantum's codegen already represents every primitive through `int64_t`. This keeps codegen simple but means there's currently no compile-time guarantee that you read back the same type you stored — reading `.value` as the wrong type is undefined behavior, the same caveat that already applies to Quantum's other `int64_t`-backed dynamic-feeling features.

## Structs and methods

Quantum supports full struct types — definition, construction, field access, field mutation, and now **methods** via `impl` blocks.

```quantum
struct Point {
    x: int,
    y: int,
}

impl Point {
    fn distance_sq(self) -> int {
        return self.x * self.x + self.y * self.y
    }
    fn sum(self) -> int {
        return self.x + self.y
    }
}

fn main() {
    let p = Point { x: 3, y: 4 }
    println(p.distance_sq())   // 25
    println(p.sum())           // 7
}
```

Methods desugar to free functions under the hood (`Point.distance_sq` becomes a function named `Point_distance_sq` with `self` as an explicit first parameter) — there's no hidden runtime magic, similar to how C++ implements methods.

**Important constraint: `self` is passed by value, not by reference.** Quantum has no pointer/reference types exposed at the language level yet, so a method that mutates `self` only mutates its own local copy — the caller's struct is unaffected:

```quantum
impl Point {
    // This method runs without error, but DOES NOT mutate the caller's p —
    // self.x and self.y here are a copy, discarded when the method returns.
    fn translate(self, dx: int, dy: int) {
        self.x = self.x + dx   // only affects the local copy
        self.y = self.y + dy
    }
}
```

Until reference/pointer types exist, **only read-only methods are reliable**. For anything that needs to "mutate" a struct, write a free function that takes the struct, returns a new one, and reassign at the call site:

```quantum
fn translated(p: Point, dx: int, dy: int) -> Point {
    return Point { x: p.x + dx, y: p.y + dy }
}

fn main() {
    let p = Point { x: 3, y: 4 }
    let p2 = translated(p, 1, 2)
    println(p2.x)   // 4
    println(p2.y)   // 6
}
```

**Other limitations of the current struct/method implementation:**
- No inheritance or interfaces/traits.
- All fields must be initialized at construction — no default values.
- Nested struct field access (`p.inner.x`) doesn't propagate type inference through more than one level.
