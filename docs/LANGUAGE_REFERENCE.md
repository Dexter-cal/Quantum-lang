# Quantum Language v2.1 — Complete Language Reference

## Overview
Quantum is a compiled, statically-typed, systems programming language with first-class
ML/AI capabilities. It compiles to native machine code via C, runs fast, and ships with
a full ML framework (`quantumai`) built in.

---

## 1. Syntax Basics

### Comments
```quantum
// Single-line comment
/* Multi-line
   comment */
```

### Variables
```quantum
let x = 42              // immutable, type inferred
let y: float = 3.14     // explicit type
let mut z = 0           // mutable
let name: string = "Quantum"
let flag: bool = true
```

### Types
| Type     | Description         | Example              |
|----------|---------------------|----------------------|
| `int`    | 64-bit integer      | `42`, `-7`           |
| `float`  | 64-bit double       | `3.14`, `-0.5`       |
| `bool`   | Boolean             | `true`, `false`      |
| `string` | UTF-8 string        | `"hello"`            |
| `[T]`    | Array of T          | `[1, 2, 3]`          |
| `(A, B)` | Tuple               | `(1, "one")`         |
| `T?`     | Optional            | `Some(42)`, `None`   |
| `Result<T, E>` | Result type   | `Ok(v)`, `Err(e)`   |

---

## 2. Functions

```quantum
// Basic function
fn add(a: int, b: int) -> int {
    return a + b
}

// Implicit return (last expression)
fn square(n: int) -> int {
    n * n
}

// Void function
fn greet(name: string) {
    println("Hello, " + name + "!")
}

// Recursive function
fn factorial(n: int) -> int {
    if n <= 1 { return 1 }
    return n * factorial(n - 1)
}

// Generic function
fn max<T: Comparable>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

// Higher-order function
fn apply(f: fn(int) -> int, x: int) -> int {
    f(x)
}

// Variadic
fn sum(args: ...int) -> int {
    let mut total = 0
    for x in args { total = total + x }
    total
}
```

---

## 3. Control Flow

### If / Else
```quantum
if x > 10 {
    println("big")
} else if x > 5 {
    println("medium")
} else {
    println("small")
}

// If as expression
let label = if x > 0 { "positive" } else { "non-positive" }
```

### Match
```quantum
match value {
    0       => println("zero"),
    1 | 2   => println("one or two"),
    3..10   => println("three to ten"),
    _       => println("other"),
}

// Match with binding
match shape {
    Shape::Circle(r)       => println("circle r=" + r),
    Shape::Rect(w, h)      => println("rect " + w + "x" + h),
    Shape::Triangle(a,b,c) => println("triangle"),
}
```

### Loops
```quantum
// For range
for i in 0..10 {
    print(i + " ")
}

// For in collection
for item in my_list {
    println(item)
}

// While
let mut n = 0
while n < 10 {
    n = n + 1
}

// Infinite loop with break
loop {
    let input = read_line()
    if input == "quit" { break }
    println("You said: " + input)
}

// Loop with continue
for i in 0..20 {
    if i % 2 == 0 { continue }
    println(i)  // only odd numbers
}
```

---

## 4. Structs & Enums

### Structs
```quantum
struct Point {
    x: float,
    y: float,
}

impl Point {
    fn new(x: float, y: float) -> Point {
        Point { x: x, y: y }
    }

    fn distance(self, other: Point) -> float {
        sqrt((self.x - other.x)^2 + (self.y - other.y)^2)
    }

    fn to_string(self) -> string {
        "(" + self.x + ", " + self.y + ")"
    }
}

let p1 = Point::new(0.0, 0.0)
let p2 = Point::new(3.0, 4.0)
println(p1.distance(p2))   // 5.0
```

### Enums
```quantum
enum Color {
    Red,
    Green,
    Blue,
    Custom(int, int, int),  // RGB
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}

enum Shape {
    Circle(float),
    Rectangle(float, float),
    Triangle(float, float, float),
}
```

---

## 5. Error Handling

```quantum
fn divide(a: float, b: float) -> Result<float, string> {
    if b == 0.0 {
        return Err("Division by zero")
    }
    Ok(a / b)
}

// Using match
match divide(10.0, 2.0) {
    Ok(v)  => println("Result: " + v),
    Err(e) => println("Error: " + e),
}

// Using ? operator (propagates errors)
fn compute() -> Result<float, string> {
    let a = divide(10.0, 2.0)?
    let b = divide(a, 3.0)?
    Ok(a + b)
}

// Unwrap with default
let val = divide(5.0, 0.0).unwrap_or(0.0)
```

---

## 6. Collections

```quantum
// Arrays
let nums: [int] = [1, 2, 3, 4, 5]
let first = nums[0]
let len   = nums.len()
nums.push(6)
nums.pop()
let sliced = nums[1..3]

// Maps / Dictionaries
let mut scores: {string: int} = {}
scores["alice"] = 95
scores["bob"]   = 87
let a_score = scores.get("alice").unwrap_or(0)
for (name, score) in scores { println(name + ": " + score) }

// Sets
let mut seen: Set<int> = Set::new()
seen.insert(1)
seen.contains(1)   // true

// Functional operations
let doubled  = nums.map(|x| x * 2)
let evens    = nums.filter(|x| x % 2 == 0)
let total    = nums.reduce(0, |acc, x| acc + x)
let sorted   = nums.sort_by(|a, b| a - b)
let flat     = [[1,2],[3,4]].flat_map(|x| x)
let any_neg  = nums.any(|x| x < 0)
let all_pos  = nums.all(|x| x > 0)
let zipped   = nums.zip(other_list)
```

---

## 7. Traits

```quantum
trait Printable {
    fn print(self)
    fn to_string(self) -> string  // must implement
}

trait Numeric: Printable {
    fn add(self, other: Self) -> Self
    fn zero() -> Self
}

// Implement for a type
impl Printable for Point {
    fn print(self) {
        println("(" + self.x + ", " + self.y + ")")
    }
    fn to_string(self) -> string {
        "(" + self.x + ", " + self.y + ")"
    }
}
```

---

## 8. Closures & Lambdas

```quantum
let double   = |x| x * 2
let add      = |a, b| a + b
let greet    = |name| "Hello, " + name + "!"

// Capturing variables
let factor = 3
let multiply = |x| x * factor   // captures factor

// Passing closures
fn apply_twice(f: fn(int) -> int, x: int) -> int {
    f(f(x))
}
println(apply_twice(double, 5))  // 20

// Returning closures
fn make_adder(n: int) -> fn(int) -> int {
    |x| x + n
}
let add5 = make_adder(5)
println(add5(10))  // 15
```

---

## 9. Concurrency

```quantum
// Threads
let handle = spawn {
    println("Running in thread!")
}
handle.join()

// Channels
let (tx, rx) = channel::<int>()
spawn { tx.send(42) }
let value = rx.recv()

// Async / Await
async fn fetch(url: string) -> string {
    let response = await http.get(url)
    response.text()
}

async fn main() {
    let result = await fetch("https://api.example.com")
    println(result)
}

// Parallel map (auto-parallelised)
let results = big_list.par_map(|x| expensive_compute(x))

// Mutex
let shared = Mutex::new(0)
spawn {
    let mut val = shared.lock()
    *val += 1
}
```

---

## 10. Modules & Imports

```quantum
// Import standard library
import std.io
import std.math as math
import std.collections.{HashMap, Vec, HashSet}

// Import QuantumAI
import quantumai as qai
import quantumai.ml as ml
import quantumai.nlp as nlp
import quantumai.data as data

// Define a module
mod geometry {
    pub struct Point { pub x: float, pub y: float }
    pub fn origin() -> Point { Point { x: 0.0, y: 0.0 } }
}

// Use it
let p = geometry::origin()
```

---

## 11. Generics

```quantum
// Generic struct
struct Stack<T> {
    items: [T],
}

impl<T> Stack<T> {
    fn new() -> Stack<T> { Stack { items: [] } }
    fn push(mut self, item: T) { self.items.push(item) }
    fn pop(mut self) -> T? { self.items.pop() }
    fn is_empty(self) -> bool { self.items.len() == 0 }
}

// Generic function with constraints
fn sort_and_print<T: Comparable + Printable>(items: [T]) {
    let sorted = items.sort()
    for item in sorted { item.print() }
}
```

---

## 12. Macros

```quantum
// Define a macro
macro_rules! vec {
    ($($x:expr),*) => {
        {
            let mut v = []
            $(v.push($x);)*
            v
        }
    }
}

// Use it
let v = vec![1, 2, 3, 4, 5]

// Built-in macros
println!("Hello, {}!", name)
assert!(x > 0, "x must be positive")
assert_eq!(result, expected)
dbg!(my_value)     // prints value + file/line info
todo!("implement this")
unreachable!("should never get here")
```

---

## 13. Pattern Matching (Advanced)

```quantum
// Destructuring
let (a, b, c) = (1, 2, 3)
let Point { x, y } = point

// Nested patterns
match data {
    Some(Ok(value))      => println("Got: " + value),
    Some(Err(e))         => println("Error: " + e),
    None                 => println("Nothing"),
}

// Guards
match score {
    n if n >= 90 => println("A"),
    n if n >= 80 => println("B"),
    n if n >= 70 => println("C"),
    _            => println("F"),
}

// @ bindings
match value {
    n @ 1..=5 => println("small: " + n),
    n @ 6..=10 => println("medium: " + n),
    n => println("large: " + n),
}
```

---

## 14. Built-in Functions

```quantum
// Output
print("text")         // no newline
println("text")       // with newline
eprint("error")       // stderr
eprintln("error")     // stderr + newline

// Math
sqrt(x)   abs(x)   pow(x, y)
sin(x)    cos(x)   tan(x)
floor(x)  ceil(x)  round(x)
min(a, b) max(a, b)
log(x)    log2(x)  log10(x)
PI        E        TAU

// String
len(s)
s.to_uppercase()  s.to_lowercase()
s.trim()          s.split(" ")
s.replace(a, b)   s.contains(sub)
s.starts_with(p)  s.ends_with(p)
s.parse::<int>()  s.parse::<float>()
format!("{}: {}", key, value)

// I/O
read_line() -> string
read_file(path) -> string
write_file(path, content)
parse_csv(path) -> [[string]]
parse_json(text) -> Value
```
