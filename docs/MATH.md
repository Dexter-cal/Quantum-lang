# Quantum Standard Library — Mathematics (`math.qtm`)

The `math` module provides C libm FFI bindings for trigonometry, powers, roots, logarithms, rounding, and math constants.

## Import
```quantum
import math
```

## Functions

- `math_sqrt(x: float) -> float` — Square root $\sqrt{x}$.
- `math_pow(x: float, y: float) -> float` — Power $x^y$.
- `math_sin(x: float) -> float` — Sine in radians.
- `math_cos(x: float) -> float` — Cosine in radians.
- `math_tan(x: float) -> float` — Tangent in radians.
- `math_floor(x: float) -> float` — Floor $\lfloor x \rfloor$.
- `math_ceil(x: float) -> float` — Ceiling $\lceil x \rceil$.
- `math_round(x: float) -> float` — Round to nearest integer.
- `math_clamp(x: float, min_val: float, max_val: float) -> float` — Clamp $x \in [\text{min}, \text{max}]$.
- `math_pi() -> float` — Returns $\pi \approx 3.14159265$.

## Example
```quantum
import math

fn main() {
    let x = 16.0
    println("Sqrt(16): " + math.math_sqrt(x).to_string())
    println("Pi: " + math.math_pi().to_string())
}
```
