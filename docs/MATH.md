# Math Standard Library API Reference (`math`)

Welcome to the **Quantum Math Standard Library Reference**.

## Import Syntax
```quantum
import math
import math_complex
import math_discrete
```

---

## 1. Constants (`math.constants`)
- `PI`: 3.141592653589793
- `TAU`: 6.283185307179586
- `E`: 2.718281828459045
- `PHI`: 1.618033988749895
- `SQRT2`: 1.4142135623730951
- `SQRT3`: 1.7320508075688772

---

## 2. Core Math (`math.core`)
- `math_abs(x: float) -> float` — Absolute value.
- `math_sqrt(x: float) -> float` — Square root.
- `math_pow(x: float, y: float) -> float` — Exponentiation $x^y$.
- `math_min(a: float, b: float) -> float` — Minimum.
- `math_max(a: float, b: float) -> float` — Maximum.
- `math_clamp(x: float, min: float, max: float) -> float` — Clamps $x \in [\text{min}, \text{max}]$.

---

## 3. Rounding (`math.round`)
- `math_floor(x: float) -> float` — Floor $\lfloor x \rfloor$.
- `math_ceil(x: float) -> float` — Ceiling $\lceil x \rceil$.
- `math_round(x: float) -> float` — Nearest integer.

---

## 4. Trigonometry (`math.trig`)
- `math_sin(angle: float) -> float` — Sine (radians).
- `math_cos(angle: float) -> float` — Cosine (radians).
- `math_tan(angle: float) -> float` — Tangent (radians).

---

## 5. Complex Numbers (`import math_complex`)
Struct `Complex { real: float, imag: float }`
- `Complex::new(r: float, i: float) -> Complex`
- `c.magnitude() -> float`
- `c.conjugate() -> Complex`
- `c.add(other: Complex) -> Complex`
- `c.sub(other: Complex) -> Complex`
- `c.mul(other: Complex) -> Complex`

---

## 6. Discrete Mathematics (`import math_discrete`)
- `factorial(n: int) -> int`
- `permutations(n: int, k: int) -> int`
- `combinations(n: int, k: int) -> int`
- `gcd(a: int, b: int) -> int`
- `lcm(a: int, b: int) -> int`

---

## 7. Machine Learning Math (`import ai`)
- `relu(x)`, `sigmoid(x)`, `tanh(x)`, `softmax(vector)`
- Matrix multiplication, gradient computation, and loss functions.

## Example
```quantum
import math
import math_complex
import math_discrete

fn main() {
    println("Pi: " + math.math_pi().to_string())
    let z = Complex::new(3.0, 4.0)
    println("Magnitude: " + z.magnitude().to_string())
    println("5! = " + factorial(5).to_string())
}
```
