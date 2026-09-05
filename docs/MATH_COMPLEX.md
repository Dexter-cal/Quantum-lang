# Quantum Standard Library — Complex Numbers (`math_complex.qtm`)

The `math_complex` module provides a `Complex` struct and methods for complex number arithmetic, magnitude, and conjugation.

## Import
```quantum
import math_complex
```

## Struct `Complex`

### Fields
- `real: float`
- `imag: float`

### Methods
- `Complex::new(r: float, i: float) -> Complex` — Constructor.
- `c.magnitude() -> float` — Returns $|z| = \sqrt{r^2 + i^2}$.
- `c.conjugate() -> Complex` — Returns $r - i \cdot \text{imag}$.
- `c.add(other: Complex) -> Complex` — Addition.
- `c.sub(other: Complex) -> Complex` — Subtraction.
- `c.mul(other: Complex) -> Complex` — Multiplication.

## Example
```quantum
import math_complex

fn main() {
    let z1 = Complex::new(3.0, 4.0)
    println("Magnitude: " + z1.magnitude().to_string()) // 5.0
    let z2 = z1.conjugate()
    println("Conjugate imag: " + z2.imag.to_string()) // -4.0
}
```
