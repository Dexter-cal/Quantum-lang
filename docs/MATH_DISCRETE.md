# Quantum Standard Library — Discrete Mathematics (`math_discrete.qtm`)

The `math_discrete` module provides factorials, permutations, combinations, GCD, and LCM.

## Import
```quantum
import math_discrete
```

## Functions

- `factorial(n: int) -> int` — Computes $n!$.
- `permutations(n: int, k: int) -> int` — $P(n, k) = \frac{n!}{(n-k)!}$.
- `combinations(n: int, k: int) -> int` — $C(n, k) = \frac{n!}{k!(n-k)!}$.
- `gcd(a: int, b: int) -> int` — Greatest common divisor using Euclidean algorithm.
- `lcm(a: int, b: int) -> int` — Least common multiple.

## Example
```quantum
import math_discrete

fn main() {
    println("5! = " + factorial(5).to_string()) // 120
    println("C(10, 3) = " + combinations(10, 3).to_string()) // 120
    println("GCD(48, 18) = " + gcd(48, 18).to_string()) // 6
}
```
