# Quantum Standard Library — Special Math & Interpolation (`math_special.qtm`)

The `math_special` module provides linear interpolation, smoothstep, range mapping, and bitwise population count.

## Import
```quantum
import math_special
```

## Functions

### `lerp(a: float, b: float, t: float) -> float`
Linear interpolation between `a` and `b` by factor `t`.
- **Formula:** $a + (b - a) \cdot t$

### `clamp01(x: float) -> float`
Clamps $x$ to the range $[0.0, 1.0]$.

### `smoothstep(edge0: float, edge1: float, x: float) -> float`
Hermite interpolation between `edge0` and `edge1`.

### `map_range(val: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float`
Maps a value from input range $[in\_min, in\_max]$ to output range $[out\_min, out\_max]$.

### `popcount(n: int) -> int`
Counts the number of set bits (1s) in integer `n`.

## Example
```quantum
import math_special

fn main() {
    let interpolated = math_special.lerp(10.0, 20.0, 0.5)
    println("Lerp: " + interpolated.to_string()) // 15.0
    let bits = math_special.popcount(7)
    println("Popcount(7): " + bits.to_string()) // 3
}
```
