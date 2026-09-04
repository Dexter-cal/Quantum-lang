// Quantum Math Utilities

/// Absolute value
pub fn abs(n: f64) -> f64 {
    n.abs()
}

/// Square root
pub fn sqrt(n: f64) -> f64 {
    n.sqrt()
}

/// Power
pub fn pow(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

/// Floor
pub fn floor(n: f64) -> f64 {
    n.floor()
}

/// Ceiling
pub fn ceil(n: f64) -> f64 {
    n.ceil()
}

/// Round
pub fn round(n: f64) -> f64 {
    n.round()
}

/// Minimum of two numbers
pub fn min(a: f64, b: f64) -> f64 {
    a.min(b)
}

/// Maximum of two numbers
pub fn max(a: f64, b: f64) -> f64 {
    a.max(b)
}

/// Constants
pub const PI: f64 = std::f64::consts::PI;
pub const E: f64 = std::f64::consts::E;

/// Trigonometric functions
pub fn sin(n: f64) -> f64 {
    n.sin()
}

pub fn cos(n: f64) -> f64 {
    n.cos()
}

pub fn tan(n: f64) -> f64 {
    n.tan()
}
