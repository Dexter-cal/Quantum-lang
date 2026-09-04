// Quantum Standard Library
// Core functionality that all Quantum programs can use

pub mod collections;
pub mod io;
pub mod string;
pub mod math;

/// Print a value to stdout with newline
pub fn println(s: &str) {
    println!("{}", s);
}

/// Print a value to stdout without newline
pub fn print(s: &str) {
    print!("{}", s);
}

/// Convert any value to string
pub fn to_string<T: std::fmt::Display>(value: T) -> String {
    format!("{}", value)
}

/// Assert that a condition is true
pub fn assert(condition: bool, message: &str) {
    if !condition {
        panic!("Assertion failed: {}", message);
    }
}

/// Assert that two values are equal
pub fn assert_eq<T: PartialEq + std::fmt::Debug>(left: T, right: T) {
    if left != right {
        panic!("Assertion failed: {:?} != {:?}", left, right);
    }
}
