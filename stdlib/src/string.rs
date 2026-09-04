// Quantum String Utilities

/// Split string by delimiter
pub fn split(s: &str, delimiter: &str) -> Vec<String> {
    s.split(delimiter).map(|s| s.to_string()).collect()
}

/// Join strings with separator
pub fn join(strings: &[String], separator: &str) -> String {
    strings.join(separator)
}

/// Convert string to uppercase
pub fn to_upper(s: &str) -> String {
    s.to_uppercase()
}

/// Convert string to lowercase
pub fn to_lower(s: &str) -> String {
    s.to_lowercase()
}

/// Trim whitespace from both ends
pub fn trim(s: &str) -> String {
    s.trim().to_string()
}

/// Check if string starts with prefix
pub fn starts_with(s: &str, prefix: &str) -> bool {
    s.starts_with(prefix)
}

/// Check if string ends with suffix
pub fn ends_with(s: &str, suffix: &str) -> bool {
    s.ends_with(suffix)
}

/// Check if string contains substring
pub fn contains(s: &str, substring: &str) -> bool {
    s.contains(substring)
}

/// Replace all occurrences
pub fn replace(s: &str, from: &str, to: &str) -> String {
    s.replace(from, to)
}

/// Get string length
pub fn length(s: &str) -> usize {
    s.len()
}
