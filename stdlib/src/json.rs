// Quantum JSON Module - Parse and serialize JSON
use serde::{Serialize, Deserialize};
use serde_json;
use std::collections::HashMap;

/// JSON value type
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(HashMap<String, Value>),
}

impl Value {
    /// Parse JSON string to Value
    pub fn parse(json: &str) -> Result<Self, String> {
        serde_json::from_str(json)
            .map_err(|e| format!("JSON parse error: {}", e))
    }

    /// Convert Value to JSON string
    pub fn stringify(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "null".to_string())
    }

    /// Convert Value to pretty JSON string
    pub fn stringify_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "null".to_string())
    }

    /// Check if value is null
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Check if value is bool
    pub fn is_bool(&self) -> bool {
        matches!(self, Value::Bool(_))
    }

    /// Check if value is number
    pub fn is_number(&self) -> bool {
        matches!(self, Value::Number(_))
    }

    /// Check if value is string
    pub fn is_string(&self) -> bool {
        matches!(self, Value::String(_))
    }

    /// Check if value is array
    pub fn is_array(&self) -> bool {
        matches!(self, Value::Array(_))
    }

    /// Check if value is object
    pub fn is_object(&self) -> bool {
        matches!(self, Value::Object(_))
    }

    /// Get as bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Get as number
    pub fn as_number(&self) -> Option<f64> {
        match self {
            Value::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Get as string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get as array
    pub fn as_array(&self) -> Option<&Vec<Value>> {
        match self {
            Value::Array(arr) => Some(arr),
            _ => None,
        }
    }

    /// Get as object
    pub fn as_object(&self) -> Option<&HashMap<String, Value>> {
        match self {
            Value::Object(obj) => Some(obj),
            _ => None,
        }
    }

    /// Get object property
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(obj) => obj.get(key),
            _ => None,
        }
    }

    /// Get array element
    pub fn get_index(&self, index: usize) -> Option<&Value> {
        match self {
            Value::Array(arr) => arr.get(index),
            _ => None,
        }
    }

    /// Get nested value using dot notation
    pub fn get_path(&self, path: &str) -> Option<&Value> {
        let parts: Vec<&str> = path.split('.').collect();
        let mut current = self;

        for part in parts {
            if let Some(idx) = part.parse::<usize>().ok() {
                current = current.get_index(idx)?;
            } else {
                current = current.get(part)?;
            }
        }

        Some(current)
    }
}

/// JSON Builder for creating JSON values
pub struct Builder {
    value: Value,
}

impl Builder {
    pub fn object() -> Self {
        Self {
            value: Value::Object(HashMap::new()),
        }
    }

    pub fn array() -> Self {
        Self {
            value: Value::Array(Vec::new()),
        }
    }

    pub fn set(&mut self, key: &str, value: Value) -> &mut Self {
        if let Value::Object(ref mut obj) = self.value {
            obj.insert(key.to_string(), value);
        }
        self
    }

    pub fn push(&mut self, value: Value) -> &mut Self {
        if let Value::Array(ref mut arr) = self.value {
            arr.push(value);
        }
        self
    }

    pub fn build(self) -> Value {
        self.value
    }
}

/// Macro-like functions for building JSON
pub fn null() -> Value {
    Value::Null
}

pub fn bool(b: bool) -> Value {
    Value::Bool(b)
}

pub fn number(n: f64) -> Value {
    Value::Number(n)
}

pub fn string(s: &str) -> Value {
    Value::String(s.to_string())
}

pub fn array(values: Vec<Value>) -> Value {
    Value::Array(values)
}

pub fn object(pairs: Vec<(&str, Value)>) -> Value {
    let mut map = HashMap::new();
    for (k, v) in pairs {
        map.insert(k.to_string(), v);
    }
    Value::Object(map)
}

/// JSON Path query
pub fn query(json: &Value, path: &str) -> Vec<&Value> {
    // Simple JSONPath implementation
    let mut results = Vec::new();
    query_recursive(json, path, &mut results);
    results
}

fn query_recursive<'a>(value: &'a Value, path: &str, results: &mut Vec<&'a Value>) {
    if path.is_empty() {
        results.push(value);
        return;
    }

    // TODO: Implement full JSONPath query language
    if let Some(v) = value.get_path(path) {
        results.push(v);
    }
}

/// JSON validation
pub fn validate(json: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(json).is_ok()
}

/// JSON diff
pub fn diff(left: &Value, right: &Value) -> Vec<Difference> {
    let mut diffs = Vec::new();
    diff_recursive("", left, right, &mut diffs);
    diffs
}

#[derive(Debug, Clone, PartialEq)]
pub enum DiffType {
    Added,
    Removed,
    Modified,
}

#[derive(Debug, Clone)]
pub struct Difference {
    pub path: String,
    pub diff_type: DiffType,
    pub left: Option<Value>,
    pub right: Option<Value>,
}

fn diff_recursive(path: &str, left: &Value, right: &Value, diffs: &mut Vec<Difference>) {
    if left == right {
        return;
    }

    match (left, right) {
        (Value::Object(l), Value::Object(r)) => {
            for (key, left_val) in l {
                let new_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{}.{}", path, key)
                };

                if let Some(right_val) = r.get(key) {
                    diff_recursive(&new_path, left_val, right_val, diffs);
                } else {
                    diffs.push(Difference {
                        path: new_path,
                        diff_type: DiffType::Removed,
                        left: Some(left_val.clone()),
                        right: None,
                    });
                }
            }

            for (key, right_val) in r {
                if !l.contains_key(key) {
                    let new_path = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{}.{}", path, key)
                    };

                    diffs.push(Difference {
                        path: new_path,
                        diff_type: DiffType::Added,
                        left: None,
                        right: Some(right_val.clone()),
                    });
                }
            }
        }
        _ => {
            diffs.push(Difference {
                path: path.to_string(),
                diff_type: DiffType::Modified,
                left: Some(left.clone()),
                right: Some(right.clone()),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json() {
        let json = r#"{"name": "Alice", "age": 30}"#;
        let value = Value::parse(json).unwrap();

        assert!(value.is_object());
        assert_eq!(value.get("name").unwrap().as_string(), Some("Alice"));
        assert_eq!(value.get("age").unwrap().as_number(), Some(30.0));
    }

    #[test]
    fn test_stringify() {
        let value = object(vec![
            ("name", string("Bob")),
            ("age", number(25.0)),
        ]);

        let json = value.stringify();
        assert!(json.contains("Bob"));
        assert!(json.contains("25"));
    }

    #[test]
    fn test_builder() {
        let value = Builder::object()
            .set("name", string("Charlie"))
            .set("age", number(35.0))
            .build();

        assert_eq!(value.get("name").unwrap().as_string(), Some("Charlie"));
    }

    #[test]
    fn test_path_query() {
        let value = object(vec![
            ("user", object(vec![
                ("name", string("Dave")),
                ("address", object(vec![
                    ("city", string("NYC")),
                ])),
            ])),
        ]);

        assert_eq!(
            value.get_path("user.address.city").unwrap().as_string(),
            Some("NYC")
        );
    }
}
