// Quantum Collections - Vec, HashMap, HashSet
use std::collections::HashMap as StdHashMap;
use std::collections::HashSet as StdHashSet;

/// Dynamic array type
pub struct Vec<T> {
    inner: std::vec::Vec<T>,
}

impl<T> Vec<T> {
    pub fn new() -> Self {
        Self { inner: std::vec::Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self { inner: std::vec::Vec::with_capacity(capacity) }
    }

    pub fn push(&mut self, value: T) {
        self.inner.push(value);
    }

    pub fn pop(&mut self) -> Option<T> {
        self.inner.pop()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.inner.get(index)
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

/// Hash map type
pub struct HashMap<K, V> {
    inner: StdHashMap<K, V>,
}

impl<K: std::hash::Hash + Eq, V> HashMap<K, V> {
    pub fn new() -> Self {
        Self { inner: StdHashMap::new() }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.inner.insert(key, value)
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.inner.get(key)
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.inner.remove(key)
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.inner.contains_key(key)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// Hash set type
pub struct HashSet<T> {
    inner: StdHashSet<T>,
}

impl<T: std::hash::Hash + Eq> HashSet<T> {
    pub fn new() -> Self {
        Self { inner: StdHashSet::new() }
    }

    pub fn insert(&mut self, value: T) -> bool {
        self.inner.insert(value)
    }

    pub fn contains(&self, value: &T) -> bool {
        self.inner.contains(value)
    }

    pub fn remove(&mut self, value: &T) -> bool {
        self.inner.remove(value)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}
