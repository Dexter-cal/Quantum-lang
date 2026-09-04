// Quantum Utilities
use std::path::PathBuf;

pub fn find_project_root() -> Option<PathBuf> {
    let mut current = std::env::current_dir().ok()?;

    loop {
        let quantum_toml = current.join("quantum.toml");
        if quantum_toml.exists() {
            return Some(current);
        }

        if !current.pop() {
            return None;
        }
    }
}

pub fn get_cache_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());

    PathBuf::from(home).join(".quantum").join("cache")
}

pub fn ensure_dir(path: &PathBuf) -> std::io::Result<()> {
    if !path.exists() {
        std::fs::create_dir_all(path)?;
    }
    Ok(())
}
