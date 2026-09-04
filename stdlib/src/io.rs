// Quantum I/O - File operations, reading, writing
use std::fs::File as StdFile;
use std::io::{Read, Write, BufReader, BufRead};
use std::path::Path;

pub struct File {
    inner: StdFile,
}

impl File {
    pub fn open(path: &str) -> Result<Self, std::io::Error> {
        Ok(Self {
            inner: StdFile::open(path)?
        })
    }

    pub fn create(path: &str) -> Result<Self, std::io::Error> {
        Ok(Self {
            inner: StdFile::create(path)?
        })
    }

    pub fn read_to_string(&mut self) -> Result<String, std::io::Error> {
        let mut contents = String::new();
        self.inner.read_to_string(&mut contents)?;
        Ok(contents)
    }

    pub fn write(&mut self, data: &str) -> Result<(), std::io::Error> {
        self.inner.write_all(data.as_bytes())
    }
}

/// Read entire file to string
pub fn read_file(path: &str) -> Result<String, std::io::Error> {
    std::fs::read_to_string(path)
}

/// Write string to file
pub fn write_file(path: &str, contents: &str) -> Result<(), std::io::Error> {
    std::fs::write(path, contents)
}

/// Read file line by line
pub fn read_lines(path: &str) -> Result<Vec<String>, std::io::Error> {
    let file = StdFile::open(path)?;
    let reader = BufReader::new(file);
    let lines: Result<Vec<_>, _> = reader.lines().collect();
    lines
}

/// Check if file exists
pub fn exists(path: &str) -> bool {
    Path::new(path).exists()
}

/// Delete a file
pub fn remove_file(path: &str) -> Result<(), std::io::Error> {
    std::fs::remove_file(path)
}
