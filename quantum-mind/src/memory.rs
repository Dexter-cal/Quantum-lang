//! Memory System — QuantumMind remembers conversations, projects, and learned patterns

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub role: String,       // "user" or "mind"
    pub content: String,
    pub timestamp: u64,
    pub context_tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ProjectMemory {
    pub name: String,
    pub description: String,
    pub files_created: Vec<String>,
    pub last_accuracy: Option<f64>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct LearnedPattern {
    pub trigger: String,
    pub response: String,
    pub times_used: usize,
    pub success_rate: f64,
}

pub struct Memory {
    pub conversation_history: Vec<ConversationTurn>,
    pub projects: HashMap<String, ProjectMemory>,
    pub learned_patterns: Vec<LearnedPattern>,
    pub user_preferences: HashMap<String, String>,
    pub session_context: HashMap<String, String>,
}

impl Memory {
    pub fn new() -> Self {
        Self {
            conversation_history: Vec::new(),
            projects: HashMap::new(),
            learned_patterns: Vec::new(),
            user_preferences: HashMap::new(),
            session_context: HashMap::new(),
        }
    }

    pub fn remember_turn(&mut self, role: &str, content: &str) {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs()).unwrap_or(0);
        // Extract context tags
        let tags: Vec<String> = content.split_whitespace()
            .filter(|w| w.len() > 4)
            .take(5)
            .map(|w| w.to_lowercase())
            .collect();
        self.conversation_history.push(ConversationTurn {
            role: role.to_string(),
            content: content.to_string(),
            timestamp: ts,
            context_tags: tags,
        });
        // Keep last 50 turns
        if self.conversation_history.len() > 50 {
            self.conversation_history.remove(0);
        }
    }

    pub fn remember_project(&mut self, name: &str, description: &str, files: Vec<String>) {
        self.projects.insert(name.to_string(), ProjectMemory {
            name: name.to_string(),
            description: description.to_string(),
            files_created: files,
            last_accuracy: None,
            notes: Vec::new(),
        });
    }

    pub fn update_project_accuracy(&mut self, name: &str, accuracy: f64) {
        if let Some(p) = self.projects.get_mut(name) {
            p.last_accuracy = Some(accuracy);
        }
    }

    pub fn learn_pattern(&mut self, trigger: &str, response: &str) {
        // Check if pattern already exists
        if let Some(p) = self.learned_patterns.iter_mut().find(|p| p.trigger == trigger) {
            p.times_used += 1;
            return;
        }
        self.learned_patterns.push(LearnedPattern {
            trigger: trigger.to_string(),
            response: response.to_string(),
            times_used: 1,
            success_rate: 1.0,
        });
    }

    pub fn recall_pattern(&self, query: &str) -> Option<&LearnedPattern> {
        let q = query.to_lowercase();
        self.learned_patterns.iter()
            .find(|p| q.contains(&p.trigger.to_lowercase()))
    }

    pub fn set_preference(&mut self, key: &str, value: &str) {
        self.user_preferences.insert(key.to_string(), value.to_string());
        println!("  💾 Preference saved: {} = {}", key, value);
    }

    pub fn get_preference(&self, key: &str) -> Option<&String> {
        self.user_preferences.get(key)
    }

    pub fn set_context(&mut self, key: &str, value: &str) {
        self.session_context.insert(key.to_string(), value.to_string());
    }

    pub fn get_context(&self, key: &str) -> Option<&String> {
        self.session_context.get(key)
    }

    pub fn recent_topics(&self, n: usize) -> Vec<String> {
        self.conversation_history.iter().rev().take(n)
            .flat_map(|t| t.context_tags.iter().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter().collect()
    }

    pub fn summary(&self) -> String {
        format!(
            "Session: {} turns | {} projects | {} patterns learned | {} preferences",
            self.conversation_history.len(),
            self.projects.len(),
            self.learned_patterns.len(),
            self.user_preferences.len()
        )
    }
}
