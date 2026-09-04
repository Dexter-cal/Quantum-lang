//! Conversation engine — handles natural language chat in English

#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn user(content: &str) -> Self { Self { role: "user".to_string(), content: content.to_string() } }
    pub fn mind(content: &str) -> Self { Self { role: "mind".to_string(), content: content.to_string() } }
}

pub struct ConversationEngine;

impl ConversationEngine {
    /// Detect the intent of a message
    pub fn detect_intent(msg: &str) -> Intent {
        let m = msg.to_lowercase();
        // Check FullProject before Generate: "build full project X" should
        // not be caught by the "build " prefix below.
        if m.contains("project") && (m.contains("full") || m.contains("complete") || m.contains("entire")) {
            return Intent::FullProject;
        }
        if m.starts_with("generate ") || m.starts_with("create ") || m.starts_with("build ") || m.starts_with("make ") || m.starts_with("write ") {
            return Intent::Generate;
        }
        if m.starts_with("fix ") || m.starts_with("debug ") || m.contains("error") || m.contains("not working") || m.contains("broken") {
            return Intent::Debug;
        }
        if m.starts_with("explain ") || m.starts_with("what is ") || m.starts_with("what are ") || m.starts_with("how does ") || m.contains("?") {
            return Intent::Explain;
        }
        if m.starts_with("how do i ") || m.starts_with("how to ") || m.starts_with("show me how") {
            return Intent::HowTo;
        }
        if m.contains("train") || m.contains("model") || m.contains("classif") || m.contains("neural") {
            return Intent::MLTask;
        }
        if m.contains("dataset") || m.contains("data") || m.contains("download") || m.contains("csv") {
            return Intent::DataQuery;
        }
        if m.contains("tool") || m.contains("list") || m.contains("available") || m.contains("what can") {
            return Intent::ListCapabilities;
        }
        if m.starts_with("remember ") || m.starts_with("save ") || m.contains("preference") {
            return Intent::Remember;
        }
        if m == "help" || m == "help me" || m == "?" || m == "menu" {
            return Intent::Help;
        }
        Intent::General
    }

    /// Format a response beautifully for the terminal
    pub fn format_response(title: &str, content: &str, response_type: ResponseType) -> String {
        let icon = match response_type {
            ResponseType::Code      => "💻",
            ResponseType::Answer    => "🤖",
            ResponseType::Error     => "❌",
            ResponseType::Success   => "✅",
            ResponseType::Info      => "ℹ️ ",
            ResponseType::Learning  => "🧠",
            ResponseType::Project   => "🚀",
        };
        format!("\n{} {}\n{}\n{}", icon, title, "─".repeat(52), content)
    }
}

#[derive(Debug, PartialEq)]
pub enum Intent {
    Generate,
    Debug,
    Explain,
    HowTo,
    MLTask,
    DataQuery,
    ListCapabilities,
    FullProject,
    Remember,
    Help,
    General,
}

#[derive(Debug)]
pub enum ResponseType {
    Code, Answer, Error, Success, Info, Learning, Project,
}
