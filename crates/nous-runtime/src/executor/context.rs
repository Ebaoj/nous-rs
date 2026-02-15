use std::collections::HashMap;

use nous_core::types::Embedding;

/// Runtime context passed to handlers during message execution.
///
/// Carries state embeddings for precondition matching and arbitrary
/// key/value data that handlers may read or write.
#[derive(Debug, Clone, Default)]
pub struct ExecutionContext {
    /// Current state embeddings for precondition matching.
    /// Keys are state identifiers (e.g., "db_connected", "user_authenticated").
    pub state_embeddings: HashMap<String, Embedding>,

    /// Arbitrary context data available to handlers.
    pub data: HashMap<String, String>,
}

/// Create an empty execution context.
pub fn create_context() -> ExecutionContext {
    ExecutionContext::default()
}

/// Create an execution context pre-populated with data entries.
pub fn create_context_with_data(data: HashMap<String, String>) -> ExecutionContext {
    ExecutionContext {
        state_embeddings: HashMap::new(),
        data,
    }
}

/// Add a state embedding to an existing context (builder pattern).
pub fn add_state_embedding(
    mut context: ExecutionContext,
    id: impl Into<String>,
    embedding: Embedding,
) -> ExecutionContext {
    context.state_embeddings.insert(id.into(), embedding);
    context
}

impl ExecutionContext {
    /// Create a new empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a state embedding.
    pub fn with_state_embedding(
        mut self,
        id: impl Into<String>,
        embedding: Embedding,
    ) -> Self {
        self.state_embeddings.insert(id.into(), embedding);
        self
    }

    /// Insert a data entry.
    pub fn with_data(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.data.insert(key.into(), value.into());
        self
    }

    /// Get a state embedding by key.
    pub fn get_state_embedding(&self, id: &str) -> Option<&Embedding> {
        self.state_embeddings.get(id)
    }

    /// Get a data value by key.
    pub fn get_data(&self, key: &str) -> Option<&str> {
        self.data.get(key).map(|s| s.as_str())
    }
}
