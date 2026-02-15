use std::collections::HashMap;

use nous_core::types::Embedding;
use nous_protocol::types::NousProtocolMessage;

use crate::error::RuntimeResult;

use super::context::ExecutionContext;

/// Result returned by a [`MessageHandler`] after successful execution.
#[derive(Debug, Clone)]
pub struct HandlerResult {
    /// Arbitrary result data (serialised to string).
    pub data: Option<String>,
    /// Embedding representing the result, used for postcondition verification.
    pub result_embedding: Embedding,
    /// Confidence in the result (0.0 .. 1.0).
    pub confidence: f64,
}

/// Trait for handling a specific message action.
///
/// Implementations register themselves with a [`HandlerRegistry`] keyed by
/// the action string they handle.
pub trait MessageHandler {
    /// The action string this handler is responsible for.
    fn action(&self) -> &str;

    /// Execute the handler logic against the given message and context.
    fn handle(
        &self,
        message: &NousProtocolMessage,
        context: &ExecutionContext,
    ) -> RuntimeResult<HandlerResult>;
}

/// Registry mapping action strings to boxed handlers.
///
/// The executor consults this registry to find the appropriate handler
/// for each incoming message's `intent.action`.
#[derive(Default)]
pub struct HandlerRegistry {
    handlers: HashMap<String, Box<dyn MessageHandler>>,
}

impl HandlerRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    /// Register a handler for its declared action.
    pub fn register(&mut self, handler: Box<dyn MessageHandler>) {
        self.handlers.insert(handler.action().to_string(), handler);
    }

    /// Look up a handler by action string.
    pub fn get(&self, action: &str) -> Option<&dyn MessageHandler> {
        self.handlers.get(action).map(|h| h.as_ref())
    }

    /// Check whether a handler is registered for the given action.
    pub fn has(&self, action: &str) -> bool {
        self.handlers.contains_key(action)
    }

    /// Number of registered handlers.
    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }

    /// List all registered action strings.
    pub fn actions(&self) -> Vec<&str> {
        self.handlers.keys().map(|s| s.as_str()).collect()
    }
}

impl std::fmt::Debug for HandlerRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HandlerRegistry")
            .field("actions", &self.actions())
            .finish()
    }
}
