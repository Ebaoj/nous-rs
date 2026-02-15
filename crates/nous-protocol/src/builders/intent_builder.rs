use nous_core::confidence::ConfidenceMap;
use nous_core::types::{Confidence, Embedding};

use crate::error::{ProtocolError, ProtocolResult};
use crate::types::{Intent, IntentConfidence};

/// Fluent builder for creating `Intent` values.
#[derive(Debug, Default)]
pub struct IntentBuilder {
    action: Option<String>,
    confidence: Option<IntentConfidence>,
    description: Option<String>,
    embedding: Option<Embedding>,
}

impl IntentBuilder {
    /// Create a new builder instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the action identifier.
    pub fn action(mut self, action: impl Into<String>) -> Self {
        self.action = Some(action.into());
        self
    }

    /// Set a scalar confidence value (v0.2 style).
    pub fn confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(IntentConfidence::Scalar(Confidence::new(confidence)));
        self
    }

    /// Set a multidimensional confidence map (v0.3 style).
    pub fn confidence_map(mut self, map: ConfidenceMap) -> Self {
        self.confidence = Some(IntentConfidence::Map(Box::new(map)));
        self
    }

    /// Set a description for debugging.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the semantic embedding for the intent.
    pub fn embedding(mut self, embedding: Embedding) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Build the intent, validating required fields.
    pub fn build(self) -> ProtocolResult<Intent> {
        let action = self
            .action
            .ok_or_else(|| ProtocolError::MissingField("action".into()))?;

        let confidence = self
            .confidence
            .ok_or_else(|| ProtocolError::MissingField("confidence".into()))?;

        Ok(Intent {
            action,
            confidence,
            description: self.description,
            embedding: self.embedding,
        })
    }
}

/// Quick helper to create an intent.
pub fn create_intent(
    action: impl Into<String>,
    confidence: f64,
    description: Option<String>,
    embedding: Option<Embedding>,
) -> ProtocolResult<Intent> {
    let mut builder = IntentBuilder::new().action(action).confidence(confidence);

    if let Some(desc) = description {
        builder = builder.description(desc);
    }

    if let Some(emb) = embedding {
        builder = builder.embedding(emb);
    }

    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_intent_scalar() {
        let intent = IntentBuilder::new()
            .action("query_database")
            .confidence(0.9)
            .description("Test intent")
            .build()
            .unwrap();

        assert_eq!(intent.action, "query_database");
        assert!((intent.confidence.effective_value() - 0.9).abs() < 1e-10);
        assert_eq!(intent.description.as_deref(), Some("Test intent"));
        assert!(intent.embedding.is_none());
    }

    #[test]
    fn test_build_intent_with_map() {
        let map = ConfidenceMap::new(0.85);
        let intent = IntentBuilder::new()
            .action("send_email")
            .confidence_map(map)
            .build()
            .unwrap();

        assert_eq!(intent.action, "send_email");
        assert!((intent.confidence.effective_value() - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_missing_action() {
        let result = IntentBuilder::new().confidence(0.9).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_confidence() {
        let result = IntentBuilder::new().action("test").build();
        assert!(result.is_err());
    }

    #[test]
    fn test_create_intent_helper() {
        let intent = create_intent("test_action", 0.75, Some("desc".into()), None).unwrap();
        assert_eq!(intent.action, "test_action");
        assert!((intent.confidence.effective_value() - 0.75).abs() < 1e-10);
    }
}
