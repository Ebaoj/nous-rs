use nous_core::types::{Embedding, EmbeddingModel, MessageId, Timestamp};

use crate::error::{ProtocolError, ProtocolResult};
use crate::types::{
    Contracts, Intent, IntentConfidence, NousProtocolMessage, ProtocolMeta, ProtocolVersion,
    SemanticReference, SenderInfo, TypedParam,
};

/// Main builder for `NousProtocolMessage`.
/// Provides a fluent API for constructing messages.
#[derive(Debug, Default)]
pub struct NousProtocolBuilder {
    id: Option<MessageId>,
    embedding: Option<Embedding>,
    embedding_model: EmbeddingModel,
    intent: Option<Intent>,
    params: Vec<TypedParam>,
    contracts: Contracts,
    references: Vec<SemanticReference>,
    meta: Option<ProtocolMeta>,
    text: Option<String>,
}

impl NousProtocolBuilder {
    /// Create a new builder instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the message ID. Auto-generated if not provided.
    pub fn id(mut self, id: impl Into<MessageId>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the message embedding and optionally the model.
    pub fn embedding(mut self, embedding: Embedding) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set the embedding model.
    pub fn embedding_model(mut self, model: EmbeddingModel) -> Self {
        self.embedding_model = model;
        self
    }

    /// Set intent directly.
    pub fn intent(mut self, intent: Intent) -> Self {
        self.intent = Some(intent);
        self
    }

    /// Add a typed parameter.
    pub fn param(mut self, param: TypedParam) -> Self {
        self.params.push(param);
        self
    }

    /// Set the full contracts block.
    pub fn contracts(mut self, contracts: Contracts) -> Self {
        self.contracts = contracts;
        self
    }

    /// Add a semantic reference.
    pub fn reference(mut self, reference: SemanticReference) -> Self {
        self.references.push(reference);
        self
    }

    /// Set the full metadata.
    pub fn meta(mut self, meta: ProtocolMeta) -> Self {
        self.meta = Some(meta);
        self
    }

    /// Set optional human-readable text.
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Validate and build the message.
    pub fn build(self) -> ProtocolResult<NousProtocolMessage> {
        let embedding = self
            .embedding
            .ok_or_else(|| ProtocolError::MissingField("embedding".into()))?;

        let intent = self
            .intent
            .ok_or_else(|| ProtocolError::MissingField("intent".into()))?;

        let id = self.id.unwrap_or_else(MessageId::random);

        // Determine protocol version based on confidence type
        let protocol_version = match &intent.confidence {
            IntentConfidence::Map(m) if m.dimensions.is_some() => ProtocolVersion::V0_3,
            _ => ProtocolVersion::V0_2,
        };

        let meta = self.meta.unwrap_or_else(|| ProtocolMeta {
            sender: SenderInfo {
                sender_type: "system".into(),
                id: "nous-builder".into(),
                version: None,
            },
            timestamp: Timestamp::now(),
            conversation_id: None,
            sequence: None,
            protocol_version,
            custom: None,
        });

        Ok(NousProtocolMessage {
            id,
            embedding,
            embedding_model: self.embedding_model,
            intent,
            params: self.params,
            contracts: self.contracts,
            references: self.references,
            meta,
            text: self.text,
            logical_atoms: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use nous_core::types::Confidence;

    use super::*;
    use crate::types::IntentConfidence;

    fn test_embedding() -> Embedding {
        Embedding::new(vec![1.0, 0.0, 0.0]).unwrap()
    }

    fn test_intent() -> Intent {
        Intent {
            action: "test_action".into(),
            confidence: IntentConfidence::Scalar(Confidence::new(0.9)),
            description: None,
            embedding: None,
        }
    }

    #[test]
    fn test_build_minimal_message() {
        let msg = NousProtocolBuilder::new()
            .embedding(test_embedding())
            .intent(test_intent())
            .build()
            .unwrap();

        assert_eq!(msg.intent.action, "test_action");
        assert!(msg.params.is_empty());
        assert!(msg.references.is_empty());
        assert!(msg.text.is_none());
    }

    #[test]
    fn test_build_with_text() {
        let msg = NousProtocolBuilder::new()
            .embedding(test_embedding())
            .intent(test_intent())
            .text("Hello world")
            .build()
            .unwrap();

        assert_eq!(msg.text.as_deref(), Some("Hello world"));
    }

    #[test]
    fn test_build_with_custom_id() {
        let msg = NousProtocolBuilder::new()
            .id("custom-id")
            .embedding(test_embedding())
            .intent(test_intent())
            .build()
            .unwrap();

        assert_eq!(msg.id.as_str(), "custom-id");
    }

    #[test]
    fn test_missing_embedding() {
        let result = NousProtocolBuilder::new().intent(test_intent()).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_intent() {
        let result = NousProtocolBuilder::new()
            .embedding(test_embedding())
            .build();
        assert!(result.is_err());
    }
}
