use std::collections::HashMap;
use crate::types::{
    AgentIdentifier, Embedding, EmbeddingModel, MessageId, Timestamp,
};
use crate::confidence::map::ConfidenceMap;
use crate::confidence::propagation::Transform;

/// Message content
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MessageContent {
    /// Human-readable text representation (optional)
    pub text: Option<String>,
    /// Embedding vector
    pub embedding: Embedding,
    /// Model used to generate embedding
    pub embedding_model: EmbeddingModel,
}

/// Message context: origin and history
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MessageContext {
    /// Agent that generated the message
    pub origin: AgentIdentifier,
    /// IDs of ancestor messages
    pub lineage: Vec<MessageId>,
    /// Operations applied to produce this message
    pub transformations: Vec<Transform>,
}

/// Core Nous message type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NousMessage {
    /// Unique message identifier
    pub id: MessageId,
    /// Unix timestamp in milliseconds
    pub timestamp: Timestamp,
    /// Message content
    pub content: MessageContent,
    /// Confidence/uncertainty map
    pub confidence: ConfidenceMap,
    /// Origin and history
    pub context: MessageContext,
    /// Arbitrary metadata
    pub meta: HashMap<String, String>,
}

/// Factory for creating NousMessages with consistent defaults
pub struct NousMessageFactory {
    default_agent: AgentIdentifier,
    default_model: EmbeddingModel,
}

impl NousMessageFactory {
    pub fn new(agent: AgentIdentifier, model: EmbeddingModel) -> Self {
        Self {
            default_agent: agent,
            default_model: model,
        }
    }

    /// Create a new root message (no lineage)
    pub fn create(
        &self,
        embedding: Embedding,
        text: Option<String>,
        confidence: ConfidenceMap,
    ) -> NousMessage {
        NousMessage {
            id: MessageId::random(),
            timestamp: Timestamp::now(),
            content: MessageContent {
                text,
                embedding,
                embedding_model: self.default_model.clone(),
            },
            confidence,
            context: MessageContext {
                origin: self.default_agent.clone(),
                lineage: Vec::new(),
                transformations: Vec::new(),
            },
            meta: HashMap::new(),
        }
    }

    /// Derive a new message from an existing one (preserves lineage)
    pub fn derive(
        &self,
        parent: &NousMessage,
        embedding: Embedding,
        text: Option<String>,
        confidence: ConfidenceMap,
        transform: Transform,
    ) -> NousMessage {
        let mut lineage = parent.context.lineage.clone();
        lineage.push(parent.id.clone());

        let mut transformations = parent.context.transformations.clone();
        transformations.push(transform);

        NousMessage {
            id: MessageId::random(),
            timestamp: Timestamp::now(),
            content: MessageContent {
                text,
                embedding,
                embedding_model: self.default_model.clone(),
            },
            confidence,
            context: MessageContext {
                origin: self.default_agent.clone(),
                lineage,
                transformations,
            },
            meta: HashMap::new(),
        }
    }
}

impl NousMessage {
    /// Check if this message has any ancestors
    pub fn has_lineage(&self) -> bool {
        !self.context.lineage.is_empty()
    }

    /// Get the depth of this message in the lineage chain
    pub fn depth(&self) -> usize {
        self.context.lineage.len()
    }

    /// Check if confidence meets a threshold
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.confidence.overall.value() >= threshold
    }
}
