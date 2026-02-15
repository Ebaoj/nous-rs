use nous_core::logical_atoms::LogicalAtoms;
use nous_core::types::{Embedding, EmbeddingModel, MessageId};

use super::contracts::Contracts;
use super::intent::Intent;
use super::meta::ProtocolMeta;
use super::param::TypedParam;
use super::reference::SemanticReference;

/// NousProtocolMessage -- Execution-ready message with contracts.
///
/// This is the main message type for Nous v0.2/v0.3, extending the
/// original NousMessage concept with:
/// - Explicit intent with confidence
/// - Typed parameters with uncertainty
/// - Pre/post condition contracts
/// - Semantic references
/// - Structured fallbacks
/// - Logical atoms for hybrid representation (v0.3)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NousProtocolMessage {
    /// Unique message identifier
    pub id: MessageId,

    /// Semantic embedding of the entire message
    pub embedding: Embedding,

    /// Model used for embedding
    pub embedding_model: EmbeddingModel,

    /// The intent/action with confidence
    pub intent: Intent,

    /// Typed parameters with uncertainty
    pub params: Vec<TypedParam>,

    /// Contract specifications
    pub contracts: Contracts,

    /// References to other messages/concepts
    pub references: Vec<SemanticReference>,

    /// Message metadata
    pub meta: ProtocolMeta,

    /// Optional human-readable text representation
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub text: Option<String>,

    /// Logical atoms extracted from the message (v0.3 hybrid representation)
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub logical_atoms: Option<LogicalAtoms>,
}
