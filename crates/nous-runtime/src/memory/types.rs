use std::collections::HashMap;

use nous_core::types::{Embedding, EmbeddingModel, Timestamp};

/// Compressed representation of an entire conversation.
///
/// Allows efficient relevance queries without loading full message history.
/// Created by [`super::ConversationMemoryManager::compress`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConversationMemory {
    /// Unique memory identifier.
    pub id: String,

    /// ID of the original conversation.
    pub conversation_id: String,

    /// Compressed embedding representing the entire conversation.
    pub embedding: Embedding,

    /// Model used for the embedding.
    pub embedding_model: EmbeddingModel,

    /// Number of messages in the original conversation.
    pub message_count: usize,

    /// Time range of the conversation (start, end).
    pub time_range: (Timestamp, Timestamp),

    /// Main topics/concepts extracted from the conversation.
    pub topic_summary: Vec<String>,

    /// Key entities mentioned.
    pub entities: Vec<String>,

    /// When this memory was created.
    pub created_at: Timestamp,

    /// Optional metadata.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "HashMap::is_empty")
    )]
    pub metadata: HashMap<String, String>,
}

/// Result of a relevance check between a query and a memory.
#[derive(Debug, Clone)]
pub struct RelevanceResult {
    /// Is the query relevant to this memory?
    pub relevant: bool,

    /// Cosine similarity score.
    pub similarity: f64,

    /// The memory that was checked.
    pub memory: ConversationMemory,
}

/// Options for compressing a conversation into a memory.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CompressionOptions {
    /// Minimum number of messages to compress (default: 5).
    /// Fewer messages will use embedding averaging instead.
    pub min_messages: usize,

    /// Whether to extract topic summaries from intents (default: true).
    pub extract_topics: bool,

    /// Whether to extract entities from params (default: true).
    pub extract_entities: bool,

    /// Custom metadata to attach to the resulting memory.
    pub metadata: HashMap<String, String>,
}

impl Default for CompressionOptions {
    fn default() -> Self {
        Self {
            min_messages: 5,
            extract_topics: true,
            extract_entities: true,
            metadata: HashMap::new(),
        }
    }
}

/// Options for relevance queries.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RelevanceOptions {
    /// Minimum similarity threshold (default: 0.7).
    pub threshold: f64,

    /// Maximum results to return (default: 10).
    pub limit: usize,

    /// Optional time range filter (start, end).
    pub time_range: Option<(Timestamp, Timestamp)>,
}

impl Default for RelevanceOptions {
    fn default() -> Self {
        Self {
            threshold: 0.7,
            limit: 10,
            time_range: None,
        }
    }
}
