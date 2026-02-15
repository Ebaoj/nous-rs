use nous_core::types::{Confidence, Embedding};

/// Precondition that must be satisfied before execution.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PreCondition {
    /// Condition identifier
    pub id: String,

    /// Human-readable description
    pub description: String,

    /// Semantic embedding for matching context state
    pub embedding: Embedding,

    /// Minimum similarity threshold for matching (0-1)
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub threshold: Option<f64>,

    /// Is this condition required or optional?
    pub required: bool,
}

/// Postcondition guaranteed after successful execution.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PostCondition {
    /// Condition identifier
    pub id: String,

    /// Human-readable description
    pub description: String,

    /// Semantic embedding for verification
    pub embedding: Embedding,

    /// Guaranteed confidence in the postcondition (0-1)
    pub guaranteed_confidence: Confidence,
}
