use nous_core::types::Embedding;

/// Type of semantic reference.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum ReferenceType {
    Message,
    Concept,
    Entity,
    Context,
}

/// Relationship of a reference to the current message.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum Relation {
    Input,
    Context,
    Dependency,
    Related,
    Supersedes,
}

/// Reference to another message or concept.
/// Can be resolved by ID or by embedding similarity.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SemanticReference {
    /// Reference type
    #[cfg_attr(feature = "serde", serde(rename = "type"))]
    pub ref_type: ReferenceType,

    /// Direct ID reference (if known)
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub id: Option<String>,

    /// Semantic embedding for similarity-based resolution
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub embedding: Option<Embedding>,

    /// Minimum similarity for embedding match
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub min_similarity: Option<f64>,

    /// Relationship to current message
    pub relation: Relation,
}
