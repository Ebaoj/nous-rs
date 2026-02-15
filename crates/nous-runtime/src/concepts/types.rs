use nous_core::types::{Embedding, Timestamp};

/// A concept node in the emergent graph.
///
/// Represents a cluster of semantically similar messages,
/// discovered through agglomerative clustering.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConceptNode {
    /// Unique concept identifier.
    pub id: String,

    /// Human-readable label (e.g., "user_management", "database_queries").
    pub label: String,

    /// Centroid embedding of all messages in this cluster.
    pub embedding: Embedding,

    /// IDs of messages belonging to this concept.
    pub message_ids: Vec<String>,

    /// How well-defined this cluster is (0-1, higher = tighter cluster).
    pub strength: f64,

    /// Keywords extracted from messages in this concept.
    pub keywords: Vec<String>,

    /// When this concept was last updated.
    pub last_updated: Timestamp,
}

/// An edge connecting two concepts.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConceptEdge {
    /// Source concept ID.
    pub source: String,

    /// Target concept ID.
    pub target: String,

    /// Cosine similarity between concept centroids.
    pub similarity: f64,

    /// Type of relationship (inferred from message patterns).
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub relation_type: Option<String>,
}

/// Metadata about a concept graph.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConceptGraphMetadata {
    /// Total messages analyzed.
    pub message_count: usize,
    /// Clustering threshold used.
    pub cluster_threshold: f64,
    /// When the graph was last updated.
    pub last_updated: Timestamp,
    /// Version of the graph (increments on rebuild/update).
    pub version: u64,
}

/// The complete concept graph.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConceptGraph {
    /// All concept nodes.
    pub nodes: Vec<ConceptNode>,
    /// All edges between concepts.
    pub edges: Vec<ConceptEdge>,
    /// Graph metadata.
    pub metadata: ConceptGraphMetadata,
}

/// Options for building a concept graph.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConceptGraphOptions {
    /// Minimum similarity to be in the same cluster (default: 0.75).
    pub cluster_threshold: f64,
    /// Minimum messages per concept (default: 1 for the Rust version;
    /// TS used 3 but that requires large datasets).
    pub min_cluster_size: usize,
    /// Minimum edge similarity to include (default: 0.5).
    pub min_edge_similarity: f64,
    /// Maximum concepts to create (default: 50).
    pub max_concepts: usize,
    /// Extract keywords from messages? (default: true).
    pub extract_keywords: bool,
}

impl Default for ConceptGraphOptions {
    fn default() -> Self {
        Self {
            cluster_threshold: 0.75,
            min_cluster_size: 1,
            min_edge_similarity: 0.5,
            max_concepts: 50,
            extract_keywords: true,
        }
    }
}

/// Result of finding a concept matching a query.
#[derive(Debug, Clone)]
pub struct ConceptMatch {
    /// The matched concept.
    pub concept: ConceptNode,
    /// Similarity to the query embedding.
    pub similarity: f64,
}

/// Statistics about a concept graph.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConceptGraphStats {
    /// Number of concepts.
    pub concept_count: usize,
    /// Number of edges.
    pub edge_count: usize,
    /// Average messages per concept.
    pub avg_messages_per_concept: f64,
    /// Average concept strength.
    pub avg_strength: f64,
    /// Most connected concepts (by edge count).
    pub top_connected: Vec<ConnectedConcept>,
    /// Isolated concepts (no edges).
    pub isolated_concepts: Vec<String>,
}

/// A concept label with its edge count.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConnectedConcept {
    pub label: String,
    pub edge_count: usize,
}
