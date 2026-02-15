//! # Nous
//!
//! AI-native communication protocol beyond human language.
//!
//! Agents communicate via embeddings, confidence maps, logical atoms,
//! and contracts (pre/post-conditions).
//!
//! ## Crates
//!
//! - [`core`] — Types, confidence, embeddings math, quantization, logical atoms
//! - [`protocol`] — Messages, builders, validation, contracts
//! - [`runtime`] — Executor, store, memory, concept graphs

pub use nous_core as core;
pub use nous_protocol as protocol;
pub use nous_runtime as runtime;

/// Prelude — import the most commonly needed types with `use nous::prelude::*`
pub mod prelude {
    // Core types
    pub use nous_core::types::{
        AgentIdentifier, AgentType, Confidence, Embedding, EmbeddingModel, MessageId, Timestamp,
    };
    pub use nous_core::types::NousConfig;

    // Errors
    pub use nous_core::error::{NousError, NousResult};

    // Confidence
    pub use nous_core::confidence::map::{ConfidenceMap, ConfidenceSource, DecisionStrategy};
    pub use nous_core::confidence::builder::{create_confidence, ConfidenceBuilder};
    pub use nous_core::confidence::multidimensional::MultidimensionalConfidenceBuilder;

    // Embeddings math
    pub use nous_core::embeddings::math::{
        cosine_similarity, euclidean_distance, merge_embeddings, normalize,
    };

    // Protocol types
    pub use nous_protocol::types::{NousProtocolMessage, Intent, IntentConfidence};
    pub use nous_protocol::builders::NousProtocolBuilder;

    // Runtime
    pub use nous_runtime::store::InMemoryStore;
    pub use nous_runtime::executor::NousExecutor;
}
