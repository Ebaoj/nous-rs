use nous_core::types::Embedding;
use nous_protocol::types::NousProtocolMessage;

/// Trait for storing and retrieving NousProtocolMessages.
///
/// Implementations provide ID-based lookup, embedding-similarity search,
/// and basic collection operations. Used by [`super::ReferenceResolver`] and
/// [`crate::executor::NousExecutor`] at runtime.
pub trait MessageStore {
    /// Insert or update a message in the store.
    fn add(&mut self, message: NousProtocolMessage);

    /// Retrieve a message by its unique ID.
    fn get(&self, id: &str) -> Option<&NousProtocolMessage>;

    /// Find messages whose embedding has cosine similarity >= `min_similarity`
    /// with the given `embedding`. Results are sorted by descending similarity.
    fn find_by_similarity(
        &self,
        embedding: &Embedding,
        min_similarity: f64,
    ) -> Vec<&NousProtocolMessage>;

    /// Remove a message by ID, returning it if it existed.
    fn remove(&mut self, id: &str) -> Option<NousProtocolMessage>;

    /// Number of messages currently stored.
    fn len(&self) -> usize;

    /// Whether the store is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
