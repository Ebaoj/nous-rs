use std::collections::HashMap;

use nous_core::embeddings::math::cosine_similarity;
use nous_core::types::Embedding;
use nous_protocol::types::NousProtocolMessage;

use super::traits::MessageStore;

/// In-memory message store backed by a `HashMap`.
///
/// Provides O(1) ID-based lookup and O(n) embedding-similarity search
/// using cosine similarity from `nous_core`.
#[derive(Debug, Default)]
pub struct InMemoryStore {
    messages: HashMap<String, NousProtocolMessage>,
}

impl InMemoryStore {
    /// Create a new empty store.
    pub fn new() -> Self {
        Self {
            messages: HashMap::new(),
        }
    }

    /// Return an iterator over all stored messages.
    pub fn iter(&self) -> impl Iterator<Item = &NousProtocolMessage> {
        self.messages.values()
    }

    /// Return all messages as a vector.
    pub fn all(&self) -> Vec<&NousProtocolMessage> {
        self.messages.values().collect()
    }

    /// Remove all messages from the store.
    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

impl MessageStore for InMemoryStore {
    fn add(&mut self, message: NousProtocolMessage) {
        self.messages.insert(message.id.as_str().to_string(), message);
    }

    fn get(&self, id: &str) -> Option<&NousProtocolMessage> {
        self.messages.get(id)
    }

    fn find_by_similarity(
        &self,
        embedding: &Embedding,
        min_similarity: f64,
    ) -> Vec<&NousProtocolMessage> {
        let mut results: Vec<(&NousProtocolMessage, f64)> = self
            .messages
            .values()
            .filter_map(|msg| {
                cosine_similarity(embedding, &msg.embedding)
                    .ok()
                    .and_then(|sim| {
                        if sim >= min_similarity {
                            Some((msg, sim))
                        } else {
                            None
                        }
                    })
            })
            .collect();

        // Sort by descending similarity
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter().map(|(msg, _)| msg).collect()
    }

    fn remove(&mut self, id: &str) -> Option<NousProtocolMessage> {
        self.messages.remove(id)
    }

    fn len(&self) -> usize {
        self.messages.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nous_core::types::*;
    use nous_protocol::types::*;

    fn make_message(id: &str, embedding_vals: Vec<f64>) -> NousProtocolMessage {
        NousProtocolMessage {
            id: MessageId::new(id),
            embedding: Embedding::new(embedding_vals).unwrap(),
            embedding_model: EmbeddingModel::default(),
            intent: Intent {
                action: "test_action".into(),
                confidence: IntentConfidence::from(0.9),
                description: None,
                embedding: None,
            },
            params: Vec::new(),
            contracts: Contracts::default(),
            references: Vec::new(),
            meta: ProtocolMeta {
                sender: SenderInfo {
                    sender_type: "system".into(),
                    id: "test".into(),
                    version: None,
                },
                timestamp: Timestamp::now(),
                conversation_id: None,
                sequence: None,
                protocol_version: ProtocolVersion::default(),
                custom: None,
            },
            text: None,
            logical_atoms: None,
        }
    }

    #[test]
    fn test_add_and_get() {
        let mut store = InMemoryStore::new();
        let msg = make_message("msg-1", vec![1.0, 0.0, 0.0]);
        store.add(msg);

        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());

        let retrieved = store.get("msg-1");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id.as_str(), "msg-1");
    }

    #[test]
    fn test_remove() {
        let mut store = InMemoryStore::new();
        store.add(make_message("msg-1", vec![1.0, 0.0, 0.0]));

        let removed = store.remove("msg-1");
        assert!(removed.is_some());
        assert!(store.is_empty());

        let not_found = store.remove("msg-1");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_find_by_similarity() {
        let mut store = InMemoryStore::new();
        store.add(make_message("close", vec![0.9, 0.1, 0.0]));
        store.add(make_message("far", vec![0.0, 0.0, 1.0]));
        store.add(make_message("medium", vec![0.5, 0.5, 0.0]));

        let query = Embedding::new(vec![1.0, 0.0, 0.0]).unwrap();
        let results = store.find_by_similarity(&query, 0.7);

        // "close" should match (high similarity), "far" should not
        assert!(!results.is_empty());
        assert_eq!(results[0].id.as_str(), "close");
    }

    #[test]
    fn test_empty_store() {
        let store = InMemoryStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert!(store.get("nonexistent").is_none());
    }

    #[test]
    fn test_clear() {
        let mut store = InMemoryStore::new();
        store.add(make_message("a", vec![1.0, 0.0]));
        store.add(make_message("b", vec![0.0, 1.0]));
        assert_eq!(store.len(), 2);

        store.clear();
        assert!(store.is_empty());
    }
}
