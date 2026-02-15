use std::collections::{HashMap, HashSet};

use nous_core::embeddings::math::{cosine_similarity, merge_embeddings, normalize};
use nous_core::types::{Embedding, EmbeddingModel, Timestamp};
use nous_protocol::types::NousProtocolMessage;

use crate::error::{RuntimeError, RuntimeResult};

use super::types::{
    CompressionOptions, ConversationMemory, RelevanceOptions, RelevanceResult,
};

/// Manages compressed conversation memories.
///
/// Compresses entire conversations into single embeddings that can
/// efficiently answer "is this query relevant to this conversation?"
///
/// # Example
///
/// ```ignore
/// let manager = ConversationMemoryManager::new(EmbeddingModel::default());
/// let memory = manager.compress(&messages, &CompressionOptions::default())?;
///
/// let result = manager.find_relevant(&query_embedding, &[memory], &RelevanceOptions::default());
/// ```
pub struct ConversationMemoryManager {
    embedding_model: EmbeddingModel,
    memories: HashMap<String, ConversationMemory>,
}

impl ConversationMemoryManager {
    /// Create a new memory manager.
    pub fn new(embedding_model: EmbeddingModel) -> Self {
        Self {
            embedding_model,
            memories: HashMap::new(),
        }
    }

    /// Add (or replace) a memory in the manager.
    pub fn add_memory(&mut self, memory: ConversationMemory) {
        self.memories.insert(memory.id.clone(), memory);
    }

    /// Get a memory by ID.
    pub fn get_memory(&self, id: &str) -> Option<&ConversationMemory> {
        self.memories.get(id)
    }

    /// Get all stored memories.
    pub fn get_all(&self) -> Vec<&ConversationMemory> {
        self.memories.values().collect()
    }

    /// Remove a memory by ID.
    pub fn remove_memory(&mut self, id: &str) -> Option<ConversationMemory> {
        self.memories.remove(id)
    }

    /// Number of stored memories.
    pub fn len(&self) -> usize {
        self.memories.len()
    }

    /// Whether the memory store is empty.
    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }

    /// Compress a conversation (list of messages) into a single memory.
    ///
    /// Strategy:
    /// 1. If the conversation is smaller than `min_messages`, just average
    ///    the message embeddings.
    /// 2. Otherwise, average embeddings and extract topics/entities.
    pub fn compress(
        &self,
        messages: &[NousProtocolMessage],
        options: &CompressionOptions,
    ) -> RuntimeResult<ConversationMemory> {
        if messages.is_empty() {
            return Err(RuntimeError::Memory(
                "cannot compress empty message list".into(),
            ));
        }

        // Collect embeddings for averaging
        let embeddings: Vec<Embedding> = messages
            .iter()
            .map(|m| m.embedding.clone())
            .collect();

        let merged = merge_embeddings(&embeddings, None)
            .map_err(|e| RuntimeError::Memory(format!("failed to merge embeddings: {e}")))?;

        let normalized = normalize(&merged);

        // Extract topics from intent actions
        let topic_summary = if options.extract_topics {
            let intents: HashSet<String> = messages
                .iter()
                .map(|m| m.intent.action.clone())
                .collect();
            intents.into_iter().collect()
        } else {
            Vec::new()
        };

        // Extract entities from params
        let entities = if options.extract_entities {
            self.extract_entities(messages)
        } else {
            Vec::new()
        };

        // Calculate time range
        let timestamps: Vec<u64> = messages
            .iter()
            .map(|m| m.meta.timestamp.as_millis())
            .collect();
        let min_ts = *timestamps.iter().min().unwrap();
        let max_ts = *timestamps.iter().max().unwrap();

        // Derive conversation ID from first message or generate one
        let conversation_id = messages
            .first()
            .and_then(|m| m.meta.conversation_id.clone())
            .unwrap_or_else(|| format!("conv_{}", uuid::Uuid::new_v4()));

        Ok(ConversationMemory {
            id: format!("mem_{}", uuid::Uuid::new_v4()),
            conversation_id,
            embedding: normalized,
            embedding_model: self.embedding_model.clone(),
            message_count: messages.len(),
            time_range: (Timestamp::new(min_ts), Timestamp::new(max_ts)),
            topic_summary,
            entities,
            created_at: Timestamp::now(),
            metadata: options.metadata.clone(),
        })
    }

    /// Find memories relevant to a query embedding.
    ///
    /// Returns memories sorted by descending similarity.
    pub fn find_relevant(
        &self,
        query_embedding: &Embedding,
        memories: &[&ConversationMemory],
        options: &RelevanceOptions,
    ) -> Vec<RelevanceResult> {
        let mut results: Vec<RelevanceResult> = memories
            .iter()
            .filter(|m| {
                // Apply time range filter if specified
                if let Some((start, end)) = &options.time_range {
                    let (mem_start, mem_end) = &m.time_range;
                    if mem_end.as_millis() < start.as_millis() {
                        return false;
                    }
                    if mem_start.as_millis() > end.as_millis() {
                        return false;
                    }
                }
                true
            })
            .filter_map(|m| {
                cosine_similarity(query_embedding, &m.embedding)
                    .ok()
                    .map(|sim| RelevanceResult {
                        relevant: sim >= options.threshold,
                        similarity: sim,
                        memory: (*m).clone(),
                    })
            })
            .filter(|r| r.relevant)
            .collect();

        // Sort by descending similarity
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(options.limit);
        results
    }

    /// Check if a query embedding is relevant to a specific memory.
    pub fn is_relevant(
        &self,
        query_embedding: &Embedding,
        memory: &ConversationMemory,
        threshold: f64,
    ) -> RuntimeResult<RelevanceResult> {
        let similarity = cosine_similarity(query_embedding, &memory.embedding)
            .map_err(|e| RuntimeError::Memory(format!("similarity computation failed: {e}")))?;

        Ok(RelevanceResult {
            relevant: similarity >= threshold,
            similarity,
            memory: memory.clone(),
        })
    }

    /// Extract named entities from message parameters.
    ///
    /// Looks for string-typed params that look like identifiers
    /// (short values, not common text field names).
    fn extract_entities(&self, messages: &[NousProtocolMessage]) -> Vec<String> {
        let skip_names: HashSet<&str> =
            ["text", "query", "message", "content"].into_iter().collect();

        let mut entities = HashSet::new();

        for msg in messages {
            for param in &msg.params {
                if let nous_protocol::types::ParamValue::String(ref val) = param.value {
                    if val.len() < 100
                        && !skip_names.contains(param.name.to_lowercase().as_str())
                    {
                        entities.insert(format!("{}:{}", param.name, val));
                    }
                }
            }
        }

        let mut result: Vec<String> = entities.into_iter().collect();
        result.truncate(20); // Limit to 20 entities
        result
    }
}

impl std::fmt::Debug for ConversationMemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConversationMemoryManager")
            .field("embedding_model", &self.embedding_model)
            .field("memory_count", &self.memories.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nous_core::types::*;
    use nous_protocol::types::*;

    fn make_message(
        action: &str,
        embedding_vals: Vec<f64>,
        ts: u64,
    ) -> NousProtocolMessage {
        NousProtocolMessage {
            id: MessageId::random(),
            embedding: Embedding::new(embedding_vals).unwrap(),
            embedding_model: EmbeddingModel::default(),
            intent: Intent {
                action: action.into(),
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
                timestamp: Timestamp::new(ts),
                conversation_id: Some("conv-1".into()),
                sequence: None,
                protocol_version: ProtocolVersion::default(),
                custom: None,
            },
            text: None,
            logical_atoms: None,
        }
    }

    #[test]
    fn test_compress_conversation() {
        let manager = ConversationMemoryManager::new(EmbeddingModel::default());
        let messages = vec![
            make_message("query", vec![1.0, 0.0, 0.0], 1000),
            make_message("respond", vec![0.8, 0.2, 0.0], 2000),
            make_message("query", vec![0.9, 0.1, 0.0], 3000),
        ];

        let memory = manager
            .compress(&messages, &CompressionOptions::default())
            .unwrap();

        assert_eq!(memory.message_count, 3);
        assert_eq!(memory.conversation_id, "conv-1");
        assert_eq!(memory.time_range.0.as_millis(), 1000);
        assert_eq!(memory.time_range.1.as_millis(), 3000);
        assert!(!memory.topic_summary.is_empty());
    }

    #[test]
    fn test_compress_empty_fails() {
        let manager = ConversationMemoryManager::new(EmbeddingModel::default());
        let result = manager.compress(&[], &CompressionOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_find_relevant() {
        let manager = ConversationMemoryManager::new(EmbeddingModel::default());

        let memory1 = ConversationMemory {
            id: "mem-1".into(),
            conversation_id: "conv-1".into(),
            embedding: Embedding::new(vec![1.0, 0.0, 0.0]).unwrap(),
            embedding_model: EmbeddingModel::default(),
            message_count: 5,
            time_range: (Timestamp::new(1000), Timestamp::new(5000)),
            topic_summary: vec!["test".into()],
            entities: Vec::new(),
            created_at: Timestamp::now(),
            metadata: HashMap::new(),
        };

        let memory2 = ConversationMemory {
            id: "mem-2".into(),
            conversation_id: "conv-2".into(),
            embedding: Embedding::new(vec![0.0, 0.0, 1.0]).unwrap(),
            embedding_model: EmbeddingModel::default(),
            message_count: 3,
            time_range: (Timestamp::new(2000), Timestamp::new(4000)),
            topic_summary: vec!["other".into()],
            entities: Vec::new(),
            created_at: Timestamp::now(),
            metadata: HashMap::new(),
        };

        let memories = vec![&memory1, &memory2];
        let query = Embedding::new(vec![0.95, 0.05, 0.0]).unwrap();

        let results = manager.find_relevant(&query, &memories, &RelevanceOptions {
            threshold: 0.7,
            limit: 10,
            time_range: None,
        });

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].memory.id, "mem-1");
        assert!(results[0].relevant);
    }

    #[test]
    fn test_is_relevant() {
        let manager = ConversationMemoryManager::new(EmbeddingModel::default());

        let memory = ConversationMemory {
            id: "mem-1".into(),
            conversation_id: "conv-1".into(),
            embedding: Embedding::new(vec![1.0, 0.0, 0.0]).unwrap(),
            embedding_model: EmbeddingModel::default(),
            message_count: 5,
            time_range: (Timestamp::new(1000), Timestamp::new(5000)),
            topic_summary: Vec::new(),
            entities: Vec::new(),
            created_at: Timestamp::now(),
            metadata: HashMap::new(),
        };

        let relevant_query = Embedding::new(vec![0.95, 0.05, 0.0]).unwrap();
        let result = manager.is_relevant(&relevant_query, &memory, 0.7).unwrap();
        assert!(result.relevant);

        let irrelevant_query = Embedding::new(vec![0.0, 1.0, 0.0]).unwrap();
        let result = manager.is_relevant(&irrelevant_query, &memory, 0.7).unwrap();
        assert!(!result.relevant);
    }

    #[test]
    fn test_add_and_get_memory() {
        let mut manager = ConversationMemoryManager::new(EmbeddingModel::default());

        let memory = ConversationMemory {
            id: "mem-1".into(),
            conversation_id: "conv-1".into(),
            embedding: Embedding::new(vec![1.0, 0.0]).unwrap(),
            embedding_model: EmbeddingModel::default(),
            message_count: 3,
            time_range: (Timestamp::new(1000), Timestamp::new(3000)),
            topic_summary: Vec::new(),
            entities: Vec::new(),
            created_at: Timestamp::now(),
            metadata: HashMap::new(),
        };

        manager.add_memory(memory);
        assert_eq!(manager.len(), 1);
        assert!(manager.get_memory("mem-1").is_some());

        manager.remove_memory("mem-1");
        assert!(manager.is_empty());
    }
}
