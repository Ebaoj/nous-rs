use nous_protocol::types::{NousProtocolMessage, Relation, SemanticReference};

use super::traits::MessageStore;

/// Resolves [`SemanticReference`]s against a [`MessageStore`].
///
/// Resolution strategy:
/// 1. If the reference has an `id`, look up by ID directly.
/// 2. If the reference has an `embedding`, search by cosine similarity
///    using `min_similarity` (or a default of 0.8).
/// 3. For `input` / `dependency` relations, return the single best match.
/// 4. For `context` / `related` / `supersedes`, return all matches.
pub struct ReferenceResolver<'a, S: MessageStore> {
    store: &'a S,
}

/// Result of resolving a single reference: either one message or many.
#[derive(Debug)]
pub enum ResolvedReference<'a> {
    /// A single resolved message (for input/dependency relations).
    Single(&'a NousProtocolMessage),
    /// Multiple resolved messages (for context/related relations).
    Multiple(Vec<&'a NousProtocolMessage>),
    /// Reference could not be resolved.
    None,
}

impl<'a, S: MessageStore> ReferenceResolver<'a, S> {
    /// Create a new resolver backed by the given store.
    pub fn new(store: &'a S) -> Self {
        Self { store }
    }

    /// Resolve a single semantic reference.
    pub fn resolve_reference(&self, reference: &SemanticReference) -> ResolvedReference<'a> {
        // Try ID-based resolution first
        if let Some(ref id) = reference.id {
            if let Some(msg) = self.store.get(id) {
                return ResolvedReference::Single(msg);
            }
        }

        // Try embedding-based resolution
        if let Some(ref embedding) = reference.embedding {
            let min_similarity = reference.min_similarity.unwrap_or(0.8);
            let matches = self.store.find_by_similarity(embedding, min_similarity);

            if matches.is_empty() {
                return ResolvedReference::None;
            }

            // For input and dependency relations, return single best match
            match reference.relation {
                Relation::Input | Relation::Dependency => {
                    return ResolvedReference::Single(matches[0]);
                }
                Relation::Context | Relation::Related | Relation::Supersedes => {
                    return ResolvedReference::Multiple(matches);
                }
            }
        }

        ResolvedReference::None
    }

    /// Resolve all references in a message.
    ///
    /// Returns a vector of `(reference_index, resolved)` pairs.
    pub fn resolve_all_references(
        &self,
        message: &NousProtocolMessage,
    ) -> Vec<(usize, ResolvedReference<'a>)> {
        message
            .references
            .iter()
            .enumerate()
            .map(|(idx, r)| (idx, self.resolve_reference(r)))
            .collect()
    }

    /// Check if all dependency references are resolved.
    pub fn are_dependencies_resolved(&self, message: &NousProtocolMessage) -> bool {
        for reference in &message.references {
            if reference.relation == Relation::Dependency {
                if let ResolvedReference::None = self.resolve_reference(reference) {
                    return false;
                }
            }
        }
        true
    }

    /// Collect all resolved input messages.
    pub fn get_inputs(&self, message: &NousProtocolMessage) -> Vec<&'a NousProtocolMessage> {
        let mut inputs = Vec::new();
        for reference in &message.references {
            if reference.relation == Relation::Input {
                match self.resolve_reference(reference) {
                    ResolvedReference::Single(msg) => inputs.push(msg),
                    ResolvedReference::Multiple(msgs) => inputs.extend(msgs),
                    ResolvedReference::None => {}
                }
            }
        }
        inputs
    }

    /// Collect all resolved dependency messages.
    pub fn get_dependencies(&self, message: &NousProtocolMessage) -> Vec<&'a NousProtocolMessage> {
        let mut deps = Vec::new();
        for reference in &message.references {
            if reference.relation == Relation::Dependency {
                match self.resolve_reference(reference) {
                    ResolvedReference::Single(msg) => deps.push(msg),
                    ResolvedReference::Multiple(msgs) => deps.extend(msgs),
                    ResolvedReference::None => {}
                }
            }
        }
        deps
    }

    /// Collect all resolved context messages.
    pub fn get_context(&self, message: &NousProtocolMessage) -> Vec<&'a NousProtocolMessage> {
        let mut context = Vec::new();
        for reference in &message.references {
            if reference.relation == Relation::Context {
                match self.resolve_reference(reference) {
                    ResolvedReference::Single(msg) => context.push(msg),
                    ResolvedReference::Multiple(msgs) => context.extend(msgs),
                    ResolvedReference::None => {}
                }
            }
        }
        context
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::InMemoryStore;
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
    fn test_resolve_by_id() {
        let mut store = InMemoryStore::new();
        store.add(make_message("target", vec![1.0, 0.0, 0.0]));

        let resolver = ReferenceResolver::new(&store);
        let reference = SemanticReference {
            ref_type: ReferenceType::Message,
            id: Some("target".into()),
            embedding: None,
            min_similarity: None,
            relation: Relation::Input,
        };

        match resolver.resolve_reference(&reference) {
            ResolvedReference::Single(msg) => {
                assert_eq!(msg.id.as_str(), "target");
            }
            _ => panic!("expected single result"),
        }
    }

    #[test]
    fn test_resolve_by_embedding() {
        let mut store = InMemoryStore::new();
        store.add(make_message("similar", vec![0.95, 0.05, 0.0]));
        store.add(make_message("different", vec![0.0, 0.0, 1.0]));

        let resolver = ReferenceResolver::new(&store);
        let reference = SemanticReference {
            ref_type: ReferenceType::Message,
            id: None,
            embedding: Some(Embedding::new(vec![1.0, 0.0, 0.0]).unwrap()),
            min_similarity: Some(0.8),
            relation: Relation::Input,
        };

        match resolver.resolve_reference(&reference) {
            ResolvedReference::Single(msg) => {
                assert_eq!(msg.id.as_str(), "similar");
            }
            _ => panic!("expected single result for input relation"),
        }
    }

    #[test]
    fn test_resolve_context_returns_multiple() {
        let mut store = InMemoryStore::new();
        store.add(make_message("a", vec![0.9, 0.1, 0.0]));
        store.add(make_message("b", vec![0.85, 0.15, 0.0]));
        store.add(make_message("c", vec![0.0, 0.0, 1.0]));

        let resolver = ReferenceResolver::new(&store);
        let reference = SemanticReference {
            ref_type: ReferenceType::Context,
            id: None,
            embedding: Some(Embedding::new(vec![1.0, 0.0, 0.0]).unwrap()),
            min_similarity: Some(0.7),
            relation: Relation::Context,
        };

        match resolver.resolve_reference(&reference) {
            ResolvedReference::Multiple(msgs) => {
                assert!(msgs.len() >= 2);
            }
            _ => panic!("expected multiple results for context relation"),
        }
    }

    #[test]
    fn test_unresolved_reference() {
        let store = InMemoryStore::new();
        let resolver = ReferenceResolver::new(&store);
        let reference = SemanticReference {
            ref_type: ReferenceType::Message,
            id: Some("nonexistent".into()),
            embedding: None,
            min_similarity: None,
            relation: Relation::Input,
        };

        assert!(matches!(
            resolver.resolve_reference(&reference),
            ResolvedReference::None
        ));
    }

    #[test]
    fn test_are_dependencies_resolved() {
        let mut store = InMemoryStore::new();
        store.add(make_message("dep-1", vec![1.0, 0.0]));

        let resolver = ReferenceResolver::new(&store);

        let mut msg = make_message("main", vec![0.5, 0.5]);
        msg.references.push(SemanticReference {
            ref_type: ReferenceType::Message,
            id: Some("dep-1".into()),
            embedding: None,
            min_similarity: None,
            relation: Relation::Dependency,
        });

        assert!(resolver.are_dependencies_resolved(&msg));

        // Add an unresolvable dependency
        msg.references.push(SemanticReference {
            ref_type: ReferenceType::Message,
            id: Some("missing".into()),
            embedding: None,
            min_similarity: None,
            relation: Relation::Dependency,
        });

        assert!(!resolver.are_dependencies_resolved(&msg));
    }
}
