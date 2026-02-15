use nous_core::embeddings::math::cosine_similarity;
use nous_core::types::{Embedding, Timestamp};

use crate::error::RuntimeResult;

use super::types::{
    ConceptEdge, ConceptGraph, ConceptGraphMetadata, ConceptGraphOptions, ConceptMatch,
    ConceptNode,
};

/// Internal cluster representation during agglomerative clustering.
#[derive(Debug, Clone)]
struct Cluster {
    /// Raw centroid values (not yet wrapped in Embedding).
    embedding: Vec<f64>,
    /// Message IDs belonging to this cluster.
    message_ids: Vec<String>,
}

/// Builder for emergent concept graphs.
///
/// Discovers concepts automatically by clustering message embeddings
/// using a simple but effective agglomerative approach.
///
/// # Example
///
/// ```ignore
/// let mut builder = ConceptGraphBuilder::new(ConceptGraphOptions::default());
///
/// builder.add_message("msg-1", embedding_1);
/// builder.add_message("msg-2", embedding_2);
///
/// let graph = builder.build()?;
/// let matches = builder.find_concept(&query_embedding, &graph, 0.6);
/// ```
pub struct ConceptGraphBuilder {
    options: ConceptGraphOptions,
    /// Pending messages to be clustered on the next `build()`.
    pending: Vec<(String, Embedding)>,
}

impl ConceptGraphBuilder {
    /// Create a new builder with the given options.
    pub fn new(options: ConceptGraphOptions) -> Self {
        Self {
            options,
            pending: Vec::new(),
        }
    }

    /// Create a builder with default options.
    pub fn with_defaults() -> Self {
        Self::new(ConceptGraphOptions::default())
    }

    /// Add a message (by ID and embedding) to the pending set.
    pub fn add_message(&mut self, id: impl Into<String>, embedding: Embedding) {
        self.pending.push((id.into(), embedding));
    }

    /// Number of pending messages.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Build a concept graph from all pending messages.
    ///
    /// The algorithm:
    /// 1. Start with each message as its own cluster
    /// 2. Iteratively merge the most similar pair above threshold
    /// 3. Filter clusters below `min_cluster_size`
    /// 4. Generate concept nodes with labels and keywords
    /// 5. Create edges between concepts above `min_edge_similarity`
    pub fn build(&self) -> RuntimeResult<ConceptGraph> {
        if self.pending.is_empty() {
            return Ok(self.empty_graph());
        }

        // Step 1: Initialize clusters
        let clusters: Vec<Cluster> = self
            .pending
            .iter()
            .map(|(id, emb)| Cluster {
                embedding: emb.as_slice().to_vec(),
                message_ids: vec![id.clone()],
            })
            .collect();

        // Step 2: Agglomerative clustering
        let merged = self.agglomerative_clustering(clusters);

        // Step 3: Filter by minimum cluster size
        let valid: Vec<Cluster> = merged
            .into_iter()
            .filter(|c| c.message_ids.len() >= self.options.min_cluster_size)
            .collect();

        // Step 4: Generate concept nodes
        let nodes = self.generate_nodes(&valid);

        // Step 5: Create edges
        let edges = self.create_edges(&nodes);

        Ok(ConceptGraph {
            metadata: ConceptGraphMetadata {
                message_count: self.pending.len(),
                cluster_threshold: self.options.cluster_threshold,
                last_updated: Timestamp::now(),
                version: 1,
            },
            nodes,
            edges,
        })
    }

    /// Find a concept in the graph by embedding similarity.
    ///
    /// Returns the best matching concept above the threshold.
    pub fn find_concept(
        &self,
        query: &Embedding,
        graph: &ConceptGraph,
        threshold: f64,
    ) -> Option<ConceptMatch> {
        let matches = self.find_all_concepts(query, graph, threshold);
        matches.into_iter().next()
    }

    /// Find all concepts in the graph matching a query above the threshold.
    ///
    /// Results are sorted by descending similarity.
    pub fn find_all_concepts(
        &self,
        query: &Embedding,
        graph: &ConceptGraph,
        threshold: f64,
    ) -> Vec<ConceptMatch> {
        let mut matches: Vec<ConceptMatch> = graph
            .nodes
            .iter()
            .filter_map(|node| {
                cosine_similarity(query, &node.embedding)
                    .ok()
                    .and_then(|sim| {
                        if sim >= threshold {
                            Some(ConceptMatch {
                                concept: node.clone(),
                                similarity: sim,
                            })
                        } else {
                            None
                        }
                    })
            })
            .collect();

        matches.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        matches
    }

    /// Agglomerative clustering: repeatedly merge the most similar pair
    /// until no pair exceeds the threshold.
    fn agglomerative_clustering(&self, initial: Vec<Cluster>) -> Vec<Cluster> {
        let mut clusters = initial;
        let threshold = self.options.cluster_threshold;

        loop {
            if clusters.len() <= 1 {
                break;
            }

            // Find the most similar pair
            let mut best_i = 0;
            let mut best_j = 1;
            let mut best_sim = f64::NEG_INFINITY;

            for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    let sim = raw_cosine_similarity(
                        &clusters[i].embedding,
                        &clusters[j].embedding,
                    );
                    if sim > best_sim {
                        best_sim = sim;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            // Stop if best pair is below threshold
            if best_sim < threshold {
                break;
            }

            // Merge: weighted centroid by cluster size
            let size_a = clusters[best_i].message_ids.len() as f64;
            let size_b = clusters[best_j].message_ids.len() as f64;
            let total = size_a + size_b;

            let mut merged_emb: Vec<f64> = clusters[best_i]
                .embedding
                .iter()
                .zip(clusters[best_j].embedding.iter())
                .map(|(a, b)| (a * size_a + b * size_b) / total)
                .collect();

            // Normalize the merged centroid
            let magnitude: f64 = merged_emb.iter().map(|v| v * v).sum::<f64>().sqrt();
            if magnitude > 0.0 {
                for v in &mut merged_emb {
                    *v /= magnitude;
                }
            }

            let mut merged_ids = clusters[best_i].message_ids.clone();
            merged_ids.extend(clusters[best_j].message_ids.clone());

            // Remove the two old clusters (higher index first to preserve indices)
            let (lo, hi) = if best_i < best_j {
                (best_i, best_j)
            } else {
                (best_j, best_i)
            };
            clusters.remove(hi);
            clusters.remove(lo);

            clusters.push(Cluster {
                embedding: merged_emb,
                message_ids: merged_ids,
            });
        }

        clusters
    }

    /// Generate ConceptNode values from clusters.
    fn generate_nodes(&self, clusters: &[Cluster]) -> Vec<ConceptNode> {
        // Build a map from message ID to action for labeling
        let action_map: std::collections::HashMap<&str, &Embedding> = self
            .pending
            .iter()
            .map(|(id, emb)| (id.as_str(), emb))
            .collect();

        clusters
            .iter()
            .enumerate()
            .map(|(idx, cluster)| {
                let embedding = Embedding::new_unchecked(cluster.embedding.clone());

                // Calculate strength: average cosine similarity to centroid
                let strength = if cluster.message_ids.len() <= 1 {
                    1.0
                } else {
                    let mut total_sim = 0.0;
                    let mut count = 0;
                    for msg_id in &cluster.message_ids {
                        if let Some(msg_emb) = action_map.get(msg_id.as_str()) {
                            if let Ok(sim) = cosine_similarity(&embedding, msg_emb) {
                                total_sim += sim;
                                count += 1;
                            }
                        }
                    }
                    if count > 0 {
                        total_sim / count as f64
                    } else {
                        1.0
                    }
                };

                let label = format!("concept_{}", idx);

                ConceptNode {
                    id: format!("concept_{}", uuid::Uuid::new_v4()),
                    label,
                    embedding,
                    message_ids: cluster.message_ids.clone(),
                    strength,
                    keywords: Vec::new(), // Keywords would require text access
                    last_updated: Timestamp::now(),
                }
            })
            .collect()
    }

    /// Create edges between concept nodes that exceed the minimum edge similarity.
    fn create_edges(&self, nodes: &[ConceptNode]) -> Vec<ConceptEdge> {
        let mut edges = Vec::new();
        let min_sim = self.options.min_edge_similarity;

        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                if let Ok(similarity) =
                    cosine_similarity(&nodes[i].embedding, &nodes[j].embedding)
                {
                    if similarity >= min_sim {
                        edges.push(ConceptEdge {
                            source: nodes[i].id.clone(),
                            target: nodes[j].id.clone(),
                            similarity,
                            relation_type: None,
                        });
                    }
                }
            }
        }

        edges
    }

    /// Create an empty graph.
    fn empty_graph(&self) -> ConceptGraph {
        ConceptGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
            metadata: ConceptGraphMetadata {
                message_count: 0,
                cluster_threshold: self.options.cluster_threshold,
                last_updated: Timestamp::now(),
                version: 1,
            },
        }
    }
}

impl std::fmt::Debug for ConceptGraphBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConceptGraphBuilder")
            .field("options", &self.options)
            .field("pending_messages", &self.pending.len())
            .finish()
    }
}

/// Raw cosine similarity between two f64 slices (internal utility).
fn raw_cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nous_core::types::Embedding;

    fn emb(values: &[f64]) -> Embedding {
        Embedding::new(values.to_vec()).unwrap()
    }

    #[test]
    fn test_empty_build() {
        let builder = ConceptGraphBuilder::with_defaults();
        let graph = builder.build().unwrap();
        assert!(graph.nodes.is_empty());
        assert!(graph.edges.is_empty());
        assert_eq!(graph.metadata.message_count, 0);
    }

    #[test]
    fn test_single_message() {
        let mut builder = ConceptGraphBuilder::with_defaults();
        builder.add_message("msg-1", emb(&[1.0, 0.0, 0.0]));

        let graph = builder.build().unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].message_ids.len(), 1);
        assert_eq!(graph.metadata.message_count, 1);
    }

    #[test]
    fn test_similar_messages_cluster() {
        let mut builder = ConceptGraphBuilder::new(ConceptGraphOptions {
            cluster_threshold: 0.9,
            min_cluster_size: 1,
            ..Default::default()
        });

        // These are very similar and should cluster together
        builder.add_message("a", emb(&[1.0, 0.0, 0.0]));
        builder.add_message("b", emb(&[0.99, 0.01, 0.0]));
        builder.add_message("c", emb(&[0.98, 0.02, 0.0]));

        // This is different
        builder.add_message("d", emb(&[0.0, 1.0, 0.0]));

        let graph = builder.build().unwrap();

        // Should have fewer nodes than messages due to clustering
        assert!(graph.nodes.len() < 4);
        assert_eq!(graph.metadata.message_count, 4);

        // The first three should be clustered, "d" separate
        let large_cluster = graph
            .nodes
            .iter()
            .find(|n| n.message_ids.len() >= 3);
        assert!(large_cluster.is_some());
    }

    #[test]
    fn test_dissimilar_messages_no_cluster() {
        let mut builder = ConceptGraphBuilder::new(ConceptGraphOptions {
            cluster_threshold: 0.9,
            min_cluster_size: 1,
            ..Default::default()
        });

        // Orthogonal embeddings -- should not cluster
        builder.add_message("x", emb(&[1.0, 0.0, 0.0]));
        builder.add_message("y", emb(&[0.0, 1.0, 0.0]));
        builder.add_message("z", emb(&[0.0, 0.0, 1.0]));

        let graph = builder.build().unwrap();
        assert_eq!(graph.nodes.len(), 3);
    }

    #[test]
    fn test_find_concept() {
        let mut builder = ConceptGraphBuilder::new(ConceptGraphOptions {
            cluster_threshold: 0.8,
            min_cluster_size: 1,
            min_edge_similarity: 0.3,
            ..Default::default()
        });

        builder.add_message("a", emb(&[1.0, 0.0, 0.0]));
        builder.add_message("b", emb(&[0.0, 1.0, 0.0]));

        let graph = builder.build().unwrap();
        let query = emb(&[0.95, 0.05, 0.0]);

        let result = builder.find_concept(&query, &graph, 0.8);
        assert!(result.is_some());
        let m = result.unwrap();
        assert!(m.concept.message_ids.contains(&"a".to_string()));
    }

    #[test]
    fn test_edges_created() {
        let mut builder = ConceptGraphBuilder::new(ConceptGraphOptions {
            cluster_threshold: 0.95,  // High threshold so no clustering
            min_cluster_size: 1,
            min_edge_similarity: 0.3, // Low edge threshold
            ..Default::default()
        });

        // Somewhat similar pairs
        builder.add_message("a", emb(&[1.0, 0.3, 0.0]));
        builder.add_message("b", emb(&[0.8, 0.6, 0.0]));

        let graph = builder.build().unwrap();

        // Should have 2 separate nodes with an edge between them
        assert_eq!(graph.nodes.len(), 2);
        assert!(!graph.edges.is_empty());
    }

    #[test]
    fn test_find_all_concepts_sorted() {
        let mut builder = ConceptGraphBuilder::new(ConceptGraphOptions {
            cluster_threshold: 0.99, // Very high to prevent clustering
            min_cluster_size: 1,
            ..Default::default()
        });

        builder.add_message("close", emb(&[0.95, 0.05, 0.0]));
        builder.add_message("medium", emb(&[0.7, 0.3, 0.0]));
        builder.add_message("far", emb(&[0.0, 0.0, 1.0]));

        let graph = builder.build().unwrap();
        let query = emb(&[1.0, 0.0, 0.0]);

        let matches = builder.find_all_concepts(&query, &graph, 0.5);

        // Should be sorted by descending similarity
        if matches.len() >= 2 {
            assert!(matches[0].similarity >= matches[1].similarity);
        }
    }
}
