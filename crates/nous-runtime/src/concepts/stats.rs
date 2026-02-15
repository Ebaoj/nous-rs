use std::collections::HashMap;
use std::collections::HashSet;

use super::types::{ConceptGraph, ConceptGraphStats, ConnectedConcept};

/// Compute statistics for a concept graph.
pub fn compute_stats(graph: &ConceptGraph) -> ConceptGraphStats {
    let concept_count = graph.nodes.len();
    let edge_count = graph.edges.len();

    // Average messages per concept
    let total_messages: usize = graph.nodes.iter().map(|n| n.message_ids.len()).sum();
    let avg_messages_per_concept = if concept_count > 0 {
        total_messages as f64 / concept_count as f64
    } else {
        0.0
    };

    // Average strength
    let total_strength: f64 = graph.nodes.iter().map(|n| n.strength).sum();
    let avg_strength = if concept_count > 0 {
        total_strength / concept_count as f64
    } else {
        0.0
    };

    // Count edges per node
    let mut edge_counts: HashMap<&str, usize> = HashMap::new();
    for edge in &graph.edges {
        *edge_counts.entry(edge.source.as_str()).or_insert(0) += 1;
        *edge_counts.entry(edge.target.as_str()).or_insert(0) += 1;
    }

    // Top connected concepts (by edge count)
    let mut top_connected: Vec<ConnectedConcept> = graph
        .nodes
        .iter()
        .map(|n| ConnectedConcept {
            label: n.label.clone(),
            edge_count: edge_counts.get(n.id.as_str()).copied().unwrap_or(0),
        })
        .collect();
    top_connected.sort_by(|a, b| b.edge_count.cmp(&a.edge_count));
    top_connected.truncate(5);

    // Isolated concepts (no edges)
    let connected_ids: HashSet<&str> = graph
        .edges
        .iter()
        .flat_map(|e| [e.source.as_str(), e.target.as_str()])
        .collect();

    let isolated_concepts: Vec<String> = graph
        .nodes
        .iter()
        .filter(|n| !connected_ids.contains(n.id.as_str()))
        .map(|n| n.label.clone())
        .collect();

    ConceptGraphStats {
        concept_count,
        edge_count,
        avg_messages_per_concept,
        avg_strength,
        top_connected,
        isolated_concepts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::concepts::types::{
        ConceptEdge, ConceptGraph, ConceptGraphMetadata, ConceptNode,
    };
    use nous_core::types::{Embedding, Timestamp};

    fn make_node(id: &str, label: &str, msg_count: usize, strength: f64) -> ConceptNode {
        ConceptNode {
            id: id.into(),
            label: label.into(),
            embedding: Embedding::new(vec![1.0, 0.0, 0.0]).unwrap(),
            message_ids: (0..msg_count).map(|i| format!("msg-{i}")).collect(),
            strength,
            keywords: Vec::new(),
            last_updated: Timestamp::now(),
        }
    }

    #[test]
    fn test_empty_graph_stats() {
        let graph = ConceptGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
            metadata: ConceptGraphMetadata {
                message_count: 0,
                cluster_threshold: 0.75,
                last_updated: Timestamp::now(),
                version: 1,
            },
        };

        let stats = compute_stats(&graph);
        assert_eq!(stats.concept_count, 0);
        assert_eq!(stats.edge_count, 0);
        assert_eq!(stats.avg_messages_per_concept, 0.0);
        assert_eq!(stats.avg_strength, 0.0);
    }

    #[test]
    fn test_graph_with_nodes_and_edges() {
        let graph = ConceptGraph {
            nodes: vec![
                make_node("a", "concept_a", 5, 0.9),
                make_node("b", "concept_b", 3, 0.8),
                make_node("c", "concept_c", 2, 0.7),
            ],
            edges: vec![
                ConceptEdge {
                    source: "a".into(),
                    target: "b".into(),
                    similarity: 0.6,
                    relation_type: None,
                },
                ConceptEdge {
                    source: "a".into(),
                    target: "c".into(),
                    similarity: 0.55,
                    relation_type: None,
                },
            ],
            metadata: ConceptGraphMetadata {
                message_count: 10,
                cluster_threshold: 0.75,
                last_updated: Timestamp::now(),
                version: 1,
            },
        };

        let stats = compute_stats(&graph);
        assert_eq!(stats.concept_count, 3);
        assert_eq!(stats.edge_count, 2);
        assert!((stats.avg_messages_per_concept - 10.0 / 3.0).abs() < 0.01);
        assert!((stats.avg_strength - (0.9 + 0.8 + 0.7) / 3.0).abs() < 0.01);

        // "a" has 2 edges, should be top connected
        assert_eq!(stats.top_connected[0].label, "concept_a");
        assert_eq!(stats.top_connected[0].edge_count, 2);
    }

    #[test]
    fn test_isolated_concepts() {
        let graph = ConceptGraph {
            nodes: vec![
                make_node("a", "connected", 3, 0.9),
                make_node("b", "also_connected", 2, 0.8),
                make_node("c", "alone", 1, 0.7),
            ],
            edges: vec![ConceptEdge {
                source: "a".into(),
                target: "b".into(),
                similarity: 0.6,
                relation_type: None,
            }],
            metadata: ConceptGraphMetadata {
                message_count: 6,
                cluster_threshold: 0.75,
                last_updated: Timestamp::now(),
                version: 1,
            },
        };

        let stats = compute_stats(&graph);
        assert_eq!(stats.isolated_concepts, vec!["alone".to_string()]);
    }
}
