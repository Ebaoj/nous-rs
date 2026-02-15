use std::collections::HashMap;

use nous_core::embeddings::math::cosine_similarity;
use nous_core::types::Embedding;

use crate::types::{NousProtocolMessage, PreCondition, PreconditionResult};

/// Check if a precondition is satisfied by the given state embeddings.
pub fn check_precondition(
    condition: &PreCondition,
    state_embeddings: &HashMap<String, Embedding>,
) -> PreconditionResult {
    let threshold = condition.threshold.unwrap_or(0.8);
    let mut max_similarity = 0.0_f64;
    let mut matched = false;

    for state_embedding in state_embeddings.values() {
        match cosine_similarity(&condition.embedding, state_embedding) {
            Ok(similarity) => {
                if similarity > max_similarity {
                    max_similarity = similarity;
                }
                if similarity >= threshold {
                    matched = true;
                    break;
                }
            }
            Err(_) => continue,
        }
    }

    PreconditionResult {
        condition_id: condition.id.clone(),
        satisfied: matched,
        similarity: Some(max_similarity),
        reason: Some(if matched {
            format!("Matched with similarity {max_similarity:.3}")
        } else {
            format!(
                "No state matched threshold {threshold} (max similarity: {max_similarity:.3})"
            )
        }),
    }
}

/// Check all preconditions for a message.
/// Returns whether all required conditions are satisfied, along with individual results.
pub fn check_preconditions(
    message: &NousProtocolMessage,
    state_embeddings: &HashMap<String, Embedding>,
) -> (bool, Vec<PreconditionResult>) {
    let mut results = Vec::new();

    for condition in &message.contracts.requires {
        let result = check_precondition(condition, state_embeddings);
        results.push(result);
    }

    // All required conditions must be satisfied
    let all_satisfied = results.iter().zip(&message.contracts.requires).all(
        |(result, condition)| result.satisfied || !condition.required,
    );

    (all_satisfied, results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emb(values: &[f64]) -> Embedding {
        Embedding::new(values.to_vec()).unwrap()
    }

    #[test]
    fn test_precondition_satisfied() {
        let condition = PreCondition {
            id: "db_available".into(),
            description: "Database is available".into(),
            embedding: emb(&[1.0, 0.0, 0.0]),
            threshold: Some(0.8),
            required: true,
        };

        let mut state = HashMap::new();
        state.insert("db_status".into(), emb(&[0.95, 0.05, 0.0]));

        let result = check_precondition(&condition, &state);
        assert!(result.satisfied);
    }

    #[test]
    fn test_precondition_not_satisfied() {
        let condition = PreCondition {
            id: "db_available".into(),
            description: "Database is available".into(),
            embedding: emb(&[1.0, 0.0, 0.0]),
            threshold: Some(0.8),
            required: true,
        };

        let mut state = HashMap::new();
        state.insert("unrelated".into(), emb(&[0.0, 1.0, 0.0]));

        let result = check_precondition(&condition, &state);
        assert!(!result.satisfied);
    }

    #[test]
    fn test_empty_state_fails() {
        let condition = PreCondition {
            id: "test".into(),
            description: "Test".into(),
            embedding: emb(&[1.0, 0.0, 0.0]),
            threshold: Some(0.8),
            required: true,
        };

        let state = HashMap::new();
        let result = check_precondition(&condition, &state);
        assert!(!result.satisfied);
    }
}
