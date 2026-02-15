use nous_core::embeddings::math::cosine_similarity;
use nous_core::types::Embedding;

use crate::types::{NousProtocolMessage, PostCondition, PostconditionResult};

/// Default similarity threshold for postcondition embedding matching.
const POSTCONDITION_SIMILARITY_THRESHOLD: f64 = 0.7;

/// Verify a postcondition against execution result.
pub fn verify_postcondition(
    condition: &PostCondition,
    result_embedding: &Embedding,
    actual_confidence: f64,
) -> PostconditionResult {
    let similarity = cosine_similarity(&condition.embedding, result_embedding)
        .unwrap_or(0.0);

    let meets_confidence = actual_confidence >= condition.guaranteed_confidence.value();
    let meets_embedding = similarity >= POSTCONDITION_SIMILARITY_THRESHOLD;
    let verified = meets_confidence && meets_embedding;

    let reason = if verified {
        format!(
            "Verified (confidence: {actual_confidence:.3}, similarity: {similarity:.3})"
        )
    } else {
        let mut parts = Vec::new();
        if !meets_confidence {
            parts.push(format!(
                "confidence {actual_confidence:.3} < {}",
                condition.guaranteed_confidence.value()
            ));
        }
        if !meets_embedding {
            parts.push(format!(
                "similarity {similarity:.3} < {POSTCONDITION_SIMILARITY_THRESHOLD}"
            ));
        }
        format!("Failed: {}", parts.join(", "))
    };

    PostconditionResult {
        condition_id: condition.id.clone(),
        verified,
        actual_confidence: Some(actual_confidence),
        reason: Some(reason),
    }
}

/// Verify all postconditions for a message.
/// Returns whether all conditions are verified, along with individual results.
pub fn verify_postconditions(
    message: &NousProtocolMessage,
    result_embedding: &Embedding,
    actual_confidence: f64,
) -> (bool, Vec<PostconditionResult>) {
    let mut results = Vec::new();

    for condition in &message.contracts.ensures {
        let result = verify_postcondition(condition, result_embedding, actual_confidence);
        results.push(result);
    }

    let all_verified = results.iter().all(|r| r.verified);
    (all_verified, results)
}

#[cfg(test)]
mod tests {
    use nous_core::types::Confidence;

    use super::*;

    fn emb(values: &[f64]) -> Embedding {
        Embedding::new(values.to_vec()).unwrap()
    }

    #[test]
    fn test_postcondition_verified() {
        let condition = PostCondition {
            id: "result_valid".into(),
            description: "Result is valid".into(),
            embedding: emb(&[1.0, 0.0, 0.0]),
            guaranteed_confidence: Confidence::new(0.8),
        };

        let result_emb = emb(&[0.95, 0.05, 0.0]);
        let result = verify_postcondition(&condition, &result_emb, 0.9);

        assert!(result.verified);
    }

    #[test]
    fn test_postcondition_failed_confidence() {
        let condition = PostCondition {
            id: "result_valid".into(),
            description: "Result is valid".into(),
            embedding: emb(&[1.0, 0.0, 0.0]),
            guaranteed_confidence: Confidence::new(0.9),
        };

        let result_emb = emb(&[0.95, 0.05, 0.0]);
        let result = verify_postcondition(&condition, &result_emb, 0.5);

        assert!(!result.verified);
    }

    #[test]
    fn test_postcondition_failed_similarity() {
        let condition = PostCondition {
            id: "result_valid".into(),
            description: "Result is valid".into(),
            embedding: emb(&[1.0, 0.0, 0.0]),
            guaranteed_confidence: Confidence::new(0.8),
        };

        // Orthogonal embedding = 0 similarity
        let result_emb = emb(&[0.0, 1.0, 0.0]);
        let result = verify_postcondition(&condition, &result_emb, 0.95);

        assert!(!result.verified);
    }
}
