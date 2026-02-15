use crate::embeddings::math::cosine_similarity;
use crate::logical_atoms::comparison::{compare_logical_atoms, AtomComparisonResult};
use crate::logical_atoms::types::LogicalAtoms;
use crate::types::Embedding;
use crate::error::NousResult;

/// Result of hybrid validation combining embeddings + logical atoms
#[derive(Debug, Clone)]
pub struct HybridValidationResult {
    /// Are the two messages semantically compatible?
    pub compatible: bool,
    /// Embedding cosine similarity
    pub embedding_similarity: f64,
    /// Logical atoms comparison result
    pub atoms_result: Option<AtomComparisonResult>,
    /// Combined score (0-1)
    pub combined_score: f64,
    /// Failure analysis
    pub failure_reasons: Vec<String>,
}

/// Validate semantic compatibility using both embeddings and logical atoms.
///
/// This hybrid approach catches cases pure embeddings miss:
/// - "all users" vs "some users" (high embedding similarity, logical conflict)
/// - "should do X" vs "should NOT do X" (high embedding similarity, negation conflict)
pub fn validate_hybrid(
    embedding_a: &Embedding,
    embedding_b: &Embedding,
    atoms_a: Option<&LogicalAtoms>,
    atoms_b: Option<&LogicalAtoms>,
    similarity_threshold: f64,
) -> NousResult<HybridValidationResult> {
    let embedding_sim = cosine_similarity(embedding_a, embedding_b)?;
    let mut failure_reasons = Vec::new();

    // Check embedding similarity
    let embedding_ok = embedding_sim >= similarity_threshold;
    if !embedding_ok {
        failure_reasons.push(format!(
            "Embedding similarity {:.3} below threshold {:.3}",
            embedding_sim, similarity_threshold
        ));
    }

    // Check logical atoms if available
    let atoms_result = match (atoms_a, atoms_b) {
        (Some(a), Some(b)) => {
            let result = compare_logical_atoms(a, b);
            if !result.compatible {
                for conflict in &result.conflicts {
                    failure_reasons.push(format!("Logical conflict: {conflict}"));
                }
            }
            Some(result)
        }
        _ => None,
    };

    let atoms_ok = atoms_result
        .as_ref()
        .map(|r| r.compatible)
        .unwrap_or(true);

    // Combined score: embedding similarity weighted by atoms compatibility
    let combined_score = if atoms_ok {
        embedding_sim
    } else {
        // Penalize embedding similarity if logical atoms conflict
        embedding_sim * 0.5
    };

    let compatible = embedding_ok && atoms_ok;

    Ok(HybridValidationResult {
        compatible,
        embedding_similarity: embedding_sim,
        atoms_result,
        combined_score,
        failure_reasons,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logical_atoms::extractor::extract_logical_atoms_sync;

    fn emb(values: &[f64]) -> Embedding {
        Embedding::new(values.to_vec()).unwrap()
    }

    #[test]
    fn test_hybrid_compatible() {
        let a = emb(&[1.0, 0.0, 0.0]);
        let b = emb(&[0.95, 0.05, 0.0]);
        let atoms_a = extract_logical_atoms_sync("Process all records");
        let atoms_b = extract_logical_atoms_sync("Handle all records");

        let result = validate_hybrid(&a, &b, Some(&atoms_a), Some(&atoms_b), 0.9).unwrap();
        assert!(result.compatible);
    }

    #[test]
    fn test_hybrid_logical_conflict() {
        // High embedding similarity but logical conflict
        let a = emb(&[1.0, 0.0, 0.0]);
        let b = emb(&[0.99, 0.01, 0.0]);
        let atoms_a = extract_logical_atoms_sync("Delete all users");
        let atoms_b = extract_logical_atoms_sync("Delete some users");

        let result = validate_hybrid(&a, &b, Some(&atoms_a), Some(&atoms_b), 0.9).unwrap();
        assert!(!result.compatible);
        assert!(!result.failure_reasons.is_empty());
    }

    #[test]
    fn test_hybrid_no_atoms() {
        let a = emb(&[1.0, 0.0, 0.0]);
        let b = emb(&[0.95, 0.05, 0.0]);
        let result = validate_hybrid(&a, &b, None, None, 0.9).unwrap();
        assert!(result.compatible);
    }
}
