use crate::error::{NousError, NousResult};
use crate::types::Embedding;

/// Calculate cosine similarity between two embeddings.
/// Returns a value in [-1.0, 1.0].
pub fn cosine_similarity(a: &Embedding, b: &Embedding) -> NousResult<f64> {
    if a.len() != b.len() {
        return Err(NousError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..a.len() {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return Ok(0.0);
    }

    Ok(dot_product / denom)
}

/// Calculate cosine similarity from raw slices (avoids Embedding construction).
pub fn cosine_similarity_raw(a: &[f64], b: &[f64]) -> NousResult<f64> {
    if a.len() != b.len() {
        return Err(NousError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..a.len() {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return Ok(0.0);
    }

    Ok(dot_product / denom)
}

/// Calculate Euclidean distance between two embeddings.
pub fn euclidean_distance(a: &Embedding, b: &Embedding) -> NousResult<f64> {
    if a.len() != b.len() {
        return Err(NousError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }

    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum();

    Ok(sum.sqrt())
}

/// Normalize an embedding to unit length.
pub fn normalize(embedding: &Embedding) -> Embedding {
    let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        return embedding.clone();
    }
    let values: Vec<f64> = embedding.iter().map(|x| x / norm).collect();
    Embedding::new_unchecked(values)
}

/// Merge multiple embeddings via weighted average.
pub fn merge_embeddings(
    embeddings: &[Embedding],
    weights: Option<&[f64]>,
) -> NousResult<Embedding> {
    if embeddings.is_empty() {
        return Err(NousError::EmptyInput(
            "cannot merge empty list of embeddings".into(),
        ));
    }

    let dim = embeddings[0].len();
    let effective_weights: Vec<f64> = match weights {
        Some(w) => w.to_vec(),
        None => vec![1.0 / embeddings.len() as f64; embeddings.len()],
    };

    let weight_sum: f64 = effective_weights.iter().sum();
    let normalized: Vec<f64> = effective_weights.iter().map(|w| w / weight_sum).collect();

    let mut result = vec![0.0; dim];

    for (i, emb) in embeddings.iter().enumerate() {
        if emb.len() != dim {
            return Err(NousError::DimensionMismatch {
                expected: dim,
                got: emb.len(),
            });
        }
        for j in 0..dim {
            result[j] += emb[j] * normalized[i];
        }
    }

    Embedding::new(result)
}

/// Find the most similar embedding from a list.
/// Returns (index, similarity).
pub fn find_most_similar(
    query: &Embedding,
    candidates: &[Embedding],
) -> NousResult<(usize, f64)> {
    if candidates.is_empty() {
        return Err(NousError::EmptyInput("no candidates to search".into()));
    }

    let mut best_index = 0;
    let mut best_similarity = f64::NEG_INFINITY;

    for (i, candidate) in candidates.iter().enumerate() {
        let similarity = cosine_similarity(query, candidate)?;
        if similarity > best_similarity {
            best_similarity = similarity;
            best_index = i;
        }
    }

    Ok((best_index, best_similarity))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emb(values: &[f64]) -> Embedding {
        Embedding::new(values.to_vec()).unwrap()
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = emb(&[1.0, 0.0, 0.0]);
        let b = emb(&[1.0, 0.0, 0.0]);
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = emb(&[1.0, 0.0]);
        let b = emb(&[0.0, 1.0]);
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = emb(&[1.0, 0.0]);
        let b = emb(&[-1.0, 0.0]);
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = emb(&[1.0, 0.0]);
        let b = emb(&[1.0, 0.0, 0.0]);
        assert!(cosine_similarity(&a, &b).is_err());
    }

    #[test]
    fn test_euclidean_distance() {
        let a = emb(&[0.0, 0.0]);
        let b = emb(&[3.0, 4.0]);
        let d = euclidean_distance(&a, &b).unwrap();
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let e = emb(&[3.0, 4.0]);
        let n = normalize(&e);
        let norm: f64 = n.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_merge_embeddings() {
        let a = emb(&[1.0, 0.0]);
        let b = emb(&[0.0, 1.0]);
        let merged = merge_embeddings(&[a, b], None).unwrap();
        assert!((merged[0] - 0.5).abs() < 1e-10);
        assert!((merged[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_find_most_similar() {
        let query = emb(&[1.0, 0.0, 0.0]);
        let candidates = vec![
            emb(&[0.0, 1.0, 0.0]),
            emb(&[0.9, 0.1, 0.0]),
            emb(&[0.0, 0.0, 1.0]),
        ];
        let (idx, _sim) = find_most_similar(&query, &candidates).unwrap();
        assert_eq!(idx, 1);
    }
}
