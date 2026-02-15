use crate::error::{NousError, NousResult};
use crate::types::Embedding;
use super::types::*;

/// Binary quantization: convert embedding to 1-bit representation.
/// Each dimension becomes 0 or 1 based on whether it's above the mean.
///
/// 1536 floats (6144 bytes) -> 1536 bits (192 bytes) = 97% reduction.
/// Trade-off: ~5-10% accuracy loss in similarity calculations.
pub fn binary_quantize(embedding: &Embedding, source_model: &str) -> QuantizedEmbedding {
    let dimensions = embedding.len();
    let byte_length = dimensions.div_ceil(8);
    let mut data = vec![0u8; byte_length];

    // Calculate mean for centering
    let mean: f64 = embedding.iter().sum::<f64>() / dimensions as f64;

    // Pack bits: 1 if above mean, 0 if below
    for i in 0..dimensions {
        if embedding[i] > mean {
            let byte_index = i / 8;
            let bit_index = i % 8;
            data[byte_index] |= 1 << bit_index;
        }
    }

    let original_bytes = dimensions * 4; // Float32 = 4 bytes

    QuantizedEmbedding {
        method: QuantizationMethod::Binary,
        data,
        dimensions,
        params: QuantizationParams {
            mean: Some(mean),
            scale: None,
        },
        source_model: source_model.to_string(),
        stats: CompressionStats {
            original_bytes,
            compressed_bytes: byte_length,
            compression_ratio: original_bytes as f64 / byte_length as f64,
        },
    }
}

/// Reconstruct approximation from binary quantized embedding.
/// This is lossy - reconstructed values are normalized to +/- 1/sqrt(n).
pub fn binary_dequantize(quantized: &QuantizedEmbedding) -> NousResult<Vec<f64>> {
    if quantized.method != QuantizationMethod::Binary {
        return Err(NousError::Quantization(
            "expected binary quantized embedding".into(),
        ));
    }

    let magnitude = 1.0 / (quantized.dimensions as f64).sqrt();
    let embedding: Vec<f64> = (0..quantized.dimensions)
        .map(|i| {
            let byte_index = i / 8;
            let bit_index = i % 8;
            let bit = (quantized.data[byte_index] >> bit_index) & 1;
            if bit == 1 { magnitude } else { -magnitude }
        })
        .collect();

    Ok(embedding)
}

/// Calculate Hamming distance between two binary quantized embeddings.
/// Much faster than cosine similarity for binary vectors.
pub fn hamming_distance(a: &QuantizedEmbedding, b: &QuantizedEmbedding) -> NousResult<u32> {
    if a.method != QuantizationMethod::Binary || b.method != QuantizationMethod::Binary {
        return Err(NousError::Quantization(
            "both embeddings must be binary quantized".into(),
        ));
    }
    if a.data.len() != b.data.len() {
        return Err(NousError::DimensionMismatch {
            expected: a.data.len(),
            got: b.data.len(),
        });
    }

    let distance: u32 = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum();

    Ok(distance)
}

/// Convert Hamming distance to approximate cosine similarity.
/// Based on: cosine ≈ 1 - 2 * (hamming / dimensions)
pub fn hamming_to_cosine_similarity(hamming_distance: u32, dimensions: usize) -> f64 {
    1.0 - (2.0 * hamming_distance as f64 / dimensions as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_roundtrip() {
        let values: Vec<f64> = (0..128).map(|i| (i as f64 - 64.0) / 64.0).collect();
        let emb = Embedding::new(values.clone()).unwrap();
        let quantized = binary_quantize(&emb, "test");

        assert_eq!(quantized.method, QuantizationMethod::Binary);
        assert_eq!(quantized.dimensions, 128);
        assert_eq!(quantized.data.len(), 16); // 128/8

        // Verify compression ratio
        assert!(quantized.stats.compression_ratio > 30.0);

        let dequantized = binary_dequantize(&quantized).unwrap();
        assert_eq!(dequantized.len(), 128);
    }

    #[test]
    fn test_hamming_distance_identical() {
        let values: Vec<f64> = (0..64).map(|i| i as f64).collect();
        let emb = Embedding::new(values).unwrap();
        let q = binary_quantize(&emb, "test");
        assert_eq!(hamming_distance(&q, &q).unwrap(), 0);
    }

    #[test]
    fn test_hamming_to_cosine() {
        let sim = hamming_to_cosine_similarity(0, 100);
        assert!((sim - 1.0).abs() < 1e-10);

        let sim = hamming_to_cosine_similarity(50, 100);
        assert!((sim - 0.0).abs() < 1e-10);
    }
}
