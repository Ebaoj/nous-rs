use crate::error::{NousError, NousResult};
use crate::types::Embedding;
use super::types::*;

/// Scalar quantization: convert Float64 to Uint8.
/// 1536 * 8 bytes -> 1536 * 1 byte = ~87.5% reduction.
/// Better accuracy than binary, but less compression.
pub fn scalar_quantize(embedding: &Embedding, source_model: &str) -> QuantizedEmbedding {
    let dimensions = embedding.len();

    // Find min/max for scaling
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &val in embedding.iter() {
        if val < min { min = val; }
        if val > max { max = val; }
    }

    let range = max - min;
    let scale = if range > 0.0 { range / 255.0 } else { 1.0 };

    let data: Vec<u8> = embedding
        .iter()
        .map(|&v| {
            if range > 0.0 {
                ((v - min) / scale).round() as u8
            } else {
                128u8
            }
        })
        .collect();

    let original_bytes = dimensions * 8; // Float64 = 8 bytes
    let compressed_bytes = dimensions + 16; // data + 2 floats for params

    QuantizedEmbedding {
        method: QuantizationMethod::Scalar,
        data,
        dimensions,
        params: QuantizationParams {
            mean: Some(min), // "mean" stores min value for reconstruction
            scale: Some(scale),
        },
        source_model: source_model.to_string(),
        stats: CompressionStats {
            original_bytes,
            compressed_bytes,
            compression_ratio: original_bytes as f64 / compressed_bytes as f64,
        },
    }
}

/// Reconstruct embedding from scalar quantized representation.
pub fn scalar_dequantize(quantized: &QuantizedEmbedding) -> NousResult<Vec<f64>> {
    if quantized.method != QuantizationMethod::Scalar {
        return Err(NousError::Quantization(
            "expected scalar quantized embedding".into(),
        ));
    }

    let min = quantized.params.mean.unwrap_or(0.0);
    let scale = quantized.params.scale.unwrap_or(1.0);

    let embedding: Vec<f64> = quantized
        .data
        .iter()
        .map(|&v| v as f64 * scale + min)
        .collect();

    Ok(embedding)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_roundtrip() {
        let values: Vec<f64> = (0..128).map(|i| (i as f64 - 64.0) / 64.0).collect();
        let emb = Embedding::new(values.clone()).unwrap();
        let quantized = scalar_quantize(&emb, "test");

        assert_eq!(quantized.method, QuantizationMethod::Scalar);
        assert_eq!(quantized.dimensions, 128);

        let dequantized = scalar_dequantize(&quantized).unwrap();
        assert_eq!(dequantized.len(), 128);

        // Check roundtrip accuracy (should be very close)
        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            assert!(
                (orig - deq).abs() < 0.01,
                "orig={orig}, deq={deq}"
            );
        }
    }

    #[test]
    fn test_scalar_compression_ratio() {
        let values: Vec<f64> = (0..1536).map(|i| i as f64 / 1536.0).collect();
        let emb = Embedding::new(values).unwrap();
        let quantized = scalar_quantize(&emb, "test");

        // Should achieve ~7x compression (float64 -> uint8 + params)
        assert!(quantized.stats.compression_ratio > 6.0);
    }
}
