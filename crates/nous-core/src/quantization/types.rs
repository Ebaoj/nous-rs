/// Quantization method used
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum QuantizationMethod {
    Binary,
    Scalar,
}

/// Reconstruction parameters for dequantization
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct QuantizationParams {
    /// Mean value (for centering in binary quant)
    pub mean: Option<f64>,
    /// Scale factor (for scalar quant: (max-min)/255)
    pub scale: Option<f64>,
}

/// Compression statistics
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CompressionStats {
    pub original_bytes: usize,
    pub compressed_bytes: usize,
    pub compression_ratio: f64,
}

/// Quantized embedding representation
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct QuantizedEmbedding {
    pub method: QuantizationMethod,
    pub data: Vec<u8>,
    pub dimensions: usize,
    pub params: QuantizationParams,
    pub source_model: String,
    pub stats: CompressionStats,
}
