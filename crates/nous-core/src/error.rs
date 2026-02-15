use thiserror::Error;

/// Core errors for the Nous protocol
#[derive(Debug, Error)]
pub enum NousError {
    #[error("invalid confidence value {0}: must be between 0.0 and 1.0")]
    InvalidConfidence(f64),

    #[error("invalid embedding: {0}")]
    InvalidEmbedding(String),

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("empty input: {0}")]
    EmptyInput(String),

    #[error("quantization error: {0}")]
    Quantization(String),

    #[error("validation error: {0}")]
    Validation(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("build error: {0}")]
    Build(String),

    #[error("{0}")]
    Other(String),
}

pub type NousResult<T> = Result<T, NousError>;
