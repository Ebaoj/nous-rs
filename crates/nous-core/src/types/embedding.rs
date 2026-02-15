use crate::error::{NousError, NousResult};

/// Newtype wrapper for embedding vectors with validation guarantees.
/// Invariants: non-empty, no NaN values.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "Vec<f64>", into = "Vec<f64>"))]
pub struct Embedding(Vec<f64>);

impl Embedding {
    /// Create a new Embedding, validating that it's non-empty and contains no NaN/Inf.
    pub fn new(values: Vec<f64>) -> NousResult<Self> {
        if values.is_empty() {
            return Err(NousError::InvalidEmbedding(
                "embedding cannot be empty".into(),
            ));
        }
        for (i, &v) in values.iter().enumerate() {
            if v.is_nan() {
                return Err(NousError::InvalidEmbedding(format!(
                    "NaN value at index {i}"
                )));
            }
            if v.is_infinite() {
                return Err(NousError::InvalidEmbedding(format!(
                    "infinite value at index {i}"
                )));
            }
        }
        Ok(Self(values))
    }

    /// Create an Embedding without validation (use with care).
    ///
    /// # Safety (logical)
    /// Caller must ensure values are non-empty and contain no NaN/Inf.
    pub fn new_unchecked(values: Vec<f64>) -> Self {
        Self(values)
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.0
    }

    pub fn into_vec(self) -> Vec<f64> {
        self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.0.iter()
    }
}

impl TryFrom<Vec<f64>> for Embedding {
    type Error = NousError;

    fn try_from(values: Vec<f64>) -> Result<Self, Self::Error> {
        Self::new(values)
    }
}

impl From<Embedding> for Vec<f64> {
    fn from(e: Embedding) -> Self {
        e.0
    }
}

impl AsRef<[f64]> for Embedding {
    fn as_ref(&self) -> &[f64] {
        &self.0
    }
}

impl std::ops::Index<usize> for Embedding {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

/// Newtype for confidence values, clamped to [0.0, 1.0].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct Confidence(f64);

impl Confidence {
    /// Create a new Confidence, clamping to [0.0, 1.0].
    pub fn new(value: f64) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    pub fn value(&self) -> f64 {
        self.0
    }

    pub fn full() -> Self {
        Self(1.0)
    }

    pub fn zero() -> Self {
        Self(0.0)
    }

    /// Human-readable confidence level
    pub fn level(&self) -> &'static str {
        if self.0 >= 0.9 {
            "very high"
        } else if self.0 >= 0.75 {
            "high"
        } else if self.0 >= 0.5 {
            "moderate"
        } else if self.0 >= 0.25 {
            "low"
        } else {
            "very low"
        }
    }
}

impl Default for Confidence {
    fn default() -> Self {
        Self(1.0)
    }
}

impl From<f64> for Confidence {
    fn from(v: f64) -> Self {
        Self::new(v)
    }
}

impl From<Confidence> for f64 {
    fn from(c: Confidence) -> Self {
        c.0
    }
}

impl std::fmt::Display for Confidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.1}% ({})", self.0 * 100.0, self.level())
    }
}

/// Newtype for message identifiers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct MessageId(String);

impl MessageId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Generate a new random UUID v4 message ID.
    pub fn random() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for MessageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for MessageId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for MessageId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Embedding model identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct EmbeddingModel(String);

impl EmbeddingModel {
    pub fn new(model: impl Into<String>) -> Self {
        Self(model.into())
    }

    pub fn text_embedding_3_small() -> Self {
        Self("text-embedding-3-small".into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for EmbeddingModel {
    fn default() -> Self {
        Self::text_embedding_3_small()
    }
}

impl std::fmt::Display for EmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unix timestamp in milliseconds
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct Timestamp(u64);

impl Timestamp {
    pub fn new(millis: u64) -> Self {
        Self(millis)
    }

    pub fn now() -> Self {
        Self(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        )
    }

    pub fn as_millis(&self) -> u64 {
        self.0
    }
}

impl From<u64> for Timestamp {
    fn from(millis: u64) -> Self {
        Self(millis)
    }
}
