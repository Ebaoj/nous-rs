use nous_core::confidence::ConfidenceMap;
use nous_core::types::{Confidence, Embedding};

/// How confidence is represented in an intent.
/// Supports both legacy scalar (v0.2) and multidimensional map (v0.3).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(untagged))]
pub enum IntentConfidence {
    /// Legacy scalar confidence (v0.2)
    Scalar(Confidence),
    /// Multidimensional confidence map (v0.3)
    Map(Box<ConfidenceMap>),
}

impl IntentConfidence {
    /// Get the effective overall confidence value.
    pub fn effective_value(&self) -> f64 {
        match self {
            IntentConfidence::Scalar(c) => c.value(),
            IntentConfidence::Map(m) => m.effective_confidence(),
        }
    }
}

impl Default for IntentConfidence {
    fn default() -> Self {
        IntentConfidence::Scalar(Confidence::full())
    }
}

impl From<f64> for IntentConfidence {
    fn from(v: f64) -> Self {
        IntentConfidence::Scalar(Confidence::new(v))
    }
}

impl From<Confidence> for IntentConfidence {
    fn from(c: Confidence) -> Self {
        IntentConfidence::Scalar(c)
    }
}

impl From<ConfidenceMap> for IntentConfidence {
    fn from(m: ConfidenceMap) -> Self {
        IntentConfidence::Map(Box::new(m))
    }
}

/// Action intent with confidence.
/// Represents what the agent wants to accomplish.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Intent {
    /// Action identifier (e.g., "query_database", "send_email")
    pub action: String,

    /// Confidence in the intent interpretation
    pub confidence: IntentConfidence,

    /// Optional description for debugging
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub description: Option<String>,

    /// Semantic embedding of the intent (for matching)
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none")
    )]
    pub embedding: Option<Embedding>,
}
