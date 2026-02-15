use super::condition::{PostCondition, PreCondition};
use super::fallback::Fallback;

/// Contract block containing pre/post conditions and fallbacks.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Contracts {
    /// Conditions that must be satisfied before execution
    pub requires: Vec<PreCondition>,

    /// Conditions guaranteed after successful execution
    pub ensures: Vec<PostCondition>,

    /// How to handle failures
    pub fallbacks: Vec<Fallback>,
}

impl Contracts {
    /// Create an empty contracts block.
    pub fn empty() -> Self {
        Self {
            requires: Vec::new(),
            ensures: Vec::new(),
            fallbacks: Vec::new(),
        }
    }
}

impl Default for Contracts {
    fn default() -> Self {
        Self::empty()
    }
}
