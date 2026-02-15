use crate::types::{Contracts, Fallback, PostCondition, PreCondition};

/// Fluent builder for creating `Contracts` values.
#[derive(Debug, Default)]
pub struct ContractBuilder {
    requires: Vec<PreCondition>,
    ensures: Vec<PostCondition>,
    fallbacks: Vec<Fallback>,
}

impl ContractBuilder {
    /// Create a new builder instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a precondition.
    pub fn require(mut self, condition: PreCondition) -> Self {
        self.requires.push(condition);
        self
    }

    /// Add a postcondition.
    pub fn ensure(mut self, condition: PostCondition) -> Self {
        self.ensures.push(condition);
        self
    }

    /// Add a fallback.
    pub fn fallback(mut self, fallback: Fallback) -> Self {
        self.fallbacks.push(fallback);
        self
    }

    /// Build the contracts block.
    pub fn build(self) -> Contracts {
        Contracts {
            requires: self.requires,
            ensures: self.ensures,
            fallbacks: self.fallbacks,
        }
    }
}

#[cfg(test)]
mod tests {
    use nous_core::types::{Confidence, Embedding};

    use super::*;
    use crate::types::{FallbackStrategy, FallbackTrigger};

    fn test_embedding() -> Embedding {
        Embedding::new(vec![1.0, 0.0, 0.0]).unwrap()
    }

    #[test]
    fn test_empty_contracts() {
        let contracts = ContractBuilder::new().build();
        assert!(contracts.requires.is_empty());
        assert!(contracts.ensures.is_empty());
        assert!(contracts.fallbacks.is_empty());
    }

    #[test]
    fn test_full_contracts() {
        let contracts = ContractBuilder::new()
            .require(PreCondition {
                id: "pre_1".into(),
                description: "Database is available".into(),
                embedding: test_embedding(),
                threshold: Some(0.8),
                required: true,
            })
            .ensure(PostCondition {
                id: "post_1".into(),
                description: "Query returns results".into(),
                embedding: test_embedding(),
                guaranteed_confidence: Confidence::new(0.9),
            })
            .fallback(Fallback {
                trigger: FallbackTrigger::ExecutionError,
                condition_id: None,
                strategy: FallbackStrategy::Retry { max_retries: 3 },
            })
            .build();

        assert_eq!(contracts.requires.len(), 1);
        assert_eq!(contracts.ensures.len(), 1);
        assert_eq!(contracts.fallbacks.len(), 1);
    }
}
