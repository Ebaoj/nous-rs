use super::map::{ConfidenceMap, ConfidenceDimensions, ConfidenceSource, DecisionStrategy};
use crate::types::Confidence;
use std::collections::HashMap;

/// Suggested action based on confidence analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestedAction {
    Proceed,
    Verify,
    Abort,
}

/// Builder for multidimensional confidence maps.
/// Captures nuanced uncertainty across decision-critical dimensions.
pub struct MultidimensionalConfidenceBuilder {
    overall: f64,
    aspects: HashMap<String, f64>,
    dimensions: ConfidenceDimensions,
    sources: Vec<ConfidenceSource>,
    decision_strategy: DecisionStrategy,
    minimum_dimension: Option<f64>,
    minimum_dimension_name: Option<String>,
}

impl MultidimensionalConfidenceBuilder {
    pub fn new(overall: f64) -> Self {
        Self {
            overall: overall.clamp(0.0, 1.0),
            aspects: HashMap::new(),
            dimensions: ConfidenceDimensions::default(),
            sources: Vec::new(),
            decision_strategy: DecisionStrategy::Minimum,
            minimum_dimension: None,
            minimum_dimension_name: None,
        }
    }

    // Core dimensions

    pub fn factual(self, value: f64, reason: Option<&str>) -> Self {
        self.dimension("factual", value, reason)
    }

    pub fn temporal(self, value: f64, reason: Option<&str>) -> Self {
        self.dimension("temporal", value, reason)
    }

    pub fn identity(self, value: f64, reason: Option<&str>) -> Self {
        self.dimension("identity", value, reason)
    }

    pub fn intent(self, value: f64, reason: Option<&str>) -> Self {
        self.dimension("intent", value, reason)
    }

    pub fn reversible(self, value: f64, reason: Option<&str>) -> Self {
        self.dimension("reversible", value, reason)
    }

    pub fn authorized(self, value: f64, reason: Option<&str>) -> Self {
        self.dimension("authorized", value, reason)
    }

    pub fn completeness(self, value: f64, reason: Option<&str>) -> Self {
        self.dimension("completeness", value, reason)
    }

    /// Set a dimension value
    pub fn dimension(mut self, name: &str, value: f64, reason: Option<&str>) -> Self {
        self.dimensions.set(name, value);
        if let Some(r) = reason {
            self.sources.push(ConfidenceSource {
                aspect: format!("dimension:{name}"),
                reason: r.into(),
                evidence: None,
            });
        }
        self.recalculate_minimum();
        self
    }

    /// Set an aspect (from v0.1 ConfidenceMap)
    pub fn aspect(mut self, name: impl Into<String>, value: f64, reason: Option<&str>) -> Self {
        let name = name.into();
        self.aspects.insert(name.clone(), value.clamp(0.0, 1.0));
        if let Some(r) = reason {
            self.sources.push(ConfidenceSource {
                aspect: name,
                reason: r.into(),
                evidence: None,
            });
        }
        self
    }

    pub fn relevance(self, value: f64, reason: Option<&str>) -> Self {
        self.aspect("relevance", value, reason)
    }

    pub fn reasoning(self, value: f64, reason: Option<&str>) -> Self {
        self.aspect("reasoning", value, reason)
    }

    /// Set the decision strategy
    pub fn strategy(mut self, strategy: DecisionStrategy) -> Self {
        self.decision_strategy = strategy;
        self
    }

    /// Add a source/justification
    pub fn source(
        mut self,
        aspect: impl Into<String>,
        reason: impl Into<String>,
        evidence: Option<String>,
    ) -> Self {
        self.sources.push(ConfidenceSource {
            aspect: aspect.into(),
            reason: reason.into(),
            evidence,
        });
        self
    }

    /// Check if we should proceed based on threshold
    pub fn should_proceed(&self, threshold: f64) -> bool {
        self.get_effective_confidence() >= threshold
    }

    /// Get the minimum dimension info
    pub fn get_minimum_dimension(&self) -> Option<(&str, f64)> {
        self.minimum_dimension_name
            .as_deref()
            .zip(self.minimum_dimension)
    }

    /// Suggest an action based on confidence analysis
    pub fn suggest_action(&self, proceed_threshold: f64, verify_threshold: f64) -> SuggestedAction {
        let min_dim = self.minimum_dimension.unwrap_or(self.overall);

        // Critical irreversible actions
        if let Some(rev) = self.dimensions.reversible {
            if rev < 0.30 && min_dim < 0.80 {
                return SuggestedAction::Abort;
            }
        }

        // Identity/authorization concerns
        if let Some(id) = self.dimensions.identity {
            if id < verify_threshold {
                return SuggestedAction::Abort;
            }
        }
        if let Some(auth) = self.dimensions.authorized {
            if auth < verify_threshold {
                return SuggestedAction::Abort;
            }
        }

        if min_dim >= proceed_threshold {
            SuggestedAction::Proceed
        } else if min_dim >= verify_threshold {
            SuggestedAction::Verify
        } else {
            SuggestedAction::Abort
        }
    }

    /// Get dimensions below a threshold, sorted ascending
    pub fn dimensions_below(&self, threshold: f64) -> Vec<(String, f64)> {
        let mut below: Vec<(String, f64)> = self
            .dimensions
            .iter()
            .into_iter()
            .filter(|(_, v)| *v < threshold)
            .map(|(n, v)| (n.to_string(), v))
            .collect();
        below.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        below
    }

    /// Get effective confidence based on strategy
    pub fn get_effective_confidence(&self) -> f64 {
        match self.decision_strategy {
            DecisionStrategy::Minimum => self.minimum_dimension.unwrap_or(self.overall),
            DecisionStrategy::Weighted => self.calculate_weighted(),
            DecisionStrategy::Overall => self.overall,
        }
    }

    pub fn build(mut self) -> ConfidenceMap {
        self.recalculate_minimum();
        ConfidenceMap {
            overall: Confidence::new(self.overall),
            aspects: self.aspects,
            sources: if self.sources.is_empty() {
                None
            } else {
                Some(self.sources)
            },
            dimensions: Some(self.dimensions),
            minimum_dimension: self.minimum_dimension,
            minimum_dimension_name: self.minimum_dimension_name,
            decision_strategy: Some(self.decision_strategy),
        }
    }

    fn recalculate_minimum(&mut self) {
        let mut min_value = f64::MAX;
        let mut min_name: Option<String> = None;

        for (name, value) in self.dimensions.iter() {
            if value < min_value {
                min_value = value;
                min_name = Some(name.to_string());
            }
        }

        if let Some(name) = min_name {
            self.minimum_dimension = Some(min_value);
            self.minimum_dimension_name = Some(name);
        }
    }

    fn calculate_weighted(&self) -> f64 {
        let values: Vec<f64> = self.dimensions.iter().into_iter().map(|(_, v)| v).collect();
        if values.is_empty() {
            return self.overall;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// Factory function for multidimensional confidence
pub fn create_multidimensional_confidence(overall: f64) -> MultidimensionalConfidenceBuilder {
    MultidimensionalConfidenceBuilder::new(overall)
}
