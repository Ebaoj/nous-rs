use super::map::{ConfidenceMap, ConfidenceSource};
use std::collections::HashMap;

/// Fluent builder for ConfidenceMap
pub struct ConfidenceBuilder {
    overall: f64,
    aspects: HashMap<String, f64>,
    sources: Vec<ConfidenceSource>,
}

impl ConfidenceBuilder {
    pub fn new(overall: f64) -> Self {
        Self {
            overall: overall.clamp(0.0, 1.0),
            aspects: HashMap::new(),
            sources: Vec::new(),
        }
    }

    /// Set confidence for a named aspect
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

    pub fn factual(self, value: f64, reason: Option<&str>) -> Self {
        self.aspect("factual", value, reason)
    }

    pub fn completeness(self, value: f64, reason: Option<&str>) -> Self {
        self.aspect("completeness", value, reason)
    }

    pub fn relevance(self, value: f64, reason: Option<&str>) -> Self {
        self.aspect("relevance", value, reason)
    }

    pub fn reasoning(self, value: f64, reason: Option<&str>) -> Self {
        self.aspect("reasoning", value, reason)
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

    /// Recalculate overall from aspect average
    pub fn recalculate_overall(mut self) -> Self {
        let values: Vec<f64> = self.aspects.values().copied().collect();
        if !values.is_empty() {
            self.overall = values.iter().sum::<f64>() / values.len() as f64;
        }
        self
    }

    pub fn build(self) -> ConfidenceMap {
        ConfidenceMap {
            overall: crate::types::Confidence::new(self.overall),
            aspects: self.aspects,
            sources: if self.sources.is_empty() {
                None
            } else {
                Some(self.sources)
            },
            dimensions: None,
            minimum_dimension: None,
            minimum_dimension_name: None,
            decision_strategy: None,
        }
    }
}

/// Create a confidence map builder
pub fn create_confidence(overall: f64) -> ConfidenceBuilder {
    ConfidenceBuilder::new(overall)
}
