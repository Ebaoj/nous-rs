use std::collections::HashMap;
use crate::types::Confidence;

/// Source/justification for a confidence level
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConfidenceSource {
    pub aspect: String,
    pub reason: String,
    pub evidence: Option<String>,
}

/// Contextual dimensions for decision-critical confidence
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConfidenceDimensions {
    /// Factual correctness
    pub factual: Option<f64>,
    /// Temporal validity
    pub temporal: Option<f64>,
    /// Identity certainty
    pub identity: Option<f64>,
    /// Intent certainty
    pub intent: Option<f64>,
    /// Reversibility (low = irreversible)
    pub reversible: Option<f64>,
    /// Authorization
    pub authorized: Option<f64>,
    /// Completeness
    pub completeness: Option<f64>,
    /// Custom dimensions
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub custom: HashMap<String, f64>,
}

impl ConfidenceDimensions {
    /// Get a dimension value by name
    pub fn get(&self, name: &str) -> Option<f64> {
        match name {
            "factual" => self.factual,
            "temporal" => self.temporal,
            "identity" => self.identity,
            "intent" => self.intent,
            "reversible" => self.reversible,
            "authorized" => self.authorized,
            "completeness" => self.completeness,
            _ => self.custom.get(name).copied(),
        }
    }

    /// Set a dimension value by name
    pub fn set(&mut self, name: &str, value: f64) {
        let clamped = value.clamp(0.0, 1.0);
        match name {
            "factual" => self.factual = Some(clamped),
            "temporal" => self.temporal = Some(clamped),
            "identity" => self.identity = Some(clamped),
            "intent" => self.intent = Some(clamped),
            "reversible" => self.reversible = Some(clamped),
            "authorized" => self.authorized = Some(clamped),
            "completeness" => self.completeness = Some(clamped),
            _ => {
                self.custom.insert(name.to_string(), clamped);
            }
        }
    }

    /// Iterate over all set dimensions
    pub fn iter(&self) -> Vec<(&str, f64)> {
        let mut result = Vec::new();
        if let Some(v) = self.factual { result.push(("factual", v)); }
        if let Some(v) = self.temporal { result.push(("temporal", v)); }
        if let Some(v) = self.identity { result.push(("identity", v)); }
        if let Some(v) = self.intent { result.push(("intent", v)); }
        if let Some(v) = self.reversible { result.push(("reversible", v)); }
        if let Some(v) = self.authorized { result.push(("authorized", v)); }
        if let Some(v) = self.completeness { result.push(("completeness", v)); }
        for (k, v) in &self.custom {
            result.push((k.as_str(), *v));
        }
        result
    }
}

/// Decision strategy for multidimensional confidence
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum DecisionStrategy {
    /// Use overall confidence (legacy)
    #[default]
    Overall,
    /// Use the lowest dimension value (safest)
    Minimum,
    /// Use weighted average
    Weighted,
}

/// Multidimensional confidence map
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConfidenceMap {
    /// Overall confidence score
    pub overall: Confidence,
    /// Confidence by aspect
    pub aspects: HashMap<String, f64>,
    /// Sources/justifications
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub sources: Option<Vec<ConfidenceSource>>,
    /// Contextual dimensions (v0.3)
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub dimensions: Option<ConfidenceDimensions>,
    /// Lowest dimension value
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub minimum_dimension: Option<f64>,
    /// Name of the lowest dimension
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub minimum_dimension_name: Option<String>,
    /// Decision strategy
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub decision_strategy: Option<DecisionStrategy>,
}

impl ConfidenceMap {
    /// Create a simple confidence map with just an overall score
    pub fn new(overall: f64) -> Self {
        Self {
            overall: Confidence::new(overall),
            aspects: HashMap::new(),
            sources: None,
            dimensions: None,
            minimum_dimension: None,
            minimum_dimension_name: None,
            decision_strategy: None,
        }
    }

    /// Get the effective confidence based on decision strategy
    pub fn effective_confidence(&self) -> f64 {
        match self.decision_strategy.unwrap_or_default() {
            DecisionStrategy::Overall => self.overall.value(),
            DecisionStrategy::Minimum => {
                self.minimum_dimension.unwrap_or(self.overall.value())
            }
            DecisionStrategy::Weighted => self.overall.value(),
        }
    }

    /// Check if confidence meets a threshold
    pub fn meets_threshold(&self, threshold: f64, aspect: Option<&str>) -> bool {
        if let Some(aspect_name) = aspect {
            self.aspects
                .get(aspect_name)
                .map(|&v| v >= threshold)
                .unwrap_or(false)
        } else {
            self.overall.value() >= threshold
        }
    }

    /// Format for display
    pub fn format(&self) -> String {
        let mut lines = vec![format!("Overall: {}", self.overall)];

        for (aspect, value) in &self.aspects {
            lines.push(format!("  {aspect}: {:.1}%", value * 100.0));
        }

        if let Some(dims) = &self.dimensions {
            lines.push("Dimensions:".into());
            for (name, value) in dims.iter() {
                let marker = if self.minimum_dimension_name.as_deref() == Some(name) {
                    " (minimum)"
                } else {
                    ""
                };
                lines.push(format!("  {name}: {:.1}%{marker}", value * 100.0));
            }
        }

        if let Some(sources) = &self.sources {
            if !sources.is_empty() {
                lines.push("Sources:".into());
                for source in sources {
                    lines.push(format!("  - [{}] {}", source.aspect, source.reason));
                }
            }
        }

        lines.join("\n")
    }
}

impl Default for ConfidenceMap {
    fn default() -> Self {
        Self::new(1.0)
    }
}
