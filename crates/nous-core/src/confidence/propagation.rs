use super::builder::create_confidence;
use super::map::{ConfidenceMap, ConfidenceSource};
use crate::types::{Confidence, NousConfig};
use std::collections::{HashMap, HashSet};

/// Transformation applied to a message
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Transform {
    #[cfg_attr(feature = "serde", serde(rename = "type"))]
    pub transform_type: String,
    pub agent: crate::types::AgentIdentifier,
    pub timestamp: crate::types::Timestamp,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub params: Option<HashMap<String, String>>,
    /// Impact on confidence (-1 to 1)
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub confidence_delta: Option<f64>,
}

/// Merge multiple confidence maps with optional weights
pub fn merge_confidence(maps: &[ConfidenceMap], weights: Option<&[f64]>) -> ConfidenceMap {
    if maps.is_empty() {
        return create_confidence(0.0).build();
    }

    let effective_weights: Vec<f64> = match weights {
        Some(w) => w.to_vec(),
        None => vec![1.0 / maps.len() as f64; maps.len()],
    };

    let weight_sum: f64 = effective_weights.iter().sum();
    let normalized: Vec<f64> = effective_weights.iter().map(|w| w / weight_sum).collect();

    // Merge overall
    let merged_overall: f64 = maps
        .iter()
        .zip(normalized.iter())
        .map(|(m, w)| m.overall.value() * w)
        .sum();

    // Merge aspects
    let all_aspects: HashSet<String> = maps
        .iter()
        .flat_map(|m| m.aspects.keys().cloned())
        .collect();

    let mut merged_aspects = HashMap::new();
    for aspect in all_aspects {
        let mut sum = 0.0;
        let mut count = 0.0;
        for (i, m) in maps.iter().enumerate() {
            if let Some(&value) = m.aspects.get(&aspect) {
                sum += value * normalized[i];
                count += normalized[i];
            }
        }
        if count > 0.0 {
            merged_aspects.insert(aspect, sum / count);
        }
    }

    // Merge sources
    let merged_sources: Vec<ConfidenceSource> = maps
        .iter()
        .filter_map(|m| m.sources.as_ref())
        .flat_map(|s| s.iter().cloned())
        .collect();

    ConfidenceMap {
        overall: Confidence::new(merged_overall),
        aspects: merged_aspects,
        sources: if merged_sources.is_empty() {
            None
        } else {
            Some(merged_sources)
        },
        dimensions: None,
        minimum_dimension: None,
        minimum_dimension_name: None,
        decision_strategy: None,
    }
}

/// Propagate confidence through a transformation
pub fn propagate_confidence(
    source: &ConfidenceMap,
    transform: &Transform,
    config: &NousConfig,
) -> ConfidenceMap {
    let factor = config
        .propagation_factors
        .get(&transform.transform_type)
        .copied()
        .unwrap_or(0.9);
    let delta = transform.confidence_delta.unwrap_or(0.0);
    let adjusted_factor = (factor + delta).clamp(0.0, 1.0);

    let mut aspects = HashMap::new();
    for (key, &value) in &source.aspects {
        aspects.insert(key.clone(), value * adjusted_factor);
    }

    let mut sources: Vec<ConfidenceSource> = source
        .sources
        .as_ref()
        .cloned()
        .unwrap_or_default();
    sources.push(ConfidenceSource {
        aspect: "transformation".into(),
        reason: format!(
            "Applied {} (factor: {:.2})",
            transform.transform_type, adjusted_factor
        ),
        evidence: transform
            .params
            .as_ref()
            .map(|p| format!("{p:?}")),
    });

    ConfidenceMap {
        overall: Confidence::new(source.overall.value() * adjusted_factor),
        aspects,
        sources: Some(sources),
        dimensions: None,
        minimum_dimension: None,
        minimum_dimension_name: None,
        decision_strategy: None,
    }
}

/// Calculate minimum confidence across multiple maps
pub fn min_confidence(maps: &[ConfidenceMap]) -> ConfidenceMap {
    if maps.is_empty() {
        return create_confidence(0.0).build();
    }

    let min_overall = maps
        .iter()
        .map(|m| m.overall.value())
        .fold(f64::INFINITY, f64::min);

    let all_aspects: HashSet<String> = maps
        .iter()
        .flat_map(|m| m.aspects.keys().cloned())
        .collect();

    let mut min_aspects = HashMap::new();
    for aspect in all_aspects {
        let values: Vec<f64> = maps
            .iter()
            .filter_map(|m| m.aspects.get(&aspect).copied())
            .collect();
        if !values.is_empty() {
            min_aspects.insert(
                aspect,
                values.iter().copied().fold(f64::INFINITY, f64::min),
            );
        }
    }

    let sources: Vec<ConfidenceSource> = maps
        .iter()
        .filter_map(|m| m.sources.as_ref())
        .flat_map(|s| s.iter().cloned())
        .collect();

    ConfidenceMap {
        overall: Confidence::new(min_overall),
        aspects: min_aspects,
        sources: if sources.is_empty() {
            None
        } else {
            Some(sources)
        },
        dimensions: None,
        minimum_dimension: None,
        minimum_dimension_name: None,
        decision_strategy: None,
    }
}
