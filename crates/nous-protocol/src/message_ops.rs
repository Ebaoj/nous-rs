use nous_core::confidence::ConfidenceMap;
use nous_core::embeddings::math::cosine_similarity;

use crate::types::{IntentConfidence, NousProtocolMessage};

/// Compute cosine similarity between two protocol messages.
pub fn message_similarity(a: &NousProtocolMessage, b: &NousProtocolMessage) -> f64 {
    cosine_similarity(&a.embedding, &b.embedding).unwrap_or(0.0)
}

/// Check if the confidence is the legacy scalar format (v0.2).
pub fn is_legacy_confidence(confidence: &IntentConfidence) -> bool {
    matches!(confidence, IntentConfidence::Scalar(_))
}

/// Upgrade a legacy scalar confidence to a ConfidenceMap.
/// If already a map, returns it unchanged.
pub fn upgrade_confidence(confidence: &IntentConfidence) -> ConfidenceMap {
    match confidence {
        IntentConfidence::Scalar(c) => ConfidenceMap::new(c.value()),
        IntentConfidence::Map(m) => m.as_ref().clone(),
    }
}

/// Get the effective confidence value for decision-making.
/// Respects the decision strategy if using ConfidenceMap.
pub fn get_effective_confidence(confidence: &IntentConfidence) -> f64 {
    confidence.effective_value()
}

#[cfg(test)]
mod tests {
    use nous_core::confidence::DecisionStrategy;
    use nous_core::types::Confidence;

    use super::*;

    #[test]
    fn test_is_legacy_scalar() {
        let conf = IntentConfidence::Scalar(Confidence::new(0.9));
        assert!(is_legacy_confidence(&conf));
    }

    #[test]
    fn test_is_not_legacy_map() {
        let map = ConfidenceMap::new(0.85);
        let conf = IntentConfidence::Map(Box::new(map));
        assert!(!is_legacy_confidence(&conf));
    }

    #[test]
    fn test_upgrade_scalar() {
        let conf = IntentConfidence::Scalar(Confidence::new(0.7));
        let map = upgrade_confidence(&conf);
        assert!((map.overall.value() - 0.7).abs() < 1e-10);
        assert!(map.aspects.is_empty());
    }

    #[test]
    fn test_upgrade_map_unchanged() {
        let mut original = ConfidenceMap::new(0.85);
        original.aspects.insert("test".into(), 0.9);
        let conf = IntentConfidence::Map(Box::new(original));
        let result = upgrade_confidence(&conf);
        assert!((result.overall.value() - 0.85).abs() < 1e-10);
        assert_eq!(result.aspects.get("test"), Some(&0.9));
    }

    #[test]
    fn test_effective_confidence_scalar() {
        let conf = IntentConfidence::Scalar(Confidence::new(0.8));
        assert!((get_effective_confidence(&conf) - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_effective_confidence_map_overall() {
        let map = ConfidenceMap::new(0.75);
        let conf = IntentConfidence::Map(Box::new(map));
        assert!((get_effective_confidence(&conf) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_effective_confidence_map_minimum_strategy() {
        let mut map = ConfidenceMap::new(0.9);
        map.decision_strategy = Some(DecisionStrategy::Minimum);
        map.minimum_dimension = Some(0.6);
        let conf = IntentConfidence::Map(Box::new(map));
        assert!((get_effective_confidence(&conf) - 0.6).abs() < 1e-10);
    }
}
