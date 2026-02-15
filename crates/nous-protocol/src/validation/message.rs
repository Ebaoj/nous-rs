use crate::types::NousProtocolMessage;
use crate::validation::contracts::validate_contracts;
use crate::validation::params::validate_params;

/// A single validation issue (error or warning).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ValidationIssue {
    pub field: String,
    pub message: String,
}

/// Result of comprehensive message validation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MessageValidationResult {
    pub valid: bool,
    pub errors: Vec<ValidationIssue>,
    pub warnings: Vec<ValidationIssue>,
}

/// Comprehensive validation of a NousProtocolMessage.
pub fn validate_message(message: &NousProtocolMessage) -> MessageValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    // ID validation
    if message.id.as_str().is_empty() {
        errors.push(ValidationIssue {
            field: "id".into(),
            message: "ID is required".into(),
        });
    }

    // Embedding validation (Embedding newtype already ensures non-empty + no NaN)
    // but we can still check for zero-length (would be caught by Embedding::new)

    // Intent validation
    if message.intent.action.is_empty() {
        errors.push(ValidationIssue {
            field: "intent.action".into(),
            message: "Intent action is required".into(),
        });
    }

    let confidence_value = message.intent.confidence.effective_value();
    if !(0.0..=1.0).contains(&confidence_value) {
        errors.push(ValidationIssue {
            field: "intent.confidence".into(),
            message: "Confidence must be between 0 and 1".into(),
        });
    }
    if confidence_value < 0.5 {
        warnings.push(ValidationIssue {
            field: "intent.confidence".into(),
            message: "Low confidence intent -- consider adding alternatives".into(),
        });
    }

    // Parameter validation
    let param_errors = validate_params(&message.params);
    for err in param_errors {
        errors.push(ValidationIssue {
            field: "params".into(),
            message: err,
        });
    }

    // Warn about high-uncertainty params without alternatives
    for param in &message.params {
        if param.uncertainty.value() > 0.5 && param.alternatives.is_empty() {
            warnings.push(ValidationIssue {
                field: format!("params.{}", param.name),
                message: format!(
                    "High uncertainty ({:.2}) without alternatives",
                    param.uncertainty.value()
                ),
            });
        }
    }

    // Contract precondition embedding validation
    for condition in &message.contracts.requires {
        if condition.embedding.is_empty() {
            errors.push(ValidationIssue {
                field: format!("contracts.requires.{}", condition.id),
                message: "Precondition embedding is required".into(),
            });
        }
    }

    // Contract postcondition embedding validation
    for condition in &message.contracts.ensures {
        if condition.embedding.is_empty() {
            errors.push(ValidationIssue {
                field: format!("contracts.ensures.{}", condition.id),
                message: "Postcondition embedding is required".into(),
            });
        }
    }

    // Contract consistency validation
    let contract_issues = validate_contracts(message);
    for issue in contract_issues {
        warnings.push(ValidationIssue {
            field: "contracts".into(),
            message: issue,
        });
    }

    // Meta validation
    if message.meta.sender.id.is_empty() {
        errors.push(ValidationIssue {
            field: "meta.sender".into(),
            message: "Sender ID is required".into(),
        });
    }

    MessageValidationResult {
        valid: errors.is_empty(),
        errors,
        warnings,
    }
}

#[cfg(test)]
mod tests {
    use nous_core::types::Embedding;

    use super::*;
    use crate::builders::{IntentBuilder, NousProtocolBuilder};

    fn emb() -> Embedding {
        Embedding::new(vec![1.0, 0.0, 0.0]).unwrap()
    }

    #[test]
    fn test_validate_valid_message() {
        let intent = IntentBuilder::new()
            .action("test")
            .confidence(0.9)
            .build()
            .unwrap();

        let msg = NousProtocolBuilder::new()
            .embedding(emb())
            .intent(intent)
            .build()
            .unwrap();

        let result = validate_message(&msg);
        assert!(result.valid, "Expected valid, got errors: {:?}", result.errors);
    }

    #[test]
    fn test_validate_low_confidence_warning() {
        let intent = IntentBuilder::new()
            .action("test")
            .confidence(0.3)
            .build()
            .unwrap();

        let msg = NousProtocolBuilder::new()
            .embedding(emb())
            .intent(intent)
            .build()
            .unwrap();

        let result = validate_message(&msg);
        assert!(result.valid); // warnings don't make it invalid
        assert!(!result.warnings.is_empty());
    }
}
