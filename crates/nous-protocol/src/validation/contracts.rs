use std::collections::HashSet;

use crate::types::{FallbackStrategy, FallbackTrigger, NousProtocolMessage};

/// Check if contracts are internally consistent.
/// Returns a list of issues found (empty if consistent).
pub fn validate_contracts(message: &NousProtocolMessage) -> Vec<String> {
    let mut issues = Vec::new();

    let requires_ids: HashSet<&str> = message
        .contracts
        .requires
        .iter()
        .map(|c| c.id.as_str())
        .collect();

    let ensures_ids: HashSet<&str> = message
        .contracts
        .ensures
        .iter()
        .map(|c| c.id.as_str())
        .collect();

    // Check that fallbacks reference existing conditions
    for fallback in &message.contracts.fallbacks {
        if let Some(ref cid) = fallback.condition_id {
            let in_requires = requires_ids.contains(cid.as_str());
            let in_ensures = ensures_ids.contains(cid.as_str());

            if !in_requires && !in_ensures {
                issues.push(format!(
                    "Fallback references unknown condition: {cid}"
                ));
            }

            // Check trigger/condition type consistency
            if in_requires && fallback.trigger == FallbackTrigger::PostconditionFailed {
                issues.push(format!(
                    "Fallback for precondition {cid} has postcondition trigger"
                ));
            }
            if in_ensures && fallback.trigger == FallbackTrigger::PreconditionFailed {
                issues.push(format!(
                    "Fallback for postcondition {cid} has precondition trigger"
                ));
            }
        }

        // Validate strategy-specific fields
        match &fallback.strategy {
            FallbackStrategy::Retry { max_retries } => {
                if *max_retries == 0 {
                    issues.push("Retry fallback has max_retries = 0".into());
                }
            }
            FallbackStrategy::Alternative { .. } => {
                // message_id is required by the type system
            }
            FallbackStrategy::Degrade { params } => {
                if params.is_empty() {
                    issues.push(
                        "Degrade fallback should specify degraded params".into(),
                    );
                }
            }
            FallbackStrategy::Abort | FallbackStrategy::Escalate => {}
        }
    }

    // Check for duplicate condition IDs
    let mut all_ids = Vec::new();
    for c in &message.contracts.requires {
        all_ids.push(c.id.as_str());
    }
    for c in &message.contracts.ensures {
        all_ids.push(c.id.as_str());
    }

    let mut seen = HashSet::new();
    let mut duplicates = HashSet::new();
    for id in &all_ids {
        if !seen.insert(*id) {
            duplicates.insert(*id);
        }
    }

    if !duplicates.is_empty() {
        let dup_list: Vec<&str> = duplicates.into_iter().collect();
        issues.push(format!("Duplicate condition IDs: {}", dup_list.join(", ")));
    }

    issues
}

#[cfg(test)]
mod tests {
    use nous_core::types::{Confidence, Embedding};

    use super::*;
    use crate::builders::{IntentBuilder, NousProtocolBuilder};
    use crate::types::{
        Contracts, Fallback, FallbackStrategy, FallbackTrigger, PostCondition,
        PreCondition,
    };

    fn emb() -> Embedding {
        Embedding::new(vec![1.0, 0.0, 0.0]).unwrap()
    }

    fn base_message(contracts: Contracts) -> NousProtocolMessage {
        let intent = IntentBuilder::new()
            .action("test")
            .confidence(0.9)
            .build()
            .unwrap();

        NousProtocolBuilder::new()
            .embedding(emb())
            .intent(intent)
            .contracts(contracts)
            .build()
            .unwrap()
    }

    #[test]
    fn test_valid_contracts() {
        let contracts = Contracts {
            requires: vec![PreCondition {
                id: "pre_1".into(),
                description: "test".into(),
                embedding: emb(),
                threshold: Some(0.8),
                required: true,
            }],
            ensures: vec![PostCondition {
                id: "post_1".into(),
                description: "test".into(),
                embedding: emb(),
                guaranteed_confidence: Confidence::new(0.9),
            }],
            fallbacks: vec![Fallback {
                trigger: FallbackTrigger::PreconditionFailed,
                condition_id: Some("pre_1".into()),
                strategy: FallbackStrategy::Retry { max_retries: 3 },
            }],
        };

        let msg = base_message(contracts);
        let issues = validate_contracts(&msg);
        assert!(issues.is_empty(), "Expected no issues, got: {issues:?}");
    }

    #[test]
    fn test_unknown_condition_reference() {
        let contracts = Contracts {
            requires: vec![],
            ensures: vec![],
            fallbacks: vec![Fallback {
                trigger: FallbackTrigger::ExecutionError,
                condition_id: Some("nonexistent".into()),
                strategy: FallbackStrategy::Abort,
            }],
        };

        let msg = base_message(contracts);
        let issues = validate_contracts(&msg);
        assert!(issues.iter().any(|i| i.contains("unknown condition")));
    }

    #[test]
    fn test_duplicate_condition_ids() {
        let contracts = Contracts {
            requires: vec![PreCondition {
                id: "dup".into(),
                description: "a".into(),
                embedding: emb(),
                threshold: None,
                required: true,
            }],
            ensures: vec![PostCondition {
                id: "dup".into(),
                description: "b".into(),
                embedding: emb(),
                guaranteed_confidence: Confidence::new(0.9),
            }],
            fallbacks: vec![],
        };

        let msg = base_message(contracts);
        let issues = validate_contracts(&msg);
        assert!(issues.iter().any(|i| i.contains("Duplicate")));
    }
}
