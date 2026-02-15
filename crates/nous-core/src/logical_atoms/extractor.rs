use regex::Regex;
use super::types::*;

/// Extract logical atoms using rule-based patterns (no LLM needed).
/// Useful for quick validation and testing.
pub fn extract_logical_atoms_sync(text: &str) -> LogicalAtoms {
    let mut atoms = LogicalAtoms::empty(text);

    extract_quantifiers(text, &mut atoms);
    extract_negations(text, &mut atoms);
    extract_numbers(text, &mut atoms);
    extract_constraints(text, &mut atoms);
    extract_orderings(text, &mut atoms);

    atoms
}

fn extract_quantifiers(text: &str, atoms: &mut LogicalAtoms) {
    let patterns: Vec<(Regex, QuantifierType)> = vec![
        (Regex::new(r"(?i)\ball\s+(\w+(?:\s+\w+)?)").unwrap(), QuantifierType::All),
        (Regex::new(r"(?i)\bevery\s+(\w+)").unwrap(), QuantifierType::All),
        (Regex::new(r"(?i)\beach\s+(\w+)").unwrap(), QuantifierType::All),
        (Regex::new(r"(?i)\bsome\s+(\w+(?:\s+\w+)?)").unwrap(), QuantifierType::Some),
        (Regex::new(r"(?i)\bany\s+(\w+)").unwrap(), QuantifierType::Some),
        (Regex::new(r"(?i)\bno\s+(\w+(?:\s+\w+)?)").unwrap(), QuantifierType::None),
        (Regex::new(r"(?i)\bnone\s+of\s+the\s+(\w+)").unwrap(), QuantifierType::None),
        (Regex::new(r"(?i)\bmost\s+(\w+(?:\s+\w+)?)").unwrap(), QuantifierType::Most),
        (Regex::new(r"(?i)\bfew\s+(\w+)").unwrap(), QuantifierType::Few),
        (Regex::new(r"(?i)\bexactly\s+\d+\s+(\w+)").unwrap(), QuantifierType::Exactly),
        (Regex::new(r"(?i)\bat\s+least\s+\d+\s+(\w+)").unwrap(), QuantifierType::AtLeast),
        (Regex::new(r"(?i)\bat\s+most\s+\d+\s+(\w+)").unwrap(), QuantifierType::AtMost),
    ];

    for (pattern, qtype) in &patterns {
        for cap in pattern.captures_iter(text) {
            atoms.quantifiers.push(ExtractedQuantifier {
                quantifier_type: qtype.clone(),
                scope: cap.get(1).map(|m| m.as_str().to_string()).unwrap_or_default(),
                original: cap.get(0).map(|m| m.as_str().to_string()).unwrap_or_default(),
            });
        }
    }
}

fn extract_negations(text: &str, atoms: &mut LogicalAtoms) {
    let patterns = vec![
        Regex::new(r"(?i)\b(not|don't|doesn't|won't|can't|cannot|never|isn't|aren't|wasn't|weren't)\s+(\w+(?:\s+\w+)?)").unwrap(),
        Regex::new(r"(?i)\b(must\s+not|should\s+not|will\s+not|do\s+not|does\s+not)\s+(\w+)").unwrap(),
    ];

    for pattern in &patterns {
        for cap in pattern.captures_iter(text) {
            atoms.negations.push(ExtractedNegation {
                target: cap.get(2).map(|m| m.as_str().to_string()).unwrap_or_default(),
                negated: true,
                original: cap.get(0).map(|m| m.as_str().to_string()).unwrap_or_default(),
            });
        }
    }
}

fn extract_numbers(text: &str, atoms: &mut LogicalAtoms) {
    // Pattern: name: 30s, delay: 100ms
    let p1 = Regex::new(r"(?i)(\w+):\s*(\d+(?:\.\d+)?)\s*(ms|s|sec|seconds?|min|minutes?|hrs?|hours?|%|kb|mb|gb)").unwrap();
    for cap in p1.captures_iter(text) {
        let name = cap.get(1).map(|m| m.as_str()).unwrap_or("value");
        let value: f64 = cap.get(2).and_then(|m| m.as_str().parse().ok()).unwrap_or(0.0);
        let unit = cap.get(3).map(|m| m.as_str().to_string());
        atoms.numbers.push(ExtractedNumber {
            name: name.to_string(),
            value,
            unit,
            original: cap.get(0).map(|m| m.as_str().to_string()).unwrap_or_default(),
        });
    }

    // Pattern: 30 seconds, 100 milliseconds
    let p2 = Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|sec|seconds?|min|minutes?|hrs?|hours?|days?|%|kb|mb|gb)\b").unwrap();
    for cap in p2.captures_iter(text) {
        let value: f64 = cap.get(1).and_then(|m| m.as_str().parse().ok()).unwrap_or(0.0);
        let unit = cap.get(2).map(|m| m.as_str().to_string());
        atoms.numbers.push(ExtractedNumber {
            name: "value".to_string(),
            value,
            unit,
            original: cap.get(0).map(|m| m.as_str().to_string()).unwrap_or_default(),
        });
    }

    // Pattern: max 100, minimum 5
    let p3 = Regex::new(r"(?i)(max|min|maximum|minimum|limit|timeout|delay|retries?)\s*[=:of]?\s*(\d+(?:\.\d+)?)\s*(\w*)").unwrap();
    for cap in p3.captures_iter(text) {
        let name = cap.get(1).map(|m| m.as_str()).unwrap_or("value");
        let value: f64 = cap.get(2).and_then(|m| m.as_str().parse().ok()).unwrap_or(0.0);
        let unit = cap.get(3).map(|m| m.as_str().to_string()).filter(|s| !s.is_empty());
        atoms.numbers.push(ExtractedNumber {
            name: name.to_string(),
            value,
            unit,
            original: cap.get(0).map(|m| m.as_str().to_string()).unwrap_or_default(),
        });
    }
}

fn extract_constraints(text: &str, atoms: &mut LogicalAtoms) {
    let patterns: Vec<(Regex, ConstraintType)> = vec![
        (Regex::new(r"(?i)maximum\s+(?:of\s+)?(\d+)\s+(\w+)").unwrap(), ConstraintType::Max),
        (Regex::new(r"(?i)minimum\s+(?:of\s+)?(\d+)\s+(\w+)").unwrap(), ConstraintType::Min),
        (Regex::new(r"(?i)at\s+most\s+(\d+)\s+(\w+)").unwrap(), ConstraintType::Max),
        (Regex::new(r"(?i)at\s+least\s+(\d+)\s+(\w+)").unwrap(), ConstraintType::Min),
        (Regex::new(r"(?i)no\s+more\s+than\s+(\d+)\s+(\w+)").unwrap(), ConstraintType::Max),
        (Regex::new(r"(?i)no\s+less\s+than\s+(\d+)\s+(\w+)").unwrap(), ConstraintType::Min),
    ];

    for (pattern, ctype) in &patterns {
        for cap in pattern.captures_iter(text) {
            let value: f64 = cap.get(1).and_then(|m| m.as_str().parse().ok()).unwrap_or(0.0);
            let target = cap.get(2).map(|m| m.as_str().to_string()).unwrap_or_default();
            atoms.constraints.push(ExtractedConstraint {
                constraint_type: ctype.clone(),
                value: ConstraintValue::Single(value),
                target,
                original: cap.get(0).map(|m| m.as_str().to_string()).unwrap_or_default(),
            });
        }
    }

    // Range pattern
    let range_pattern = Regex::new(r"(?i)between\s+(\d+)\s+and\s+(\d+)\s+(\w+)").unwrap();
    for cap in range_pattern.captures_iter(text) {
        let low: f64 = cap.get(1).and_then(|m| m.as_str().parse().ok()).unwrap_or(0.0);
        let high: f64 = cap.get(2).and_then(|m| m.as_str().parse().ok()).unwrap_or(0.0);
        let target = cap.get(3).map(|m| m.as_str().to_string()).unwrap_or_default();
        atoms.constraints.push(ExtractedConstraint {
            constraint_type: ConstraintType::Range,
            value: ConstraintValue::Range(low, high),
            target,
            original: cap.get(0).map(|m| m.as_str().to_string()).unwrap_or_default(),
        });
    }
}

fn extract_orderings(text: &str, atoms: &mut LogicalAtoms) {
    let patterns: Vec<(Regex, OrderingType)> = vec![
        (Regex::new(r"(?i)(\w+(?:\s+\w+)?)\s+before\s+(\w+(?:\s+\w+)?)").unwrap(), OrderingType::Before),
        (Regex::new(r"(?i)(\w+(?:\s+\w+)?)\s+after\s+(\w+(?:\s+\w+)?)").unwrap(), OrderingType::After),
        (Regex::new(r"(?i)first\s+(\w+(?:\s+\w+)?)\s*,?\s*then\s+(\w+(?:\s+\w+)?)").unwrap(), OrderingType::Before),
        (Regex::new(r"(?i)(\w+(?:\s+\w+)?)\s+until\s+(\w+(?:\s+\w+)?)").unwrap(), OrderingType::Until),
    ];

    for (pattern, otype) in &patterns {
        for cap in pattern.captures_iter(text) {
            atoms.orderings.push(ExtractedOrdering {
                ordering_type: otype.clone(),
                first: cap.get(1).map(|m| m.as_str().to_string()).unwrap_or_default(),
                second: cap.get(2).map(|m| m.as_str().to_string()).unwrap_or_default(),
                original: cap.get(0).map(|m| m.as_str().to_string()).unwrap_or_default(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_quantifiers() {
        let atoms = extract_logical_atoms_sync("Delete all users from the database");
        assert!(!atoms.quantifiers.is_empty());
        assert_eq!(atoms.quantifiers[0].quantifier_type, QuantifierType::All);
    }

    #[test]
    fn test_extract_negations() {
        let atoms = extract_logical_atoms_sync("Do not delete any records");
        assert!(!atoms.negations.is_empty());
        assert!(atoms.negations[0].negated);
    }

    #[test]
    fn test_extract_numbers() {
        let atoms = extract_logical_atoms_sync("timeout: 30s with maximum 5 retries");
        assert!(!atoms.numbers.is_empty());
    }

    #[test]
    fn test_extract_constraints() {
        let atoms = extract_logical_atoms_sync("Process at most 100 records per batch");
        assert!(!atoms.constraints.is_empty());
        assert_eq!(atoms.constraints[0].constraint_type, ConstraintType::Max);
    }

    #[test]
    fn test_extract_orderings() {
        let atoms = extract_logical_atoms_sync("validate input before processing");
        assert!(!atoms.orderings.is_empty());
        assert_eq!(atoms.orderings[0].ordering_type, OrderingType::Before);
    }
}
