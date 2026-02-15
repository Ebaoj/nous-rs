use super::types::*;

/// Result of comparing two sets of logical atoms
#[derive(Debug, Clone)]
pub struct AtomComparisonResult {
    pub compatible: bool,
    pub conflicts: Vec<String>,
    pub details: AtomComparisonDetails,
}

/// Per-category match details
#[derive(Debug, Clone)]
pub struct AtomComparisonDetails {
    pub quantifier_match: bool,
    pub negation_match: bool,
    pub number_match: bool,
    pub constraint_match: bool,
    pub ordering_match: bool,
}

/// Compare two sets of logical atoms for compatibility.
/// Returns true if atoms are logically compatible.
pub fn compare_logical_atoms(a: &LogicalAtoms, b: &LogicalAtoms) -> AtomComparisonResult {
    let mut conflicts = Vec::new();

    let quantifier_match = compare_quantifiers(&a.quantifiers, &b.quantifiers, &mut conflicts);
    let negation_match = compare_negations(&a.negations, &b.negations, &mut conflicts);
    let number_match = compare_numbers(&a.numbers, &b.numbers, &mut conflicts);
    let constraint_match = compare_constraints(&a.constraints, &b.constraints, &mut conflicts);
    let ordering_match = compare_orderings(&a.orderings, &b.orderings, &mut conflicts);

    let compatible = quantifier_match && negation_match && number_match && constraint_match && ordering_match;

    AtomComparisonResult {
        compatible,
        conflicts,
        details: AtomComparisonDetails {
            quantifier_match,
            negation_match,
            number_match,
            constraint_match,
            ordering_match,
        },
    }
}

fn words_overlap(a: &str, b: &str) -> bool {
    let lower_a = a.to_lowercase();
    let lower_b = b.to_lowercase();
    let words_a: Vec<&str> = lower_a.split_whitespace().collect();
    let words_b: Vec<&str> = lower_b.split_whitespace().collect();
    words_a.iter().any(|w| words_b.contains(w))
}

fn is_quantifier_conflict(a: &QuantifierType, b: &QuantifierType) -> bool {
    use QuantifierType::*;
    let pairs: &[(QuantifierType, QuantifierType)] = &[
        (All, None), (All, Some), (All, Few),
        (None, Some), (None, Most),
    ];
    pairs.iter().any(|(x, y)| {
        (a == x && b == y) || (a == y && b == x)
    })
}

fn compare_quantifiers(a: &[ExtractedQuantifier], b: &[ExtractedQuantifier], conflicts: &mut Vec<String>) -> bool {
    for qa in a {
        for qb in b {
            if words_overlap(&qa.scope, &qb.scope) && qa.quantifier_type != qb.quantifier_type
                && is_quantifier_conflict(&qa.quantifier_type, &qb.quantifier_type) {
                    conflicts.push(format!(
                        "Quantifier conflict: \"{:?} {}\" vs \"{:?} {}\"",
                        qa.quantifier_type, qa.scope, qb.quantifier_type, qb.scope
                    ));
                    return false;
                }
        }
    }
    true
}

fn compare_negations(a: &[ExtractedNegation], b: &[ExtractedNegation], conflicts: &mut Vec<String>) -> bool {
    // Both have negations - compare
    for na in a {
        for nb in b {
            if words_overlap(&na.target, &nb.target) && na.negated != nb.negated {
                conflicts.push(format!(
                    "Negation conflict: \"{}\" vs \"{}\"",
                    na.original, nb.original
                ));
                return false;
            }
        }
    }

    // Asymmetric negation
    if !a.is_empty() && b.is_empty() {
        for na in a {
            if na.negated {
                conflicts.push(format!(
                    "Negation asymmetry: A negates \"{}\" but B doesn't", na.target
                ));
                return false;
            }
        }
    }

    if !b.is_empty() && a.is_empty() {
        for nb in b {
            if nb.negated {
                conflicts.push(format!(
                    "Negation asymmetry: B negates \"{}\" but A doesn't", nb.target
                ));
                return false;
            }
        }
    }

    true
}

fn compare_numbers(a: &[ExtractedNumber], b: &[ExtractedNumber], conflicts: &mut Vec<String>) -> bool {
    for na in a {
        for nb in b {
            let name_match = na.name.to_lowercase() == nb.name.to_lowercase()
                || (na.unit == nb.unit && na.name.contains(&nb.name));

            if name_match && (na.value - nb.value).abs() > f64::EPSILON {
                let max_val = na.value.abs().max(nb.value.abs());
                let relative_diff = if max_val > 0.0 {
                    (na.value - nb.value).abs() / max_val
                } else {
                    (na.value - nb.value).abs()
                };

                if relative_diff > 0.01 {
                    let ua = na.unit.as_deref().unwrap_or("");
                    let ub = nb.unit.as_deref().unwrap_or("");
                    conflicts.push(format!(
                        "Number conflict: {}={}{} vs {}={}{}",
                        na.name, na.value, ua, nb.name, nb.value, ub
                    ));
                    return false;
                }
            }
        }
    }
    true
}

fn compare_constraints(a: &[ExtractedConstraint], b: &[ExtractedConstraint], conflicts: &mut Vec<String>) -> bool {
    for ca in a {
        for cb in b {
            let target_a = ca.target.to_lowercase();
            let target_b = cb.target.to_lowercase();
            if !(target_a.contains(&target_b) || target_b.contains(&target_a)) {
                continue;
            }

            // max vs min conflict
            if ca.constraint_type == ConstraintType::Max && cb.constraint_type == ConstraintType::Min {
                if let (ConstraintValue::Single(max_v), ConstraintValue::Single(min_v)) = (&ca.value, &cb.value) {
                    if max_v < min_v {
                        conflicts.push(format!(
                            "Constraint conflict: max {}={} but min {}={}",
                            ca.target, max_v, cb.target, min_v
                        ));
                        return false;
                    }
                }
            }

            // Same type, different values
            if ca.constraint_type == cb.constraint_type {
                if let (ConstraintValue::Single(va), ConstraintValue::Single(vb)) = (&ca.value, &cb.value) {
                    if (va - vb).abs() > f64::EPSILON {
                        conflicts.push(format!(
                            "Constraint conflict: {:?} {}={} vs {}",
                            ca.constraint_type, ca.target, va, vb
                        ));
                        return false;
                    }
                }
            }
        }
    }
    true
}

fn compare_orderings(a: &[ExtractedOrdering], b: &[ExtractedOrdering], conflicts: &mut Vec<String>) -> bool {
    for oa in a {
        for ob in b {
            if oa.ordering_type == OrderingType::Before && ob.ordering_type == OrderingType::After {
                let a_first = oa.first.to_lowercase();
                let a_second = oa.second.to_lowercase();
                let b_first = ob.first.to_lowercase();
                let b_second = ob.second.to_lowercase();

                if a_first.contains(&b_second) && a_second.contains(&b_first) {
                    conflicts.push(format!(
                        "Ordering conflict: \"{} before {}\" vs \"{} after {}\"",
                        oa.first, oa.second, ob.first, ob.second
                    ));
                    return false;
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logical_atoms::extractor::extract_logical_atoms_sync;

    #[test]
    fn test_quantifier_conflict() {
        let a = extract_logical_atoms_sync("Delete all users");
        let b = extract_logical_atoms_sync("Delete some users");
        let result = compare_logical_atoms(&a, &b);
        assert!(!result.compatible);
        assert!(!result.conflicts.is_empty());
    }

    #[test]
    fn test_compatible_atoms() {
        let a = extract_logical_atoms_sync("Process all records");
        let b = extract_logical_atoms_sync("Handle all records");
        let result = compare_logical_atoms(&a, &b);
        assert!(result.compatible);
    }

    #[test]
    fn test_negation_conflict() {
        let a = extract_logical_atoms_sync("Do not delete records");
        let b = extract_logical_atoms_sync("Delete some records");
        // a has negation on "delete", b has quantifier on what to delete
        // This tests negation asymmetry
        let result = compare_logical_atoms(&a, &b);
        assert!(!result.compatible);
    }
}
